import cv2 # 导入 OpenCV 库，用于图像处理和计算机视觉任务
import numpy as np # 导入 NumPy 库，并简写为 np，用于高效的矩阵和数组计算

from calibration_utils import CalibrationProcessor # 从自定义的 calibration_utils 模块中导入 CalibrationProcessor 类，用于处理相机标定
from refactor_utils import chiose_image, find_H_world_from_img, find_chessboard_corners # 从 refactor_utils 模块导入图像选择、计算世界单应矩阵、查找棋盘格角点等辅助函数

class restore_algorithm: # 定义一个名为 restore_algorithm 的类，封装视角还原和面积计算的算法
    def __init__(self, model, prompt, threshold=0.5): # 类的初始化方法，接收分割模型、提示词和可选的置信度阈值作为参数
        self.model = model # 将传入的分割模型实例保存为类的属性，用于后续图像预测
        self.prompt = prompt # 保存模型预测时需要的提示词（如“植物”、“叶片”等）
        self.threshold = threshold # 保存掩码的置信度阈值，默认为 0.5（目前代码中暂未直接使用）

        self.calibrator = CalibrationProcessor() # 实例化 CalibrationProcessor 标定处理器类，用于获取相机参数
        self.K = self.calibrator.mtx # 获取相机的内参矩阵（Camera Intrinsic Matrix），用于构建单应矩阵和 PnP 解算
        self.checkerboard = self.calibrator.checkerboard  # 获取棋盘格的内角点规格（如 (9, 6) 等）
        self.square_size = self.calibrator.square_size # 获取棋盘格单个方格的实际物理尺寸（例如 25 毫米等）

    def return_poses(self, first_image, second_image, visualize=True): # 定义方法计算两帧图像中相机的位姿（旋转和平移向量）
        ok1, corners1, _ = find_chessboard_corners( # 调用函数在第一张图寻找棋盘格角点
            first_image, chessboard_size=self.checkerboard, visualize=visualize) # 传入第一张图片、棋盘格尺寸，并指定是否可视化
        ok2, corners2, _ = find_chessboard_corners( # 调用函数在第二张图寻找棋盘格角点
            second_image, chessboard_size=self.checkerboard, visualize=visualize) # 传入第二张图片、棋盘格尺寸，并指定是否可视化

        if not ok1 or corners1 is None: # 检查第一张图是否成功找到了棋盘格角点
            raise ValueError("第一张图未检测到棋盘格角点。") # 如果没有找到，抛出异常，中断程序
        if not ok2 or corners2 is None: # 检查第二张图是否成功找到了棋盘格角点
            raise ValueError("第二张图未检测到棋盘格角点。") # 如果没有找到，抛出异常，中断程序

        rvec1, tvec1 = self.calibrator.solve_pnp(corners1) # 用第一张图的角点，通过 PnP 算法求解相机在该图的旋转向量 (rvec1) 和平移向量 (tvec1)
        rvec2, tvec2 = self.calibrator.solve_pnp(corners2) # 用第二张图的角点，通过 PnP 算法求解相机在该图的旋转向量 (rvec2) 和平移向量 (tvec2)

        return rvec1, tvec1, rvec2, tvec2, corners1, corners2 # 返回两张图各自的旋转向量、平移向量，以及像素坐标下的角点集合
    
    def homography_from_pose(self, K, rvec, tvec): # 定义方法，根据相机的内参、旋转向量和平移向量计算棋盘格平面到图像平面的单应矩阵 (Homography)
        if rvec is None or tvec is None: # 首先检查传入的旋转或平移向量是否为空
            raise ValueError("Rotation vector and translation vector cannot be None.") # 若为空则抛出异常，提醒输入不能为 None
        R, _ = cv2.Rodrigues(rvec) # 将旋转向量 (3x1) 通过罗德里格斯变换转换为旋转矩阵 (3x3)
        H = K @ np.column_stack([R[:, 0], R[:, 1], tvec.reshape(3)]) # 将内参矩阵 K 与 (旋转矩阵前两列, 平移向量) 组合成的 3x3 矩阵相乘，得到从世界 z=0 平面到图像的单应矩阵 H
        return H # 返回计算得到的单应矩阵 H
    
    def transform_pose(self, first_image, second_image): # 定义方法，将第二帧图像根据地平面（棋盘格）透视变形，对齐到第一帧图像的视角
        # 修正：匹配返回的 6 个参数。调用 return_poses 获取两张图的旋转量 r、平移量 t，并忽略掉返回的角点数据
        r1, t1, r2, t2, _, _ = self.return_poses(first_image, second_image) # 解包获取相机基于世界坐标系的位姿

        H1 = self.homography_from_pose(self.K, r1, t1) # 计算世界平面铺设到第一张图像平面的单应矩阵 H1
        H2 = self.homography_from_pose(self.K, r2, t2) # 计算世界平面铺设到第二张图像平面的单应矩阵 H2

        H_1_from_2 = H1 @ np.linalg.inv(H2) # 计算第二张图到第一张图的透视变换矩阵。先经过 H2 的逆（回到物理平面），再通过 H1（映射到第一张图）
        H_1_from_2 /= H_1_from_2[2, 2] # 对求算出来的综合单应矩阵进行归一化（使矩阵右下角元素为1，避免数值不稳定）

        h, w = first_image.shape[:2] # 提取第一张图的高度 (h) 和宽度 (w)
        restored = cv2.warpPerspective(second_image, H_1_from_2, (w, h)) # 执行仿射透视变换，把第二帧图像扭曲并铺满到与第一张图相同的尺寸上，实现视角贴合
        return restored # 返回视角被强行扭曲/还原处理过后的第二帧图像
    
    @staticmethod  # 声明此方法是一个静态方法，因为它并未调用属于实例 self 的专有属性
    def contour_maskarea_world(contour_px, H_world_from_img): # 定义方法，将图像中的轮廓像素坐标转换为真实物理世界坐标，并计算其在物理世界中的现实面积
        pts = contour_px.reshape(-1, 2).astype(np.float32) # 将输入的轮廓数组拉平为 Nx2 的形状，并转化为 NumPy 的 32 位浮点型数组，代表 (x, y) 像素坐标
        ones = np.ones((pts.shape[0],1), dtype=np.float32) # 创建一个形状为 Nx1 的全 1 数组，作为齐次坐标的第三项
        pts_h = np.hstack([pts,ones]) # 沿着水平方向将 (x, y) 与全 1 数组拼接，得到 Nx3 的齐次坐标数组 (x, y, 1)

        pts_world_h = (H_world_from_img @ pts_h.T).T # 利用输入图像到世界平面的单应矩阵，将像素的齐次坐标系转换为物理世界下的齐次坐标 (.T 表示转置用于矩阵对齐)
        pts_world_xy = pts_world_h[:, :2] / pts_world_h[:, 2:3] # 对计算出的世界齐次坐标除以第三维度（深度比例），降维得到位于 z=0 平面上的真实三维空间之 X、Y 物理坐标

        x = pts_world_xy[:, 0] # 提取所有物理边界点的 X 坐标集合
        y = pts_world_xy[:, 1] # 提取所有物理边界点的 Y 坐标集合
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) # 运用鞋带公式（Shoelace formula），利用叉乘原理计算这组多边形顶点围成的二维真实多边形面积

        return area, pts_world_xy # 函数返回计算得出的真实物理面积值，以及转换后的多边形物理坐标点集

    def reailty_maskarea_restroe(self, first_image, second_image): # 定义一个流水线主调方法：先提取位姿 -> 求算掩码 -> 视角对齐 -> 后获取各自的真实物理掩码面积
        # 修正：参数解包以及调用时的变量名。调用 return_poses，这里不仅获取位移，还取出了图像中的角点坐标（conner1, conner2）
        _, _, _, _, conner1, conner2 = self.return_poses(first_image, second_image) # 用下划线忽略丢弃不需要的位姿，保留两个角点集
        
        # 修正：使用自有的 checkerboard 等属性。调用 refactor_utils 中的方法，根据图片中的角点及尺寸求出图像像素坐标系到物理世界二维平面的映射单应矩阵 
        H_world_from_img1 = find_H_world_from_img(conner1, checkerboard_size=self.checkerboard, checkerboard_gap=self.square_size) # 获取第一帧图素到真实平面的对应变换矩阵
        H_world_from_img2 = find_H_world_from_img(conner2, checkerboard_size=self.checkerboard, checkerboard_gap=self.square_size) # 获取第二帧图素到真实平面的对应变换矩阵

        mask1 = self.model.predict(first_image, self.prompt) # 调用外部模型对象，传入第一帧原图和提示语，预测出相关的目标像素轮廓/分割掩码
        mask2 = self.model.predict(second_image, self.prompt) # 对第二帧执行相同操作，预测目标在第二帧图像上的像素级掩码

        restore = self.transform_pose(first_image, second_image) # 调用类方法，基于平面映射假定，强行将第二帧的图片视角形变拉扯至第一帧相机的视角维度下

        area1, _ = self.contour_maskarea_world(mask1, H_world_from_img1) # 拿着第一帧得到的分割掩码数据及它的“图像-世界”单应变换矩阵，计算还原出它真实的物理面积
        area2, _ = self.contour_maskarea_world(mask2, H_world_from_img2) # 对第二帧做一样的事情，获取第二帧掩码面积系在物理空间下映射出来的真实大小
        
        # 记得返回你需要的数据
        return area1, area2, restore # 最终将第一和第二张图算出的真实物理测量面积以致对齐合成后的图片返回，供外部保存打印和展示分析使用