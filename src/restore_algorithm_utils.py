
import cv2

from calibration_utils import CalibrationProcessor
from refactor_utils import chiose_image, find_H_world_from_img,find_chessboard_corners

class restore_algorithm:
    def __init__(self,model,prompt,threshold=0.5):
        self.model=model
        self.prompt=prompt
        self.threshold=threshold

        self.calibrator = CalibrationProcessor()
        self.K = self.calibrator.mtx
        self.chechkerboard = self.calibrator.checkerboard
        self.square_size = self.calibrator.square_size


    # 查找两张图片的棋盘格角点，获取摄像头位姿
    def return_poses(self, first_image, second_image, visualize=True):
        ok1, corners1, _ = find_chessboard_corners(
            first_image,
            chessboard_size=self.checkerboard,
            visualize=visualize
        )
        ok2, corners2, _ = find_chessboard_corners(
            second_image,
            chessboard_size=self.checkerboard,
            visualize=visualize
        )

        if not ok1 or corners1 is None:
            raise ValueError("第一张图未检测到棋盘格角点。")
        if not ok2 or corners2 is None:
            raise ValueError("第二张图未检测到棋盘格角点。")

        rvec1, tvec1 = self.calibrator.solve_pnp(corners1)
        rvec2, tvec2 = self.calibrator.solve_pnp(corners2)

        return rvec1, tvec1, rvec2, tvec2, corners1, corners2
    
    # 将旋转向量和位移向量转换为单应矩阵，使用单应矩阵将第二帧图像变换到第一帧图像的视角
    def homography_from_pose(self, K, rvec, tvec):
        if rvec is None or tvec is None:
            raise ValueError("Rotation vector and translation vector cannot be None.")
        R, _ = cv2.Rodrigues(rvec)# 将旋转向量转换为旋转矩阵
        H = K @ np.column_stack([R[:, 0], R[:, 1], tvec.reshape(3)])# 构建单应矩阵，使用旋转矩阵的前两列和位移向量
        return H # 返回单应矩阵，表示从世界坐标到图像坐标的变换关系，可以用于将第二帧图像变换到第一帧图像的视角
    

    #将第二帧图像恢复到第一帧图像的视角，实现图像的对齐和融合，使得两帧图像能够在同一视角下进行比较和分析，提升图像处理的效果和准确性。   
    def transform_pose(self,first_image,second_image):
        R1,t1,R2,t2 = self.return_poses(first_image,second_image)# 拿到两帧图像的位姿

        # 已有: rvec1, tvec1, rvec2, tvec2, K
        H1 = self.homography_from_pose(self.K, R1, t1)# 根据第一帧图像的位姿计算单应矩阵
        H2 = self.homography_from_pose(self.K, R2, t2)# 根据第二帧图像的位姿计算单应矩阵

        H_1_from_2 = H1 @ np.linalg.inv(H2)# 计算从第二帧图像到第一帧图像的单应矩阵
        H_1_from_2 /= H_1_from_2[2, 2]# 归一化单应矩阵

        h, w = first_image.shape[:2]# 获取第一帧图像的尺寸
        restored = cv2.warpPerspective(second_image, H_1_from_2, (w, h))# 使用单应矩阵将第二帧图像变换到第一帧图像的视角
        return restored
    

    def contour_maskarea_world(contour_px,H_world_from_img):
        pts = contour_px.reshape(-1, 2).astype(np.float32) # 将轮廓点转换为二维坐标

        # 将二维像素坐标转化为齐次坐标，方便进行矩阵运算
        ones = np.ones((pts.shape[0],1),dtype = np.float32)# 在每个点的坐标后面添加一个1，形成齐次坐标
        pts_h = np.hstack([pts,ones])# 将二维坐标和齐次坐标合并成一个新的数组

        # 使用单应矩阵将像素坐标转换为世界坐标
        pts_world_h = (H_world_from_img @ pts_h.T).T# 将齐次坐标乘以单应矩阵，得到世界坐标的齐次表示
        pts_world_xy = pts_world_h[:, :2] / pts_world_h[:, 2:3]# 将齐次坐标转换为二维坐标，除以最后一个元素

        # 鞋带公式计算多边形的面积
        x = pts_world_xy[:, 0]# 获取世界坐标的x坐标
        y = pts_world_xy[:, 1]# 获取世界坐标的y坐标
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))# 使用鞋带公式计算多边形的面积

        return area,pts_world_xy

    def reailty_maskarea_restroe(self,first_image,second_image):
        R1, t1, R2, t2 = self.return_poses(first_image,second_image)# 获取第二帧图像的位姿
        conner1 = find_chessboard_corners(first_image, chessboard_size=calibration_processor.checkerboard)
        conner2 = find_chessboard_corners(second_image, chessboard_size=calibration_processor.checkerboard)
        H_world_from_img1 = find_H_world_from_img(conner1[1], checkerboard_size=calibration_processor.checkerboard, checkerboard_gap=calibration_processor.checkerboard_gap)# 计算从第一帧图像到世界坐标的单应矩阵
        H_world_from_img2 = find_H_world_from_img(conner2[1], checkerboard_size=calibration_processor.checkerboard, checkerboard_gap=calibration_processor.checkerboard_gap)# 计算从第二帧图像到世界坐标的单应矩阵

        mask1 = self.model.predict(first_image, self.prompt)# 使用模型预测第一帧图像的掩码
        mask2 = self.model.predict(second_image, self.prompt)# 使用模型预测第二帧图像的掩码

        restore = self.transform_pose(first_image,second_image)# 将第二帧图像恢复到第一帧图像的视角

        area1, _ = self.contour_maskarea_world(mask1, H_world_from_img1)# 计算第一帧图像的掩码在世界坐标下的面积
        area2, _ = self.contour_maskarea_world(mask2, H_world_from_img2)# 计算第二帧图像的掩码在世界坐标下的面积
        
        


