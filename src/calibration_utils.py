# pnp计算工具
"""
    目的是根据npz文件里的相机内参矩阵，和相机畸变系数，计算出相机的平移矩阵和旋转向量
    solvepnp算法需要的就是相机内参矩阵和相机畸变系数
"""
import cv2
import numpy as np
import refactor_utils as utils

class CalibrationProcessor:
    def __init__(self,camera_params_path="camera_params.npz",checkerboard_size=(9,6),checkerboard_gap=1.0):
        self.checkerboard = checkerboard_size
        self.square_size = checkerboard_gap # 设定好棋盘格在现实世界中的实际距离，做理想针孔相机映射
        
        # 加载相机内参
        
        data = np.load(camera_params_path) # 加载字典 camera_matrix, dist_coeffs, rvecs, tvecs, calibration_error
        self.mtx = data['camera_matrix'] # 拿相机内参矩阵，焦距和光心
        self.dist = data['dist_coeffs'] # 拿相机畸变系数，镜头的弯曲程度，复习自https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
        # 生成世界坐标点，3D坐标点，z坐标为0，以棋盘格为坐标系，x和y坐标根据棋盘格的大小和格子实际距离生成
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1],3),np.float32)# 为啥是二维？因为只要每组有xyz就行了，所以是(9*6，3)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)# 将棋盘格的坐标点生成出来，mgrid生成一个网格，reshape成(54,2)，每行是一个点的x和y坐标，z坐标默认为0
        self.objp *= self.square_size # 乘以格子实际大小(比如1cm)

    # @staticmethod
    # def load_camera_parmas(camera_proams_path: Path):
    
    def solve_pnp(self, image_points,refine = True):
        # image_points是2D像素坐标点，objp是3D世界坐标点，mtx是相机内参矩阵，dist是相机畸变系数,flags是求解pnp问题的算法选择，
        success, R, t = cv2.solvePnP(self.objp, image_points, self.mtx, self.dist, flags=cv2.SOLVEPNP_IPPE) # 选择IPPE算法，适用于平面对象，提供更稳定的结果，返回三个参数：是否成功，旋转向量和平移向量
        if not success:
            raise ValueError("PnP求解失败，可能是因为输入的图像点不够准确或数量不足。")
        if refine:
            R, t = cv2.solvePnPRefineLM(self.objp, image_points, self.mtx, self.dist, R, t) # 选择LM算法，非线性优化方法，进一步优化R和t的结果,返回两个参数
        return R, t

    def rvec_to_euler_zxy_deg(self,rvec):
        R,_ = cv2.Rodrigues(rvec) # 将旋转向量转换为旋转矩阵，返回一个3x3的旋转矩阵R和一个雅可比矩阵（不需要）
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2) # 计算sy，判断xy方向是否接近于零使得与z轴对齐，避免万向锁问题,万向节锁的解释：当某个旋转轴的旋转角度接近于90度时，可能会导致两个旋转轴重合，从而失去一个自由度，无法区分两个旋转轴的旋转，这时就会出现万向节锁问题。
        singular = sy < 1e-6 # 判断是否接近于零，设置一个小的阈值
        if not singular:
            yaw = np.arctan2(R[2,1], R[2,2]) # 计算yaw角，绕z轴旋转
            pitch = np.arctan2(-R[2,0], sy) # 计算pitch角，绕y轴旋转
            roll = np.arctan2(R[1,0], R[0,0]) # 计算roll角，绕x轴旋转
        else:
            yaw = np.arctan2(-R[1,2], R[1,1]) # 计算yaw角，绕z轴旋转
            pitch = np.arctan2(-R[2,0], sy) # 计算pitch角，绕y轴旋转
            roll = 0 # 当接近于零时，roll角设置为0，因为无法区分两个旋转轴的旋转
        return np.degrees([yaw, pitch, roll]) # 将弧度转换为角度，返回一个包含yaw、pitch和roll角度的数组
    
    def reprojection_rmse(self, image_points, R, t):
        # 将3D世界坐标点投影回2D像素坐标点，计算重投影误差,测试pnp求解的准确性
        image_points = np.asarray(image_points,dtype=np.float32).reshape(-1,1,2) # 强制转换成np浮点数的数组，点的数量自动推断，点的排列方式
        proj,_ = cv2.projectPoints(self.objp, R, t, self.mtx, self.dist) # 将3D世界坐标点投影回2D像素坐标点，返回一个包含投影点坐标的数组和一个雅可比矩阵（不需要）
        error = np.linalg.norm(image_points - proj) / np.sqrt(len(image_points)) # 计算重投影误差，使用欧几里得距离，除以点的数量得到平均误差
        return error
    
    def two_pic_position_detection(self):
        img = utils.chiose_image(index=1)
        img2 = utils.chiose_image(index=2)
        _ ,r,t = utils.find_chessboard_corners(img,chessboard_size=self.checkerboard,visualize=False)
        _ ,r2,t2 = utils.find_chessboard_corners(img2,chessboard_size=self.checkerboard,visualize=False)
        r, t = self.solve_pnp(r)
        r2, t2 = self.solve_pnp(r2)
        err = np.linalg.norm(t-t2) # 计算两个位置的平移向量之间的距离，作为位置检测的误差
        r = self.rvec_to_euler_zxy_deg(r)# 将旋转向量转换为欧拉角，便于理解和比较
        r2 = self.rvec_to_euler_zxy_deg(r2)
        re_Yerr = abs(r[0]-r2[0]) # 计算yaw角的误差，单位为度
        re_Perr = abs(r[1]-r2[1]) # 计算pitch角的误差，单位为度
        re_Rerr = abs(r[2]-r2[2]) # 计算roll角的误差，单位为度
        re_rror = np.sqrt(re_Yerr**2 + re_Perr**2 + re_Rerr**2) # 计算总的旋转误差，使用欧几里得距离，单位为度
        print("第一张图的旋转角度（yaw, pitch, roll）：",r) # 打印第一张图的旋转角度，单位为度
        print("第二张图的旋转角度（yaw, pitch, roll）：",r2) # 打印第二张图的旋转角度，单位为度
        print("旋转检测误差：",re_rror)# 打印旋转检测误差和平移检测误差，评估位置检测的准确性，单位为度和距离单位（比如厘米，看棋盘格每格的单位）
        print("位置检测误差：",err)
    
    
            
if __name__ == "__main__":
    caliuitls = CalibrationProcessor("camera_params.npz",checkerboard_size=(9,6),checkerboard_gap=1.0)
    caliuitls.two_pic_position_detection()
    
        