from tkinter  import Tk,filedialog

import cv2

import numpy as np

root = Tk()
root.withdraw() #隐藏主窗口



# 用中英文方法读取图片，兼容不同系统和路径格式
def read_image(image_path):
    try:
        img_data = np.fromfile(image_path,dtype=np.uint8)
        img = cv2.imdecode(img_data,cv2.IMREAD_COLOR)

        if img is None:
            img = cv2.imread(image_path)

        if img is None:
            print(f"读取失败{image_path}")

        print(f"成功读取{image_path}")
        return img
    except Exception as e:
        print(f"读取失败{image_path}，错误信息：{e.__class__.__name__}: {e}")
        return None

# 选择图片文件，调用read_image函数读取图片
def chiose_image(index = 1):
    file_path = filedialog.askopenfilename(
        title=f"选择第{index}张图片",
        filetypes=[
            ("图片文件", "*.jpg *jpeg *.png *bmp"),
            ("所有文件", "*.*")
        ]
    )

    if not file_path:
        print("未选择任何程序")
        return False,None
    
    print(f"用户选择了：{file_path}")

    img = read_image(file_path)
    
    return img

    
# 查找2d像素棋盘格角点+可视化
def find_chessboard_corners(img,chessboard_size=(9,6),visualize = True):
    if img is None:
        return False, None, None # 固定返回三元组：是否找到，角点返回，图片返回
    
    #BGR = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2BGR)# 若单通道BGR,用bayer重建原始图片的rgb值，精确灰度值
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# 转灰度图

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH #二值化 防止传rgb
    + cv2.CALIB_CB_NORMALIZE_IMAGE# 直方图归一，增强对比度，防止弱光图
    + cv2.CALIB_CB_FAST_CHECK)# 加速检测
    _ , corners = cv2.findChessboardCorners(gray,patternSize=chessboard_size,flags=flags)

    if _:
        criteria = (cv2.TERM_CRITERIA_EPS +
                     cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)# 迭代优化算法次数，和位置变化阈值（像素单位，拟合小于阈值说明收敛+结束算法）
        corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)# 亚像素精度算法，不考虑死区
        for i , corner in enumerate(corners):
            print(f"找到第{i}个角点，位置{corner}")
        #可视化
        if visualize:
            img_vis = img.copy()
            cv2.drawChessboardCorners(img_vis,chessboard_size,corners,_)

            h,w = img_vis.shape[:2]
            scale = min(800/w,600/h) # 调大小

            if scale < 1:
                img_vis = cv2.resize(img_vis,None,fx=scale,fy=scale)

            cv2.imshow("corner_img",img_vis)
            cv2.waitKey(3000)


    return _ , corners , img

# 计算单应矩阵的函数和将像素坐标转换为世界坐标的函数，使用单应矩阵将第二帧图像变换到第一帧图像的视角的函数，以及一个测试函数来验证位置检测的准确性。
def find_H_world_from_img(corners,checkerboard_size=(9,6),checkerboard_gap=1.0):
    objp = np.zeros((checkerboard_size[0]*checkerboard_size[1],3),dtype=np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0],0:checkerboard_size[1]].T.reshape(-1,2)*checkerboard_gap
    H_world_from_img, _ = cv2.findHomography(corners.reshape(-1,2), objp[:,:2]) # 计算单应矩阵，使用像素坐标和世界坐标的对应关系
    return H_world_from_img
    

if __name__ == "__main__":
    img = chiose_image()
    find_chessboard_corners(img)