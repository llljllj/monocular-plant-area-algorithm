import cv2
import numpy as np
import matplotlib.pyplot as plt
import sam3_triton
sam3_triton._ensure_triton_stub()
from refactor_utils import chiose_image
from restore_algorithm_utils import restore_algorithm
from sam3_utils import SAM3Utils


if __name__ == "__main__":
    img1 = chiose_image()
    img2 = chiose_image()

    model = SAM3Utils(model="D:\personal_project\peanut_detect\model\sam3.pt", load_from_HF=True)
    restorer = restore_algorithm(model=model,prompt="plant") # 提示词建议用英文 "plant"，因为原模型可能不支持中文提示词
    area1,area2,restored_img2 = restorer.reailty_maskarea_restroe(img1, img2, precision="float16")
    print(f"第一张图的面积为：{area1}，第二张图的还原面积为：{area2}")
    plt.imshow(restored_img2)
    plt.show()