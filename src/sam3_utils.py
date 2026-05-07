from torch import export
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from PIL import Image  
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sam3.visualization_utils import plot_results

@torch.inference_mode()
class SAM3Utils:
    def __init__(self,model,load_from_HF=False):
        model = build_sam3_image_model(
            device= "cuda" if torch.cuda.is_available() else "cpu",
            checkpoint_path=model,
            load_from_HF=load_from_HF
        )
        self.model = model
        self.model.eval()

        # if load_from_HF: 错误点，SAM官方提供的代码写死了把图片强转为float32，所以我们这里不能强转，用autocast，pytorch会自动根据模型权重的类型来选择合适的计算精度（FP16或FP32），以实现更高效的推理性能。
        #     print("使用半精度推理，需要强制修改FP32为FP16")
        #     self.model.to(torch.float16)

        self.processor = Sam3Processor(model, device = "cuda" if torch.cuda.is_available() else "cpu",confidence_threshold = 0.5)
    @torch.inference_mode()
    def predict(self,image,prompt):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        # 【重要修复】如果输入是 OpenCV 格式 (NumPy ndarray)，需先在此转换为 PIL Image 提供给 processor。
        # 因为 processor.set_image 内部期望处理的是 PIL.Image 类型的对象，否则它可能无法正确提取特征，导致后续无法分割出物体。
        if isinstance(image, np.ndarray):
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(img_rgb)
        else:
            image_pil = Image.open(image)

        with torch.autocast(device_type): # 使用torch.autocast上下文管理器来自动选择适当的计算精度（如FP16或FP32），以提高推理效率，同时保持数值稳定性。
            image_inference_state = self.processor.set_image(image_pil)# 将图像输入到处理器中，set_image函数会将图像转换为模型所需的格式，并将其存储在处理器的内部状态中，以便后续的文本提示和推理操作使用。
            self.processor.reset_all_prompts(image_inference_state)# 重置处理器中的所有提示，这通常在每次新的推理开始时调用，以确保之前的提示不会影响当前的推理结果。这个函数会清除处理器内部存储的所有文本提示和相关状态，使得处理器处于一个干净的状态，准备接受新的提示和进行新的推理。
            image_inference_state = self.processor.set_text_prompt(state=image_inference_state, prompt=prompt)# 将文本提示设置到处理器中，以指导模型在推理过程中关注与提示相关的特征。这个函数会将文本提示与当前的推理状态关联起来，使得模型在处理图像时能够利用这个提示来生成更相关的结果。
            return image_inference_state

    def visualize(self, image, inference_state):
        # 判断传入的 image 是不是 numpy 数组（OpenCV 的格式）
        if isinstance(image, np.ndarray):
            # 将 OpenCV 默认的 BGR 颜色转成 RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img0 = Image.fromarray(img_rgb)
        else:
            img0 = Image.open(image) 

        # 使用处理器的推理状态来可视化结果
        plot_results(img0, inference_state) 
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("result.jpg", dpi=200, bbox_inches="tight", pad_inches=0)
        plt.show()
    
    def count_masks_area(self,inference_state):
        masks = inference_state["masks"] # 从字典的推理状态中提取分割掩码数据，通常是一个包含多个掩码的列表或数组，每个掩码对应于模型在图像中检测到的一个对象或区域。
        total_area = 0 # 初始化一个变量total_area，用于累积所有掩码的面积总和。
        for mask in masks: # 遍历每个掩码，计算其面积并累加到total_area中。每个掩码通常是一个二值图像，其中非零像素表示对象或区域的存在。
            area = (mask > 0).sum().item() # 计算当前掩码的面积，即统计掩码中非零像素的数量。这个操作通过将掩码中的像素值与0进行比较，得到一个布尔数组，然后使用sum()函数统计其中为True的元素数量，最后使用item()方法将结果转换为Python数值类型。
            total_area += area # 将当前掩码的面积累加到total_area中，以获得所有掩码的总面积。
        return total_area # 返回计算得到的总面积值，表示所有检测到的对象或区域在图像中的总占用面积。
    
    def find_maxmask(self,inference_state):
        masks = inference_state["masks"] # 从推理状态中提取分割掩码数据，通常是一个包含多个掩码的列表或数组，每个掩码对应于模型在图像中检测到的一个对象或区域。
        max_area = 0 # 初始化一个变量max_area，用于记录当前最大的掩码面积。
        max_mask = None # 初始化一个变量max_mask，用于存储当前最大的掩码。
        for mask in masks: # 遍历每个掩码，计算其面积并与当前的max_area进行比较，以找到最大的掩码。
            area = (mask > 0).sum().item() # 计算当前掩码的面积，即统计掩码中非零像素的数量。这个操作通过将掩码中的像素值与0进行比较，得到一个布尔数组，然后使用sum()函数统计其中为True的元素数量，最后使用item()方法将结果转换为Python数值类型。
            if area > max_area: # 如果当前掩码的面积大于max_area，则更新max_area和max_mask，以记录新的最大掩码和其面积。
                max_area = area
                max_mask = mask
        return max_mask, max_area # 返回找到的最大掩码和其对应的面积值，表示在所有检测到的对象或区域中占用面积最大的那个。