import os

import cv2
import sam3_triton
sam3_triton._ensure_triton_stub()  # 确保在不支持triton的平台上提供stub
import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

import torch

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
with open("D:\\personal_project\\peanut_detect\\model\\sam3.pt", "rb") as f:
    print(f.read(4))
    if(f.read(4) == b'PK\x03\x04'):
        print("[INFO] 模型文件是一个zip文件，可能是torchscript格式的模型")
    else:
        print("[INFO] 模型文件不是一个zip文件，可能是torchscript格式的模型")
        # 启用TF32以加速Ampere GPU上的矩阵乘法和卷积运算,因为sam3.pt是torchscript格式的模型，可能无法使用triton加速，所以启用TF32来提升性能。对于非Ampere GPU，TF32无效，不会有任何影响。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # autocast上下文管理器可以在支持的设备上自动选择适当的数据类型进行推理，以提高性能和效率。对于Ampere GPU，推荐使用bfloat16，因为它在保持数值稳定性的同时提供了更好的性能。对于不支持bfloat16的设备，可以回退到float16或float32。
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # 选择适当的数据推理类型，官方推荐使用bfloat16,因为float16无法满足输入输出的
        # interebce_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        MODEL_PATH = r"D:\personal_project\peanut_detect\model\sam3.pt"
        USE_CUDA = torch.cuda.is_available()
        DEVICE_STR = "cuda" if USE_CUDA else "cpu"   # 关键：传字符串
        DEVICE = torch.device(DEVICE_STR)
        

        model = build_sam3_image_model( # 模型构建函数，负责加载模型权重并创建模型实例。它接受设备字符串、模型权重路径以及是否从Hugging Face加载模型等参数，并返回一个准备好进行推理的模型实例。
            device=DEVICE_STR,
            checkpoint_path=MODEL_PATH,
            load_from_HF=False,
        )
        model.eval()# 将模型设置为评估模式，这对于某些层（如dropout和batch normalization）在推理阶段的行为是必要的，评估模式的作用是确保模型在推理时的行为与训练时一致，特别是对于那些在训练和推理阶段表现不同的层。评估模式会禁用dropout层的随机丢弃行为，并使用batch normalization层的全局统计数据，而不是每个批次的统计数据，从而确保推理结果的稳定性和一致性。
        # 使用sam3模型创建一个处理器实例，处理器负责管理输入图像、文本提示和推理状态，并提供接口来设置图像、文本提示以及重置提示等功能。它还接受一个置信度阈值参数，用于在推理过程中筛选结果。
        processor = Sam3Processor(model, device=DEVICE_STR, confidence_threshold=0.5)        
        image_path = r"D:\personal_project\monocular-plant-area\img\2\1.jpg"
        image = Image.open(image_path)

        # 使用半精度（bfloat16）进行推理，以提高性能和效率，特别是在支持bfloat16的设备上。对于不支持bfloat16的设备，回退到float32以确保兼容性和数值稳定性。选择适当的数据类型可以显著提升推理速度，同时保持足够的精度。
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if USE_CUDA:# 检查是否有可用的CUDA设备，如果有，则在CUDA设备上进行推理，并使用torch.autocast自动选择适当的数据类型以提高性能。如果没有CUDA设备，则在CPU上进行推理，不使用torch.autocast。
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                
                inference_state = processor.set_image(image)# 将输入图像设置到处理器中，准备进行推理。这个函数会将图像转换为模型所需的格式，并将其存储在处理器的内部状态中，以便后续的文本提示和推理操作使用。
                processor.reset_all_prompts(inference_state)# 重置处理器中的所有提示，这通常在每次新的推理开始时调用，以确保之前的提示不会影响当前的推理结果。这个函数会清除处理器内部存储的所有文本提示和相关状态，使得处理器处于一个干净的状态，准备接受新的提示和进行新的推理。
                inference_state = processor.set_text_prompt(state=inference_state, prompt="plant")# 将文本提示（在这个例子中是“plant”）设置到处理器中，以指导模型在推理过程中关注与植物相关的特征。这个函数会将文本提示与当前的推理状态关联起来，使得模型在处理图像时能够利用这个提示来生成更相关的结果。
                img0 = Image.open(image_path) # 打开输入图像文件，准备进行可视化。这个步骤是为了在后续的结果可视化中使用原始图像，以便将模型的推理结果（如边界框）绘制在原始图像上进行展示。
                plot_results(img0, inference_state) # 使用处理器的推理状态来可视化结果。传两个参数，1为图片，2为字典，包含置信度，边界坐标，分割掩码
                plt.axis("off")
                plt.tight_layout()
                plt.savefig("result.jpg", dpi=200, bbox_inches="tight", pad_inches=0)
                plt.show()    
        else:
            with torch.inference_mode():
                inference_state = processor.set_image(image)
                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(state=inference_state, prompt="plant")
                img0 = Image.open(image_path)
                plot_results(img0, inference_state)    


