# SAM3 植物面积还原系统使用说明

## 简介
本系统使用 SAM3 (Segment Anything Model) 进行植物分割，结合棋盘格标定技术，实现不同高度、距离、角度下的植物面积还原。

## 文件说明

### 核心文件
- **sam3_area_calculation.py** - 主程序，使用SAM3进行分割和面积计算
- **calibration_utils.py** - 棋盘格标定工具函数库
- **pnp.py** - 旧版本（使用YOLO+DeepLabV3），已被SAM3方案替代

### 数据文件
- **camera_params.npz** - 相机标定参数（内参矩阵和畸变系数）
- **models/sam3.pt** - SAM3 分割模型权重

## 环境配置

### 1. 安装依赖

```bash
# 基础依赖
pip install opencv-python numpy pandas pillow

# SAM模型（FastSAM，推荐）
pip install ultralytics

# 如果使用其他SAM版本
# pip install segment-anything  # 原版SAM
# pip install sam2              # SAM2
```

### 2. 准备数据

1. **相机标定**：运行 `camera_calibration.py` 生成 `camera_params.npz`
2. **图片准备**：
   - 所有待处理图片放在同一文件夹
   - **必须**包含一张名为 `1.jpg`（或 `1.png`）的标定参考图片
   - 每张图片应包含：
     - 9x6 棋盘格标定板
     - 要测量的植物

## 使用方法

### 基本命令

```bash
# 高度变化分析
python sam3_area_calculation.py --folder "图片文件夹路径" --mode h

# 距离变化分析
python sam3_area_calculation.py --folder "图片文件夹路径" --mode d

# 角度变化分析
python sam3_area_calculation.py --folder "图片文件夹路径" --mode r
```

### 指定SAM模型路径

```bash
python sam3_area_calculation.py \
    --folder "D:\个人项目\植物面积还原算法\src\1" \
    --sam_model "D:\个人项目\peanut_detect\model\sam3.pt" \
    --mode h
```

### 参数说明

- `--folder`: 图片文件夹路径（必需）
- `--sam_model`: SAM3模型路径（可选，默认为 `D:\个人项目\peanut_detect\model\sam3.pt`）
- `--mode`: 分析维度（必需）
  - `h`: 高度变化
  - `d`: 距离变化
  - `r`: 角度/旋转变化

## 输出结果

### 1. 标注图像
- 在输入文件夹中生成 `annotated_*.jpg` 文件
- 绿色半透明覆盖显示分割的植物区域

### 2. CSV报告
- 文件名格式：`sam3_{mode}_变化纵向还原面积对比.csv`
- 包含以下信息：
  - 文件名、高度/距离/角度变化
  - 旋转向量、平移向量
  - 实际像素面积
  - 三种方法的还原面积：
    1. 像素间距法
    2. 姿态变化法（PnP）
    3. 姿态变化+角度补偿法
  - 各方法的误差百分比

### 3. 控制台输出
- 处理进度
- 每张图片的分割面积
- 最终对比表格

## 示例

```bash
# Windows
python sam3_area_calculation.py ^
    --folder "D:\个人项目\植物面积还原算法\src\1" ^
    --mode h

# Linux/Mac
python sam3_area_calculation.py \
    --folder "/path/to/images" \
    --mode h
```

## 与旧版本（pnp.py）的区别

| 特性 | 旧版本 (pnp.py) | 新版本 (SAM3) |
|------|-----------------|---------------|
| 检测方式 | YOLO粗检测 + DeepLabV3分割 | SAM3直接分割 |
| 模型复杂度 | 需要两个模型 | 单一模型 |
| 分割精度 | 依赖DeepLabV3训练质量 | SAM预训练，泛化性强 |
| 速度 | 双模型推理较慢 | 单模型更快 |
| 易用性 | 需要调试模型加载 | 开箱即用 |

## 故障排除

### 1. "未安装任何 SAM 库"
```bash
pip install ultralytics
```

### 2. "未找到名为'1'的标定图片"
确保文件夹中有 `1.jpg`、`1.png` 或 `1.jpeg`

### 3. "未检测到棋盘格"
- 检查图片中是否清晰可见 9x6 棋盘格
- 确保棋盘格没有遮挡或模糊
- 调整光照条件

### 4. "没有检测到植物"
- SAM模型可能需要调整参数
- 尝试修改 `conf` 和 `iou` 阈值：
  ```python
  sam.segment_image(path, conf=0.3, iou=0.8)
  ```

### 5. CUDA内存不足
- 减小图像尺寸：修改 `imgsz=1024` 为更小值（如 `imgsz=640`）
- 使用CPU：会自动fallback到CPU

## 技术原理

1. **SAM3分割**：使用Segment Anything Model对植物进行精确分割
2. **棋盘格检测**：检测9x6棋盘格获取空间参考
3. **PnP姿态估计**：计算相机相对于棋盘格的姿态（旋转和平移）
4. **缩放因子计算**：
   - 像素间距法：基于棋盘格角点间距
   - PnP法：基于投影点间距
   - 角度补偿法：考虑俯仰角影响
5. **面积还原**：`真实面积 = 像素面积 × 缩放因子²`

## 联系与支持

如有问题或建议，请查看项目文档或联系维护者。
