import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np
from collections import defaultdict

# --- 准备工作：需要和你训练时完全一致 ---

# 1. 把你的模型定义复制过来
class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationPredictionModel, self).__init__()
        self.encoder = models.resnet18(weights=None)
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(num_features, num_classes)
    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs

# 2. 定义和你验证时一样的图像预处理步骤
inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. 加载模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "model/rotation_checkpoint.pth" # 存档文件路径
model = RotationPredictionModel(num_classes=4).to(DEVICE)

# 先加载整个检查点文件（一个字典）
print(f"正在从 '{CHECKPOINT_PATH}' 加载检查点...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
# 然后从字典中只提取模型的权重状态
model.load_state_dict(checkpoint['model_state_dict'])
# -------------------------

model.eval() # 设置为评估模式！
print("模型加载成功，已进入评估模式。")

# --- 修改：从预测单张图片改为在测试集上评估 ---

# 4. 加载CIFAR-10测试集
# 注意：这里transform=None，因为我们需要原始的PIL Image来进行手动旋转
test_dataset_raw = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
cifar10_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
NUM_IMAGES_TO_TEST = 50 # 想测试的图片数量

print(f"\n--- 将在CIFAR-10测试集的前 {NUM_IMAGES_TO_TEST} 张图片上进行评估 ---")

# 初始化统计变量
total_predictions = 0
correct_predictions = 0
angle_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

# 5. 循环遍历指定数量的测试图片
for i in range(NUM_IMAGES_TO_TEST):
    # 从数据集中获取一张原始图片和它的真实类别标签
    original_img, original_label_idx = test_dataset_raw[i]
    original_label_name = cifar10_labels[original_label_idx]

    print(f"\n===== 测试图片 {i+1}/{NUM_IMAGES_TO_TEST} | 原始类别: {original_label_name} =====")

    # 对这张图片进行四种角度的旋转测试
    for angle in [0, 90, 180, 270]:
        # 手动旋转图片
        rotated_img = original_img.rotate(angle)
        
        # 预处理并增加一个batch维度
        img_tensor = inference_transform(rotated_img).unsqueeze(0).to(DEVICE)

        # 不计算梯度，进行预测
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        # 将预测结果解码回角度
        angle_map = {0: 0, 1: 90, 2: 180, 3: 270}
        predicted_angle = angle_map[predicted_idx.item()]
        
        # 更新统计信息
        total_predictions += 1
        angle_stats[angle]['total'] += 1
        if predicted_angle == angle:
            correct_predictions += 1
            angle_stats[angle]['correct'] += 1
            print(f"  - 真实旋转: {angle:>3}° -> 模型预测: {predicted_angle:>3}°  (正确 🎉)")
        else:
            print(f"  - 真实旋转: {angle:>3}° -> 模型预测: {predicted_angle:>3}°  (错误 ❌)")

# 打印统计结果
print("\n=== 评估结果统计 ===")
print(f"总体正确率: {100 * correct_predictions / total_predictions:.2f}% ({correct_predictions}/{total_predictions})")
print("\n各角度正确率:")
for angle in sorted(angle_stats.keys()):
    stats = angle_stats[angle]
    accuracy = 100 * stats['correct'] / stats['total']
    print(f"{angle:>3}°: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
