# 不改变encoder结构 效果不佳

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import os
import glob

# --- 1. 微调超参数和设备设置 ---
FINE_TUNE_LEARNING_RATE = 1e-4  # 微调时使用一个较小的学习率
BATCH_SIZE = 32                 # 新数据集较小，可以使用小一点的batch size
NUM_EPOCHS_FINE_TUNE = 15       # 在新数据集上训练的轮数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 路径定义 ---
# 你之前训练好的模型检查点
PRETRAINED_CHECKPOINT_PATH = "model/rotation_checkpoint.pth" 
# 微调后新模型的保存路径
FINETUNED_MODEL_PATH = "model/rotation_finetuned.pth"
# 新的训练数据文件夹
NEW_DATA_DIR = "data/image/"

print(f"使用设备: {DEVICE}")
print(f"将使用 '{PRETRAINED_CHECKPOINT_PATH}' 作为预训练模型。")
print(f"新的训练数据来源: '{NEW_DATA_DIR}'")

# --- 2. 为新数据创建自定义Dataset ---
# 这个Dataset专门用于加载 data/image/ 下的图片
class CustomImageDataset(Dataset):
    """用于加载指定文件夹下图片的自定义数据集"""
    def __init__(self, folder_path, transform=None):
        # 使用glob找到所有符合 "image_*.jpg" 格式的图片路径
        self.image_paths = glob.glob(os.path.join(folder_path, 'image_*.jpg'))
        self.transform = transform
        self.rotation_transform = RandomRotation()
        print(f"在新数据集中找到 {len(self.image_paths)} 张图片。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图片
        img_path = self.image_paths[idx]
        # 使用Pillow打开图片并转换为RGB格式，以防是灰度图或带alpha通道的图
        original_img = Image.open(img_path).convert("RGB")
        
        # 应用旋转，获取旋转后的图片和对应的标签 (0, 1, 2, 3)
        rotated_img, rotation_label = self.rotation_transform(original_img)
        
        # 应用torchvision的transform (例如 ToTensor, Normalize)
        if self.transform:
            final_img = self.transform(rotated_img)
            
        return final_img, torch.tensor(rotation_label, dtype=torch.long)

# 辅助类：随机旋转 (与你原脚本中的定义一致)
class RandomRotation:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        import random
        angle_choice = random.choice(self.angles)
        rotated_img = img.rotate(angle_choice)
        label = self.angles.index(angle_choice)
        return rotated_img, label

# --- 3. 模型定义 (与你原脚本中的定义一致) ---
class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationPredictionModel, self).__init__()
        # 使用预训练的ResNet18作为编码器骨干
        self.encoder = models.resnet18(weights=None) # 注意：这里不加载ImageNet预训练权重，我们将加载自己的权重
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()  # 移除原有的全连接层
        # 定义自己的分类头
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs

# --- 4. 数据加载和预处理 ---
# 定义新数据需要进行的图像变换
# 需要和之前训练时的变换保持一致，特别是Resize和Normalize
fine_tune_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建新的数据集和数据加载器
custom_dataset = CustomImageDataset(folder_path=NEW_DATA_DIR, transform=fine_tune_transform)
# 如果新数据集样本很少，可以考虑不用DataLoader或设置batch_size为数据集大小
# 这里我们还是用DataLoader，方便扩展
if len(custom_dataset) > 0:
    train_loader_fine_tune = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
else:
    print("错误：在指定目录下没有找到任何 'image_*.jpg' 格式的图片，无法进行微调。")
    exit() # 退出脚本

# --- 5. 加载预训练模型并准备微调 ---
model = RotationPredictionModel(num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()

# 加载你之前训练好的模型权重
if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
    print(f"--- 正在加载预训练模型权重 ---")
    # 加载整个检查点字典
    checkpoint = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=DEVICE)
    # 将模型状态加载到模型中
    model.load_state_dict(checkpoint['model_state_dict'])
    print("--- 预训练模型加载成功！---")
else:
    print(f"错误：找不到预训练模型 '{PRETRAINED_CHECKPOINT_PATH}'。请先运行主训练脚本。")
    exit()

# --- 核心：冻结编码器层，只训练分类头 ---
print("--- 正在冻结编码器 (ResNet) 的权重... ---")
for param in model.encoder.parameters():
    param.requires_grad = False
print("--- 编码器已冻结，只有分类头的权重会被更新。---")


# 创建一个新的优化器，只包含需要更新的参数 (即分类头的参数)
# model.classifier.parameters() 会自动返回需要计算梯度的参数
optimizer = optim.Adam(model.classifier.parameters(), lr=FINE_TUNE_LEARNING_RATE)

# --- 6. 微调训练循环 ---
print("\n--- 开始微调训练 ---")
model.train() # 将模型设置为训练模式。注意：即使部分层被冻结，也应该调用.train()

for epoch in range(NUM_EPOCHS_FINE_TUNE):
    print(f"\n--- 微调 Epoch {epoch+1}/{NUM_EPOCHS_FINE_TUNE} ---")
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # 使用tqdm来显示进度条
    for images, labels in tqdm(train_loader_fine_tune, desc="微调中"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader_fine_tune)
    train_acc = 100 * correct_train / total_train
    print(f"微调损失: {train_loss:.4f} | 微调准确率: {train_acc:.2f}%")

print("\n--- 微调完成！---")

# --- 7. 保存微调后的模型 ---
# 创建保存目录（如果不存在）
os.makedirs(os.path.dirname(FINETUNED_MODEL_PATH), exist_ok=True)
# 只保存模型的状态字典，这是推荐的做法
torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
print(f"微调后的模型已保存至 '{FINETUNED_MODEL_PATH}'")

