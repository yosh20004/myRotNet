# 使用自己的数据集进行训练 模型仍使用RotNet
# 效果仍然一般 并且似乎出现了过拟合，可能是样本不好的原因

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import random
import os
from tqdm import tqdm # 用于显示漂亮的进度条

# --- 1. 超参数和设备设置 ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 10 # 为了演示，设置一个较小的轮数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# --- 2. 数据集和数据加载 ---

# --- MODIFICATION START: 新增自定义数据集类 ---
class CustomImageDataset(Dataset):
    """用于从指定文件夹加载图片的自定义数据集"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # 获取文件夹中所有图片的路径 (支持 jpg, png, jpeg)
        self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
        print(f"在目录 '{img_dir}' 中找到 {len(self.img_paths)} 张图片。")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # 打开图片并确保为 RGB 格式
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # 对于自监督学习，原始标签是不需要的，我们返回一个虚拟标签 0
        # 真正的标签将由 RotationDataset 包装器生成
        return image, 0
# --- MODIFICATION END ---


# 用于生成旋转任务的数据处理类（保持不变）
class RandomRotation:
    def __init__(self):
        self.angles = [0, 90, 180, 270]
    def __call__(self, img):
        angle_choice = random.choice(self.angles)
        rotated_img = img.rotate(angle_choice)
        label = self.angles.index(angle_choice)
        return rotated_img, label

# RotationDataset 包装器（保持不变）
class RotationDataset(Dataset):
    def __init__(self, underlying_dataset, transform=None):
        self.underlying_dataset = underlying_dataset
        self.rotation_transform = RandomRotation()
        self.transform = transform
    def __len__(self):
        return len(self.underlying_dataset)
    def __getitem__(self, idx):
        original_img, _ = self.underlying_dataset[idx]
        rotated_img, rotation_label = self.rotation_transform(original_img)
        if self.transform:
            final_img = self.transform(rotated_img)
        return final_img, torch.tensor(rotation_label, dtype=torch.long)

# 定义图像变换
# 论文中使用了 224x224 的输入，我们这里为了快速演示，使用 32x32
base_transform = transforms.Compose([
    transforms.Resize((32, 32)),
])
final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- MODIFICATION START: 使用你的自定义数据集 ---
# 指定你的图片文件夹路径
IMAGE_DIR = './data/image/'

# 检查路径是否存在
if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(f"错误: 找不到目录 '{IMAGE_DIR}'。请确保路径正确并且该文件夹内有图片。")

# 从你的自定义文件夹创建完整数据集
custom_dataset_raw = CustomImageDataset(img_dir=IMAGE_DIR, transform=base_transform)

# 检查数据集是否为空
if len(custom_dataset_raw) == 0:
    raise ValueError(f"错误: 在 '{IMAGE_DIR}' 中没有找到任何图片。请检查你的图片文件。")

# 将数据集分割为训练集和验证集 (例如, 80% 训练, 20% 验证)
train_size = int(0.8 * len(custom_dataset_raw))
val_size = len(custom_dataset_raw) - train_size
# 使用固定的随机种子确保每次分割结果一致
generator = torch.Generator().manual_seed(42)
custom_train_raw, custom_val_raw = random_split(custom_dataset_raw, [train_size, val_size], generator=generator)
print(f"数据集已分割: {len(custom_train_raw)} 张用于训练, {len(custom_val_raw)} 张用于验证。")

# 使用 RotationDataset 包装器来应用自监督学习任务
train_dataset = RotationDataset(custom_train_raw, transform=final_transform)
val_dataset = RotationDataset(custom_val_raw, transform=final_transform)
# --- MODIFICATION END ---


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 3. 模型定义 ---
# (保持不变)
class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationPredictionModel, self).__init__()
        # 使用不带预训练权重的 resnet18
        self.encoder = models.resnet18(weights=None)
        num_features = self.encoder.fc.in_features
        # 将 resnet 的全连接层替换为一个恒等层，以提取特征
        self.encoder.fc = nn.Identity()
        # 旋转任务的分类头
        self.classifier = nn.Linear(num_features, num_classes)
    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs

# --- 4. 初始化模型、损失函数和优化器 ---
# (保持不变)
model = RotationPredictionModel(num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. 训练循环 ---
# (保持不变)
for epoch in range(NUM_EPOCHS):
    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    for images, labels in tqdm(train_loader, desc="训练中"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")

    # --- 验证阶段 ---
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="验证中"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val
    print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")

# --- 6. 保存训练好的 Encoder ---
# (保持不变)
print("\n预训练完成!")
full_model_save_path = "model/rotation_full_model_custom_on_my_data.pth"
torch.save(model.state_dict(), full_model_save_path) 
print(f"使用你的自定义数据训练好的完整模型权重已保存至: {full_model_save_path}")
