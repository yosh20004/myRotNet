# 不仅微调encoder 也微调分类头
# 但是最终效果不好，感觉是因为样本不好的问题

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. 高级微调超参数和设备设置 ---
CLASSIFIER_LR = 1e-4  # 分类头使用较大的学习率
ENCODER_LR = 1e-5     # 编码器使用非常小的学习率进行微调
BATCH_SIZE = 32
NUM_EPOCHS_FINE_TUNE = 20       # 可以适当增加训练轮数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALIDATION_SPLIT = 0.2  # 从新数据中划分20%作为验证集
IMAGE_SIZE = 224          # 使用预训练模型的标准输入尺寸 224x224
WEIGHT_DECAY = 1e-4       # 加入权重衰减正则化，防止过拟合

# --- 路径定义 ---
PRETRAINED_CHECKPOINT_PATH = "model/rotation_checkpoint.pth" 
FINETUNED_MODEL_PATH = "model/rotation_finetuned_advanced_v2.pth"
NEW_DATA_DIR = "data/image/"

print(f"使用设备: {DEVICE}")
print(f"将使用差异化学习率: Encoder LR = {ENCODER_LR}, Classifier LR = {CLASSIFIER_LR}")
print(f"图片尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")

# --- 2. 自定义Dataset (与之前相同) ---
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.rotation_transform = RandomRotation()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        original_img = Image.open(img_path).convert("RGB")
        rotated_img, rotation_label = self.rotation_transform(original_img)
        if self.transform:
            final_img = self.transform(rotated_img)
        return final_img, torch.tensor(rotation_label, dtype=torch.long)

class RandomRotation:
    def __init__(self):
        self.angles = [0, 90, 180, 270]
    def __call__(self, img):
        import random
        angle_choice = random.choice(self.angles)
        rotated_img = img.rotate(angle_choice)
        label = self.angles.index(angle_choice)
        return rotated_img, label

# --- 3. 模型定义 (更新为更复杂的结构) ---
class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4, freeze_encoder=False):
        super(RotationPredictionModel, self).__init__()
        # 使用预训练的ResNet18
        self.encoder = models.resnet18(weights='IMAGENET1K_V1')
        
        # 如果选择冻结，则编码器的权重在训练中不更新
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        # 使用更复杂的分类头
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs

# --- 4. 数据加载与划分 ---
# 训练集使用丰富的数据增强
train_base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)), # 先放大一点
    transforms.RandomCrop(IMAGE_SIZE), # 再随机裁剪，增加多样性
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # 随机颜色抖动
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 随机仿射变换
    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
])

# 验证集只做最基础的尺寸调整
val_base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])

# 最终转换为 Tensor 和归一化的变换
final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 找到所有图片路径
all_image_paths = glob.glob(os.path.join(NEW_DATA_DIR, 'image_*.jpg'))
if not all_image_paths:
    print(f"错误：在 '{NEW_DATA_DIR}' 中未找到 'image_*.jpg' 格式的图片。")
    exit()

print(f"总共找到 {len(all_image_paths)} 张新图片。")

# 划分训练集和验证集的索引
indices = np.arange(len(all_image_paths))
train_indices, val_indices = train_test_split(
    indices, test_size=VALIDATION_SPLIT, random_state=42, stratify=None
)

train_paths = [all_image_paths[i] for i in train_indices]
val_paths = [all_image_paths[i] for i in val_indices]

print(f"划分为: {len(train_paths)} 张训练图片, {len(val_paths)} 张验证图片。")

# 创建训练和验证数据集
train_dataset = CustomImageDataset(image_paths=train_paths, transform=train_base_transform)
val_dataset = CustomImageDataset(image_paths=val_paths, transform=val_base_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 5. 加载模型并设置差异化学习率优化器 ---
model = RotationPredictionModel(num_classes=4, freeze_encoder=False).to(DEVICE)
criterion = nn.CrossEntropyLoss()

if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
    print(f"--- 正在加载预训练模型权重 ---")
    checkpoint = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("--- 预训练模型加载成功！---")
else:
    print(f"错误：找不到预训练模型 '{PRETRAINED_CHECKPOINT_PATH}'。")
    exit()

# 设置差异化学习率
optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': ENCODER_LR},
    {'params': model.classifier.parameters(), 'lr': CLASSIFIER_LR}
], weight_decay=WEIGHT_DECAY)

# 添加学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

print("--- 优化器已设置差异化学习率。开始全网络微调。---")

# --- 6. 微调与验证循环 ---
best_val_acc = 0.0
for epoch in range(NUM_EPOCHS_FINE_TUNE):
    print(f"\n--- 微调 Epoch {epoch+1}/{NUM_EPOCHS_FINE_TUNE} ---")
    
    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
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
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="验证中"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct_val / total_val
    print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")

    # 更新学习率
    scheduler.step(val_loss)

    # --- 保存最佳模型 ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(os.path.dirname(FINETUNED_MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
        print(f"*** 发现新的最佳模型！验证准确率达到 {val_acc:.2f}%，已保存至 '{FINETUNED_MODEL_PATH}' ***")

print(f"\n--- 微调完成！最佳验证准确率为: {best_val_acc:.2f}% ---") 