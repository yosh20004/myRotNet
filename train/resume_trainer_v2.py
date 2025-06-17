import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import random
from tqdm import tqdm
import os

# --- 1. 预训练超参数和设备设置 ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 64  # 根据显存大小可适当调整，224x224图像更耗显存
NUM_EPOCHS = 20  # 预训练轮数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224 # <-- 关键修改：与微调脚本的图像尺寸保持一致

# --- 路径定义 ---
# 用于中途断点续训的检查点
PRETRAIN_CHECKPOINT_PATH = "model/rotnet_cifar_checkpoint.pth" 
# 最终产出的、用于下一阶段微调的预训练模型
FINAL_MODEL_PATH = "model/rotnet_cifar10_pretrained.pth"

print(f"--- 阶段1：在CIFAR-10上进行自监督预训练 ---")
print(f"使用设备: {DEVICE}")
print(f"目标训练轮数: {NUM_EPOCHS} epochs")
print(f"输入图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")

# --- 2. 数据集和模型定义 ---

# RandomRotation 类保持不变
class RandomRotation:
    def __init__(self):
        self.angles = [0, 90, 180, 270]
    def __call__(self, img):
        angle_choice = random.choice(self.angles)
        rotated_img = img.rotate(angle_choice)
        label = self.angles.index(angle_choice)
        return rotated_img, label

# RotationDataset 类保持不变
class RotationDataset(Dataset):
    def __init__(self, underlying_dataset, transform=None):
        self.underlying_dataset = underlying_dataset
        self.rotation_transform = RandomRotation()
        self.transform = transform
    def __len__(self):
        return len(self.underlying_dataset)
    def __getitem__(self, idx):
        # CIFAR-10数据集返回 (img, class_label)，我们只需要img
        original_img, _ = self.underlying_dataset[idx]
        rotated_img, rotation_label = self.rotation_transform(original_img)
        if self.transform:
            final_img = self.transform(rotated_img)
        return final_img, torch.tensor(rotation_label, dtype=torch.long)

# 关键修改：使用与微调脚本完全一致的模型结构
class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationPredictionModel, self).__init__()
        # 编码器从零开始训练，不加载ImageNet预训练权重
        self.encoder = models.resnet18(weights=None)
        
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        # 使用与微调脚本完全相同的、更复杂的分类头
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

# --- 3. 数据加载 ---
# 关键修改：将CIFAR-10图像尺寸调整为224x224
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # 使用ImageNet的标准化参数，这是一个更通用的选择
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

cifar_train_raw = datasets.CIFAR10(root='./data', train=True, download=True)
cifar_val_raw = datasets.CIFAR10(root='./data', train=False, download=True)

train_dataset = RotationDataset(cifar_train_raw, transform=transform)
val_dataset = RotationDataset(cifar_val_raw, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- 4. 初始化模型、损失函数和优化器 ---
model = RotationPredictionModel(num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. 加载检查点以恢复训练 ---
start_epoch = 0
os.makedirs(os.path.dirname(PRETRAIN_CHECKPOINT_PATH), exist_ok=True)
if os.path.exists(PRETRAIN_CHECKPOINT_PATH):
    print(f"--- 发现预训练检查点 '{PRETRAIN_CHECKPOINT_PATH}'，加载中... ---")
    checkpoint = torch.load(PRETRAIN_CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"--- 成功加载！将从 Epoch {start_epoch + 1} 开始继续预训练。---")
else:
    print(f"--- 未发现预训练检查点，将从头开始预训练。---")

# --- 6. 预训练循环 ---
if start_epoch >= NUM_EPOCHS:
    print(f"模型已经训练完成 {start_epoch} 轮，无需额外训练。")
else:
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- 预训练 Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="预训练中"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        print(f"预训练损失: {train_loss:.4f}")

        # --- 验证阶段 ---
        model.eval()
        val_acc = 0
        with torch.no_grad():
            correct_val = 0
            total_val = 0
            for images, labels in tqdm(val_loader, desc="验证中"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
            val_acc = 100 * correct_val / total_val
        print(f"验证准确率: {val_acc:.2f}%")

        # --- 每个epoch后都保存一次检查点，用于恢复 ---
        print("正在保存当前预训练进度...")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, PRETRAIN_CHECKPOINT_PATH)
        print(f"进度已保存至 '{PRETRAIN_CHECKPOINT_PATH}'")

print(f"\n--- 预训练完成! ---")

# --- 7. 关键步骤：保存最终的、干净的模型权重用于微调 ---
print(f"正在保存最终的预训练模型至 '{FINAL_MODEL_PATH}'...")
# 只保存模型的状态字典，这正是下一阶段微调脚本所需要的
final_model_weights = model.state_dict()
torch.save(final_model_weights, FINAL_MODEL_PATH)
print("--- 模型保存成功！现在您可以运行微调脚本了。 ---")