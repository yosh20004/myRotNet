import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import random
from tqdm import tqdm
import os

# --- 1. 超参数和设备设置 ---
# 将训练过程分为两个阶段
INITIAL_LEARNING_RATE = 1e-3  # 初始阶段学习率
REFINE_LEARNING_RATE = 1e-4   # 精炼阶段学习率
BATCH_SIZE = 128
NUM_EPOCHS = 20            # 每多训练n轮，就给已有数字加n
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "model/rotation_checkpoint.pth" # 统一的存档/检查点文件名

print(f"使用设备: {DEVICE}")
print(f"总训练轮数目标: {NUM_EPOCHS} epochs")

# --- 2. 数据集和模型定义 (与之前完全相同) ---
class RandomRotation:
    def __init__(self):
        self.angles = [0, 90, 180, 270]
    def __call__(self, img):
        angle_choice = random.choice(self.angles)
        rotated_img = img.rotate(angle_choice)
        label = self.angles.index(angle_choice)
        return rotated_img, label

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

# --- 3. 数据加载 (与之前完全相同) ---
base_transform = transforms.Compose([transforms.Resize((32, 32))])
final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_train_raw = datasets.CIFAR10(root='./data', train=True, download=True, transform=base_transform)
cifar_val_raw = datasets.CIFAR10(root='./data', train=False, download=True, transform=base_transform)
train_dataset = RotationDataset(cifar_train_raw, transform=final_transform)
val_dataset = RotationDataset(cifar_val_raw, transform=final_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 4. 初始化模型、损失函数和优化器 ---
model = RotationPredictionModel(num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
# 优化器先用一个占位符，学习率将在加载检查点后确定
optimizer = optim.Adam(model.parameters()) 

# --- 5. 智能加载检查点 ---
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"--- 发现检查点 '{CHECKPOINT_PATH}'，加载中... ---")
    checkpoint = torch.load(CHECKPOINT_PATH)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

    # 关键：继续训练时，使用更小的学习率
    print(f"--- 成功加载！将从 Epoch {start_epoch + 1} 开始继续【精炼】。---")
    print(f"将学习率从 {INITIAL_LEARNING_RATE} 调整为 {REFINE_LEARNING_RATE}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = REFINE_LEARNING_RATE
else:
    print(f"--- 未发现检查点，将从头开始【初始训练】。---")
    # 关键：从头训练时，使用初始学习率
    print(f"使用初始学习率: {INITIAL_LEARNING_RATE}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = INITIAL_LEARNING_RATE

# --- 6. 统一的训练与精炼循环 ---
# 循环将从加载的epoch下一轮开始，直到总轮数
if start_epoch >= NUM_EPOCHS:
    print(f"模型已经训练完成 {start_epoch} 轮，无需额外训练。")
else:
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
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

        # --- 修改：每个epoch后都保存一次检查点 ---
        print("正在保存当前进度...")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"进度已保存至 '{CHECKPOINT_PATH}'")

    print(f"\n训练/精炼完成! 总共训练了 {NUM_EPOCHS} 轮。")

