import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import random
from tqdm import tqdm # 用于显示漂亮的进度条

# --- 1. 超参数和设备设置 ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 10 # 为了演示，设置一个较小的轮数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# --- 2. 数据集和数据加载 ---
# 我们在第一步中定义的数据处理类
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

# 为 CIFAR-10 定义的变换
# 论文中使用了 224x224 的输入，我们这里为了快速演示，使用 32x32
base_transform = transforms.Compose([transforms.Resize((32, 32))])
final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建数据集
cifar_train_raw = datasets.CIFAR10(root='./data', train=True, download=True, transform=base_transform)
cifar_val_raw = datasets.CIFAR10(root='./data', train=False, download=True, transform=base_transform)

train_dataset = RotationDataset(cifar_train_raw, transform=final_transform)
val_dataset = RotationDataset(cifar_val_raw, transform=final_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 3. 模型定义 ---
# 我们在第二步中定义的模型
class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationPredictionModel, self).__init__()
        self.encoder = models.resnet18(weights=None) # 从零开始训练
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(num_features, num_classes)
    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs

# --- 4. 初始化模型、损失函数和优化器 ---
model = RotationPredictionModel(num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss() # 交叉熵损失，适用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. 训练循环 ---
for epoch in range(NUM_EPOCHS):
    # --- 训练阶段 ---
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    for images, labels in tqdm(train_loader, desc="训练中"):
        # 将数据移动到指定设备
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad() # 清空之前的梯度
        loss.backward()      # 计算梯度
        optimizer.step()       # 更新权重

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")

    # --- 验证阶段 ---
    model.eval() # 设置模型为评估模式
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad(): # 在此模式下，不计算梯度，节省计算资源
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
print("\n预训练完成!")
# 保存encoder + 分类头 的全部参数 用以进行完整的分类任务
full_model_save_path = "rotation_full_model.pth"
torch.save(model.state_dict(), full_model_save_path) 
print(f"训练好的完整模型权重已保存至: {full_model_save_path}")