# 这次在自己的数据集上直接做训练，但是优化了一些超参数，效果比之前好了一些
# 这说明我们自己的数据集不能说不行，只能说其实也是有点用处，模型还是学到了一定东西的，验证集上准确率76%  其实也是一个很能接受的准确率了
# 下一阶段的目标是在公家的手写数据集的基础上训练一个鲁棒性很高的基准模型，然后我们用我们的数据集去进行微调

# 注意：我们的数据集其实还是有点问题的，因为训练过程中的准确率是螺旋上升的，有时候反而急速下降，不明白原因

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
from PIL import Image
import random
import os
from tqdm import tqdm # 用于显示漂亮的进度条

# --- 1. 超参数和设备设置 ---
# 关键改动: 调整超参数以适应微调
LEARNING_RATE = 1e-4      # 使用较小的学习率进行微调
BATCH_SIZE = 32           # 图像尺寸变大，如果显存不足可以适当调小此值
NUM_EPOCHS = 20           # 增加训练轮数以保证模型充分收敛
WEIGHT_DECAY = 1e-4       # 加入权重衰减正则化，防止过拟合
IMAGE_SIZE = 224          # 关键改动: 使用预训练模型的标准输入尺寸 224x224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 你的数据和模型存放路径
IMAGE_DIR = './data/image/' # <-- 请将你的图片数据放在这个文件夹下
MODEL_SAVE_DIR = './model/' # <-- 训练好的模型将保存在这里

print(f"使用设备: {DEVICE}")
print(f"图片尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"批次大小: {BATCH_SIZE}")
print(f"学习率: {LEARNING_RATE}")


# --- 2. 数据集定义 ---

class CustomImageDataset(Dataset):
    """用于从指定文件夹加载图片的自定义数据集"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # 获取文件夹中所有图片的路径 (支持 jpg, png, jpeg)
        try:
            self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
        except FileNotFoundError:
            print(f"错误: 找不到目录 '{img_dir}'。请创建该目录并放入图片。")
            self.img_paths = []
            
        if not self.img_paths:
             print(f"警告: 在目录 '{img_dir}' 中没有找到任何图片文件。")
        else:
            print(f"在目录 '{img_dir}' 中找到 {len(self.img_paths)} 张图片。")


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # 打开图片并确保为 RGB 格式
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            # 原始标签是不需要的，我们返回一个虚拟标签 0
            return image, 0
        except Exception as e:
            print(f"错误: 无法加载或处理图片 {img_path}。错误信息: {e}")
            # 返回一个占位符，或者跳过这个样本
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), 0

class RandomRotation:
    """对PIL Image进行随机旋转并返回角度标签"""
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        angle_choice = random.choice(self.angles)
        rotated_img = img.rotate(angle_choice)
        label = self.angles.index(angle_choice)
        return rotated_img, label

class RotationDataset(Dataset):
    """
    包装器数据集：
    1. 从底层数据集中获取原始PIL Image。
    2. 应用RandomRotation转换，得到旋转后的PIL Image和旋转标签。
    3. 应用最终的transform（ToTensor, Normalize）。
    """
    def __init__(self, underlying_dataset, final_transform=None):
        self.underlying_dataset = underlying_dataset
        self.rotation_transform = RandomRotation()
        self.final_transform = final_transform

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        # 1. 获取原始的、未转换的PIL Image
        original_img, _ = self.underlying_dataset[idx]
        
        # 2. 应用旋转任务
        rotated_img, rotation_label = self.rotation_transform(original_img)
        
        # 3. 应用最终的Tensor转换和归一化
        if self.final_transform:
            final_img = self.final_transform(rotated_img)
            
        return final_img, torch.tensor(rotation_label, dtype=torch.long)


# --- 3. 数据变换 ---

# 关键改动: 引入丰富的数据增强并使用ImageNet的归一化参数
# 训练集使用丰富的数据增强来提升模型鲁棒性
train_base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)), # 先放大一点
    transforms.RandomCrop(IMAGE_SIZE), # 再随机裁剪，增加多样性
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # 随机颜色抖动
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 随机仿射变换
    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
])

# 验证集只做最基础的尺寸调整，以反映真实评估环境
val_base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])

# 最终转换为 Tensor 和归一化的变换 (对训练和验证都适用)
# 使用在 ImageNet 上预训练模型的标准均值和标准差
final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- 4. 加载和准备数据 ---

# 创建原始数据集实例时，传入不同的基础变换
# 注意：这一步只是定义了变换，并没有真正应用
underlying_train_dataset = CustomImageDataset(img_dir=IMAGE_DIR, transform=train_base_transform)
underlying_val_dataset = CustomImageDataset(img_dir=IMAGE_DIR, transform=val_base_transform)

# 检查数据集是否为空
if len(underlying_train_dataset) == 0:
    raise ValueError(f"错误: 在 '{IMAGE_DIR}' 中没有找到任何图片。请检查你的图片文件并重新运行。")

# 分割数据集索引
train_size = int(0.8 * len(underlying_train_dataset))
val_size = len(underlying_train_dataset) - train_size
generator = torch.Generator().manual_seed(42) # 固定随机种子以保证每次分割结果一致
train_indices, val_indices = random_split(range(len(underlying_train_dataset)), [train_size, val_size], generator=generator)

# 使用 Subset 从原始数据集中根据索引创建训练和验证子集
# 训练子集会应用 `train_base_transform`
# 验证子集会应用 `val_base_transform`
train_subset = Subset(underlying_train_dataset, train_indices)
val_subset = Subset(underlying_val_dataset, val_indices)
print(f"数据集已分割: {len(train_subset)} 张用于训练, {len(val_subset)} 张用于验证。")

# 使用 RotationDataset 包装器来最终生成旋转任务所需的数据
train_dataset = RotationDataset(train_subset, final_transform=final_transform)
val_dataset = RotationDataset(val_subset, final_transform=final_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# --- 5. 模型定义 ---
class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4, freeze_encoder=False):
        super(RotationPredictionModel, self).__init__()
        # 关键改动: 使用预训练权重来利用迁移学习
        self.encoder = models.resnet18(weights='IMAGENET1K_V1')
        
        # 如果选择冻结，则编码器的权重在训练中不更新
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        num_features = self.encoder.fc.in_features
        # 替换掉原始的分类头
        self.encoder.fc = nn.Identity()
        
        # 添加我们自己的分类头
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

# --- 6. 初始化模型、损失函数和优化器 ---
model = RotationPredictionModel(num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 学习率调度器：在验证损失不下降时，自动降低学习率
# --- FIX START ---
# 修复: 在较新的PyTorch版本中, `verbose` 参数已被移除, 其功能默认为True
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
# --- FIX END ---


# --- 7. 训练和验证循环 ---
best_val_acc = 0.0
# 确保模型保存目录存在
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    progress_bar = tqdm(train_loader, desc="训练中", colour="green")
    for images, labels in progress_bar:
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
        
        progress_bar.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")

    # --- 验证阶段 ---
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证中", colour="cyan")
        for images, labels in progress_bar:
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

    # 更新学习率
    scheduler.step(val_loss)

    # --- 保存最佳模型 ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(MODEL_SAVE_DIR, "rotnet_on_my_data_v2.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"🎉 新的最佳验证准确率: {best_val_acc:.2f}%。模型已保存至: {best_model_path}")

# --- 8. 结束训练 ---
print("\n训练完成!")
print(f"整个训练过程中最佳的验证准确率为: {best_val_acc:.2f}%")
