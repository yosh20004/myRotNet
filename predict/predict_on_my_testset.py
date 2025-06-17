# 经过测试，我发现模型在35张样图上取得了20对/15错的成绩
# 思考1：很多图像本身就是正的，会不会模型偷懒，压根认为不旋转就是最优的呢？
# 思考2：很多图像被翻转了之后出现了上下颠倒的情况而出错的情况，而斜着的情况却很少见，这是不是有可能说明图像并没有很好地分清文字符号等的具体结构呢


import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib
# 关键改动: 强制 Matplotlib 使用非交互式后端, 避免在服务器或WSL中报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("--- Model Testing Script (Background Save Version) ---")
print("This script will generate comparison charts and save them as image files.")

# --- 1. Parameter Setup ---
# <-- 已更新为您提供的模型路径
MODEL_PATH = './model/rotnet_on_my_data_v2.pth' 
# <-- 请将您用于测试的图片放入这个文件夹
TEST_IMAGE_DIR = './data/test_images/'
# <-- 指定保存结果图片的文件夹
OUTPUT_DIR = './results/'
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Loading model weights from: {MODEL_PATH}")
print(f"Testing images from directory: {TEST_IMAGE_DIR}")
print(f"Results will be saved to: {OUTPUT_DIR}")


# --- 2. Model Definition (Must be identical to training) ---
class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationPredictionModel, self).__init__()
        self.encoder = models.resnet18(weights=None) 
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
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

# --- 3. Load Model ---
model = RotationPredictionModel(num_classes=4).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'. Please check the path.")
    exit()

model.eval()
print("Model loaded successfully and set to evaluation mode.")

# --- 4. Define Image Transformation for Testing ---
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 5. Iterate Through Test Folder, Predict, Correct, and Save ---
if not os.path.isdir(TEST_IMAGE_DIR):
    print(f"Error: Test image directory not found at '{TEST_IMAGE_DIR}'.")
    exit()
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_images = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
if not test_images:
    print(f"Warning: No image files found in '{TEST_IMAGE_DIR}'.")
    exit()
    
angles = [0, 90, 180, 270]

for image_name in test_images:
    image_path = os.path.join(TEST_IMAGE_DIR, image_name)
    
    try:
        original_image = Image.open(image_path).convert("RGB")
        input_tensor = test_transform(original_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
        
        _, predicted_idx = torch.max(output.data, 1)
        predicted_angle = angles[predicted_idx.item()]
        
        print(f"\nFile: {image_name}")
        print(f"  -> Predicted rotation: {predicted_angle} degrees")

        correction_angle = -predicted_angle
        corrected_image = original_image.rotate(correction_angle, expand=True)
        
        # --- 6. Generate and save the comparison chart ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot Original Image
        axes[0].imshow(original_image)
        axes[0].set_title(f"Input Image: {image_name}")
        axes[0].axis('off')
        
        # Plot Corrected Image
        axes[1].imshow(corrected_image)
        axes[1].set_title(f"Corrected by Model (Predicted {predicted_angle} deg)")
        axes[1].axis('off')
        
        plt.tight_layout()

        # Save the figure instead of showing it
        output_filename = f"comparison_{image_name}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free up memory
        print(f"  -> Comparison chart saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred while processing {image_name}: {e}")

print("\n--- All test images processed. ---")
