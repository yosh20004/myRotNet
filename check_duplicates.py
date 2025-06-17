import os
from PIL import Image
import imagehash
from collections import defaultdict
import numpy as np

def calculate_image_hashes(image_path):
    try:
        with Image.open(image_path) as img:
            # 计算多种哈希值
            ahash = imagehash.average_hash(img)
            phash = imagehash.phash(img)
            dhash = imagehash.dhash(img)
            whash = imagehash.whash(img)
            return {
                'average': ahash,
                'perceptual': phash,
                'difference': dhash,
                'wavelet': whash
            }
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None

def hash_similarity(hash1, hash2):
    # 计算两个哈希值之间的汉明距离
    return 1 - (hash1 - hash2) / len(hash1.hash) ** 2

def find_duplicates(similarity_threshold=0.85):
    # 训练集和测试集的路径
    train_dir = './data/image'
    test_dir = './data/test_images'
    
    # 存储所有图片的哈希值
    train_hashes = []
    test_hashes = []
    
    # 处理训练集图片
    print("正在处理训练集图片...")
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                hashes = calculate_image_hashes(img_path)
                if hashes:
                    train_hashes.append((img_path, hashes))
    
    # 处理测试集图片
    print("正在处理测试集图片...")
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, file)
            hashes = calculate_image_hashes(img_path)
            if hashes:
                test_hashes.append((img_path, hashes))
    
    # 检查重复
    print("\n检查结果:")
    found_duplicates = False
    
    # 检查测试集和训练集之间的相似度
    for test_path, test_hash_dict in test_hashes:
        for train_path, train_hash_dict in train_hashes:
            # 计算所有哈希算法的平均相似度
            similarities = []
            for hash_type in ['average', 'perceptual', 'difference', 'wavelet']:
                similarity = hash_similarity(test_hash_dict[hash_type], train_hash_dict[hash_type])
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            
            if avg_similarity >= similarity_threshold:
                found_duplicates = True
                print(f"\n发现相似图片 (相似度: {avg_similarity:.2f}):")
                print(f"- 测试集: {test_path}")
                print(f"- 训练集: {train_path}")
                print(f"  各算法相似度:")
                for hash_type, sim in zip(['average', 'perceptual', 'difference', 'wavelet'], similarities):
                    print(f"  - {hash_type}: {sim:.2f}")
    
    if not found_duplicates:
        print("未发现相似图片")

if __name__ == "__main__":
    find_duplicates() 