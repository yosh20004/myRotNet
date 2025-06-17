import os
from datetime import datetime

def rename_images(directory):
    # 获取目录下所有文件
    files = os.listdir(directory)
    
    # 只处理jpg文件
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    
    # 按时间戳排序
    jpg_files.sort()
    
    # 生成新的文件名
    for index, old_name in enumerate(jpg_files, start=1):
        # 生成新的文件名格式：image_001.jpg, image_002.jpg 等
        new_name = f'image_{index:03d}.jpg'
        
        # 构建完整的文件路径
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        
        # 重命名文件
        try:
            os.rename(old_path, new_path)
            print(f'已重命名: {old_name} -> {new_name}')
        except Exception as e:
            print(f'重命名 {old_name} 时出错: {str(e)}')

if __name__ == '__main__':
    # 指定图片目录
    image_dir = 'data/test_images'
    
    # 确认目录存在
    if not os.path.exists(image_dir):
        print(f'错误：目录 {image_dir} 不存在')
    else:
        print('开始重命名图片...')
        rename_images(image_dir)
        print('重命名完成！') 