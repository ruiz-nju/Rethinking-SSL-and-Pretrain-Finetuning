import os
from PIL import Image
from tqdm import tqdm

size = 16
root_dir = f'path/to/stl10_{size}/train'
# 遍历每个子文件夹
for class_name in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_dir):
        continue  # 跳过非文件夹项

    # 遍历该类别下的所有文件
    for fname in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # 只处理图片文件

        img_path = os.path.join(class_dir, fname)
        try:
            # 打开图片
            img = Image.open(img_path)
            resized = img.resize((size, size)) # 默认使用 PIL.Image.BILINEAR
            # 覆盖保存（也可以保存到新路径）
            resized.save(img_path)
            print(f"Resized: {img_path}")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
