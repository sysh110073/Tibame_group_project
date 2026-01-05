import os
import shutil
import random
import glob
from pathlib import Path

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    random.seed(42)  # Ensure reproducibility

    # Define strict output directories (YOLO standard)
    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, 'labels'), exist_ok=True)

    # Class mapping
    class_names = {}
    
    # Iterate over class directories (1_salt, 2_pork, etc.)
    # We assume directories are in `source_dir`
    source_path = Path(source_dir)
    
    # Filter only directories that match the pattern "ID_Name"
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    all_images = []
    
    print(f"Scanning directories in {source_dir}...")
    
    for class_dir in class_dirs:
        # Parse Class ID and Name from directory name
        dir_name = class_dir.name
        
        # Skip output dir if it's in the same folder
        if dir_name == "dataset_train_vaild_test":
            continue
            
        parts = dir_name.split('_', 1)
        if len(parts) == 2 and parts[0].isdigit():
            class_id = int(parts[0])
            class_name = parts[1]
            class_names[class_id] = class_name
        else:
            # If directory doesn't follow "NUM_NAME" split, we might skip or handle differently
            # For now, let's assume valid directories follow pattern, others are skipped (like 'images' if any)
            print(f"Skipping directory/file: {dir_name} (doesn't match ID_Name format)")
            continue

        # Find images in `images` subdir
        img_dir = class_dir / 'images'
        lbl_dir = class_dir / 'labels'
        
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Skipping {dir_name}: missing images or labels subdir")
            continue
            
        # Supported extensions
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        
        class_files = []
        for ext in valid_exts:
            class_files.extend(list(img_dir.glob(f'*{ext}')))
            
        print(f"Found {len(class_files)} images in {dir_name}")
        
        for img_file in class_files:
            # Check for corresponding label
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                all_images.append((img_file, lbl_file))
            else:
                print(f"Warning: Label not found for {img_file.name}")

    # Shuffle and split
    random.shuffle(all_images)
    
    total_files = len(all_images)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count
    
    train_set = all_images[:train_count]
    val_set = all_images[train_count:train_count+val_count]
    test_set = all_images[train_count+val_count:]
    
    print(f"Total images: {total_files}")
    print(f"Train: {len(train_set)}, Valid: {len(val_set)}, Test: {len(test_set)}")
    
    # Helper to copy
    def copy_files(file_list, subset_name):
        for img_path, lbl_path in file_list:
            shutil.copy2(img_path, os.path.join(output_dir, subset_name, 'images', img_path.name))
            shutil.copy2(lbl_path, os.path.join(output_dir, subset_name, 'labels', lbl_path.name))

    print("Copying files...")
    copy_files(train_set, 'train')
    copy_files(val_set, 'valid')
    copy_files(test_set, 'test')
    
    # Generate data.yaml
    # We sort class names by ID to ensure mapping is clean, though dict is supported
    # If IDs are sparse (e.g. 1, 2, ..., no 0), we can just pass the dict.
    
    # Generate data.yaml (Manual write to avoid pyyaml dependency)
    dataset_yaml_content = f"""path: {os.path.abspath(output_dir)}
train: train/images
val: valid/images
test: test/images
names:
"""
    # Sort by ID
    for cid in sorted(class_names.keys()):
        dataset_yaml_content += f"  {cid}: {class_names[cid]}\n"
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(dataset_yaml_content)
        
    print(f"Data split complete. Output saved to {output_dir}")
    print(f"Generated data.yaml with {len(class_names)} classes.")

if __name__ == "__main__":
    source = r"c:\Users\huang\Desktop\Tibame_Class\project\model\ingredients_dataset\class100_yolov11"
    destination = r"c:\Users\huang\Desktop\Tibame_Class\project\model\ingredients_dataset\class100_yolov11\dataset_train_vaild_test"
    
    split_dataset(source, destination)
