import os
from pathlib import Path

def generate_yaml(source_dir, output_dir):
    class_names = {}
    source_path = Path(source_dir)
    
    # Filter only directories that match the pattern "ID_Name"
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        dir_name = class_dir.name
        if dir_name == "dataset_train_vaild_test" or dir_name == "yolov11_dataset":
            continue
            
        parts = dir_name.split('_', 1)
        if len(parts) == 2 and parts[0].isdigit():
            class_id = int(parts[0])
            class_name = parts[1]
            class_names[class_id] = class_name

    dataset_yaml_content = f"""path: {os.path.abspath(output_dir)}
train: train/images
val: valid/images
test: test/images
names:
"""
    for cid in sorted(class_names.keys()):
        dataset_yaml_content += f"  {cid}: {class_names[cid]}\n"
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(dataset_yaml_content)
        
    print(f"Generated data.yaml with {len(class_names)} classes.")

if __name__ == "__main__":
    source = r"c:\Users\huang\Desktop\Tibame_Class\project\model\ingredients_dataset\class100_yolov11"
    destination = r"c:\Users\huang\Desktop\Tibame_Class\project\model\ingredients_dataset\class100_yolov11\dataset_train_vaild_test"
    generate_yaml(source, destination)
