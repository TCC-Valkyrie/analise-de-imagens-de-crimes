from pathlib import Path

def create_labels_from_images(images_dir_path, output_labels_dir_path):
    images_dir = Path(images_dir_path)
    labels_dir = Path(output_labels_dir_path)
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for file_path in images_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            txt_path = labels_dir / f"{file_path.stem}.txt"
            txt_path.touch()

if __name__ == "__main__":
    IMAGES_FOLDER = "dataset_faces/train/images"
    LABELS_FOLDER = "dataset_faces/train/labels"
    
    create_labels_from_images(IMAGES_FOLDER, LABELS_FOLDER)