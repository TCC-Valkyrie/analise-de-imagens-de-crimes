import os
import random
import shutil
from pathlib import Path

def create_dataset_subset(source_dir, output_dir, num_images=1000):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Estrutura esperada do YOLO: source_dir/train/images e source_dir/train/labels
    train_img_dir = source_path / "train" / "images"
    train_lbl_dir = source_path / "train" / "labels"
    
    if not train_img_dir.exists():
        print(f"Erro: Pasta de treino não encontrada em {train_img_dir}")
        return
        
    out_train_img = output_path / "train" / "images"
    out_train_lbl = output_path / "train" / "labels"
    
    # Criar pastas de destino
    out_train_img.mkdir(parents=True, exist_ok=True)
    out_train_lbl.mkdir(parents=True, exist_ok=True)
    
    # Listar todas as imagens de treino
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [p for p in train_img_dir.iterdir() if p.suffix.lower() in image_extensions]
    
    print(f"Total de imagens disponíveis no treino original: {len(all_images)}")
    
    # Selecionar aleatoriamente o número desejado (ou pegar todas se houver menos que o limite)
    sample_size = min(num_images, len(all_images))
    sampled_images = random.sample(all_images, sample_size)
    
    print(f"Selecionando aleatoriamente {sample_size} imagens...")
    
    copied_count = 0
    for img_path in sampled_images:
        # Copiar a imagem
        shutil.copy(img_path, out_train_img / img_path.name)
        
        # Encontrar e copiar o arquivo de label correspondente (.txt)
        label_name = img_path.stem + ".txt"
        lbl_path = train_lbl_dir / label_name
        
        if lbl_path.exists():
            shutil.copy(lbl_path, out_train_lbl / label_name)
        
        copied_count += 1
        
    # Copiar as pastas 'valid' e 'test' inteiras (para manter a validação correta)
    for split in ["valid", "test"]:
        split_src = source_path / split
        if split_src.exists():
            split_dest = output_path / split
            if split_dest.exists():
                shutil.rmtree(split_dest)
            shutil.copytree(split_src, split_dest)
            print(f"Pasta '{split}' copiada integralmente.")
            
    # Copiar o arquivo data.yaml
    yaml_path = source_path / "data.yaml"
    if yaml_path.exists():
        out_yaml = output_path / "data.yaml"
        shutil.copy(yaml_path, out_yaml)
        print(f"Arquivo data.yaml copiado para {out_yaml}")
        
    print(f"\nSubset criado com sucesso! Pasta gerada: '{output_path.absolute()}'")

if __name__ == "__main__":
    create_dataset_subset(
        source_dir="dataset-treino-novo",         # Nome da pasta atual do seu dataset completo
        output_dir="dataset_1000",    # Nome da nova pasta que será criada com o recorte
        num_images=1000               # Quantidade de imagens que você quer no treino
    )