"""
========================================================================
 Preparação do dataset REAL para o MLP (risco vs normal)
========================================================================

Este script combina fontes de dados em uma única estrutura de pastas
pronta para o mlp_deteccao_risco.py:

    dataset_real/
        risco/    <- imagens de burglary, fighting, robbery (dataset CDS)
        normal/   <- imagens sem ocorrência de crime

FONTE DE RISCO - CDS (Roboflow, formato YOLO/Detecção de objetos):
    cds-gc5yj/
        train/images/*.jpg   + train/labels/*.txt
        valid/images/*.jpg   + valid/labels/*.txt
        test/images/*.jpg    + test/labels/*.txt

FONTES DE NORMAL (pode usar uma ou combinar as duas):

  A) Dataset CCTV genérico (Roboflow, também formato Detecção de
     objetos, ex: k52p/cctv-yzk4h) -- mesma estrutura train/valid/test
     acima. Como não tem classes de crime, tratamos toda imagem como
     'normal'. RECOMENDADO: dê uma olhada visual em uma amostra das
     imagens antes de usar, para confirmar que realmente são cenas
     cotidianas sem ocorrências (documente isso no TCC).

  B) Dataset harmesti95/violence-nonviolence (Hugging Face), já
     preparado com baixar_normal_huggingface.py -> pasta única de .jpg
     (sem estrutura train/valid/test).

Ajuste os caminhos abaixo (CAMINHO_CDS, CAMINHO_NORMAL_CCTV,
CAMINHO_NORMAL_HF) para onde você extraiu/gerou cada um. Deixe como
None qualquer fonte de 'normal' que você não for usar.
------------------------------------------------------------------------
"""

import os
import shutil
import random
from glob import glob

# ------------------------------------------------------------------------
# AJUSTE ESTES CAMINHOS PARA O SEU COMPUTADOR
# ------------------------------------------------------------------------
CAMINHO_CDS = "MLP/path_cds"              # dataset de risco (burglary/fighting/robbery)

CAMINHO_NORMAL_CCTV = "MLP/path_normal_cctv"     # dataset CCTV genérico (formato detecção),
                                        # ou None se não for usar essa fonte

CAMINHO_NORMAL_HF = None               # pasta gerada pelo baixar_normal_huggingface.py
                                        # (ex: "crime-duvoy/normal_hf"), ou None

PASTA_SAIDA = "dataset_real"
LIMITE_POR_CLASSE = 400   # None = usa tudo; ou um número (ex: 1500) para
                            # limitar e acelerar o treino/teste inicial

SEED = 42
random.seed(SEED)


def coletar_imagens_yolo_export(caminho_base):
    """
    Junta as imagens de train/valid/test de um dataset exportado no
    formato de Detecção de objetos (YOLO) do Roboflow. Serve tanto para
    o CDS (risco) quanto para qualquer outro dataset no mesmo formato
    (ex: o CCTV genérico) -- ignoramos os arquivos de anotação (.txt),
    só usamos as imagens.
    """
    extensoes = ["*.jpg", "*.jpeg", "*.png"]
    arquivos = []
    for split in ["train", "valid", "test"]:
        for ext in extensoes:
            arquivos.extend(glob(os.path.join(caminho_base, split, "images", ext)))
    return arquivos


def coletar_imagens_pasta_simples(caminho_base):
    """Para fontes que já vêm como uma pasta única de imagens (ex: o
    resultado do baixar_normal_huggingface.py)."""
    if caminho_base is None:
        return []
    extensoes = ["*.jpg", "*.jpeg", "*.png"]
    arquivos = []
    for split in ["train", "test"]:
        for ext in extensoes:
            arquivos.extend(glob(os.path.join(caminho_base, split, "images", ext)))
    return arquivos


def copiar_para(lista_arquivos, pasta_destino, limite=None):
    os.makedirs(pasta_destino, exist_ok=True)
    if limite is not None and len(lista_arquivos) > limite:
        lista_arquivos = random.sample(lista_arquivos, limite)

    for i, origem in enumerate(lista_arquivos):
        ext = os.path.splitext(origem)[1]
        destino = os.path.join(pasta_destino, f"img_{i:05d}{ext}")
        shutil.copyfile(origem, destino)
    return len(lista_arquivos)


if __name__ == "__main__":
    print("Coletando imagens de risco (CDS)...")
    imgs_risco = coletar_imagens_yolo_export(CAMINHO_CDS)
    print(f"  Encontradas: {len(imgs_risco)} imagens")

    print("Coletando imagens normais...")
    imgs_normal = []
    if CAMINHO_NORMAL_CCTV:
        imgs_cctv = coletar_imagens_yolo_export(CAMINHO_NORMAL_CCTV)
        print(f"  CCTV genérico: {len(imgs_cctv)} imagens")
        imgs_normal.extend(imgs_cctv)
    if CAMINHO_NORMAL_HF:
        imgs_hf = coletar_imagens_pasta_simples(CAMINHO_NORMAL_HF)
        print(f"  Hugging Face (Non-violence): {len(imgs_hf)} imagens")
        imgs_normal.extend(imgs_hf)
    print(f"  Total normal: {len(imgs_normal)} imagens")

    if len(imgs_risco) == 0 or len(imgs_normal) == 0:
        print("\n[ERRO] Uma das listas de imagens ficou vazia. Verifique os")
        print("caminhos (CAMINHO_CDS / CAMINHO_NORMAL_CCTV / CAMINHO_NORMAL_HF)")
        print("e se a estrutura de pastas bate com o que o script espera.")
        raise SystemExit(1)

    # Balanceamento: como o MLPClassifier do sklearn não aceita pesos por
    # classe, o jeito mais simples de evitar viés é igualar as quantidades.
    limite = LIMITE_POR_CLASSE or min(len(imgs_risco), len(imgs_normal))
    print(f"\nBalanceando para {limite} imagens por classe...")

    n_risco = copiar_para(imgs_risco, os.path.join(PASTA_SAIDA, "risco"), limite)
    n_normal = copiar_para(imgs_normal, os.path.join(PASTA_SAIDA, "normal"), limite)

    print(f"\nPronto! Dataset final em '{PASTA_SAIDA}/':")
    print(f"  risco:  {n_risco} imagens")
    print(f"  normal: {n_normal} imagens")
    print("\nAgora aponte PASTA_DATASET = 'dataset_real' no mlp_deteccao_risco.py")
    print("e comente a chamada de gerar_dataset_sintetico() no bloco principal.")