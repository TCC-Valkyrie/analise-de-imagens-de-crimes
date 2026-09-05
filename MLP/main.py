"""
========================================================================
 MLP (Perceptron Multicamadas) aplicado à detecção de risco em imagens
 TCC: Visão Computacional + ML para detecção de assalto/risco
========================================================================

Este script:
1. Gera um dataset SINTÉTICO (imagens simples) para você testar o
   pipeline funcionando AGORA, sem precisar do dataset real ainda.
2. Carrega e pré-processa as imagens (resize, escala de cinza, flatten).
3. Treina um MLP (equivalente ao Sequential+Dense do Keras).
4. Avalia o modelo (acurácia, matriz de confusão, relatório).
5. Salva gráficos e o modelo treinado.

Quando você tiver o dataset real de vigilância, é só trocar a função
`gerar_dataset_sintetico()` por `carregar_imagens_reais()` (já incluída
mais abaixo, comentada) apontando para as pastas com seus frames.

------------------------------------------------------------------------
EQUIVALÊNCIA COM KERAS (caso queira rodar com Keras/TensorFlow depois):

    MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu')

    é conceitualmente o mesmo que:

    Sequential([
        Dense(128, activation='relu', input_shape=(N,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import joblib

# ------------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ------------------------------------------------------------------------
IMG_SIZE = (64, 64)          # todas as imagens serão redimensionadas para isso
PASTA_DATASET = "dataset_real"
PASTA_SAIDA = "resultados_mlp_real"
N_IMAGENS_POR_CLASSE = 150   # quantidade de imagens sintéticas por classe
SEED = 42                    # fixa a aleatoriedade -> resultados reproduzíveis
                              # em qualquer máquina (importante para o TCC!)

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(PASTA_SAIDA, exist_ok=True)


# ------------------------------------------------------------------------
# ETAPA 1: GERAÇÃO DE UM DATASET SINTÉTICO (apenas para teste do pipeline)
# ------------------------------------------------------------------------
def gerar_dataset_sintetico():
    """
    Cria imagens artificiais simples:
      - Classe 'risco'  -> fundo escuro + uma forma vermelha/afiada (simula
                            "objeto/situação anômala" na cena)
      - Classe 'normal' -> fundo claro/uniforme, sem elementos estranhos

    Isso NÃO substitui seu dataset real de vigilância. É só um teste
    controlado para você ver o pipeline (MLP) funcionando de ponta a ponta.
    """
    pasta_risco = os.path.join(PASTA_DATASET, "risco")
    pasta_normal = os.path.join(PASTA_DATASET, "normal")
    os.makedirs(pasta_risco, exist_ok=True)
    os.makedirs(pasta_normal, exist_ok=True)

    print("Gerando dataset sintético...")

    for i in range(N_IMAGENS_POR_CLASSE):
        # ---- Classe RISCO: fundo escurecido + forma triangular/vermelha ----
        img = Image.new("RGB", IMG_SIZE, color=tuple(np.random.randint(20, 70, 3)))
        draw = ImageDraw.Draw(img)
        x, y = random.randint(10, 40), random.randint(10, 40)
        tam = random.randint(10, 20)
        draw.polygon(
            [(x, y), (x + tam, y), (x + tam // 2, y - tam)],
            fill=(200 + random.randint(-20, 20), 20, 20),
        )
        # ruído leve para não ficar "perfeito" demais
        ruido = (np.random.randn(*IMG_SIZE, 3) * 8).astype(np.int16)
        arr = np.array(img).astype(np.int16) + ruido
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(pasta_risco, f"risco_{i}.png"))

        # ---- Classe NORMAL: fundo claro e mais uniforme ----
        cor_base = np.random.randint(150, 220, 3)
        img2 = Image.new("RGB", IMG_SIZE, color=tuple(cor_base))
        ruido2 = (np.random.randn(*IMG_SIZE, 3) * 8).astype(np.int16)
        arr2 = np.array(img2).astype(np.int16) + ruido2
        arr2 = np.clip(arr2, 0, 255).astype(np.uint8)
        Image.fromarray(arr2).save(os.path.join(pasta_normal, f"normal_{i}.png"))

    print(f"Dataset sintético criado em: {PASTA_DATASET}/")


# ------------------------------------------------------------------------
# ETAPA 2: CARREGAMENTO E PRÉ-PROCESSAMENTO DAS IMAGENS
# ------------------------------------------------------------------------
def carregar_imagens(pasta, label):
    """
    Lê todas as imagens de uma pasta, redimensiona, converte para escala
    de cinza, normaliza (0-1) e achata (flatten) em um vetor 1D.
    """
    X, y = [], []
    for arquivo in sorted(os.listdir(pasta)):
        caminho = os.path.join(pasta, arquivo)
        try:
            img = Image.open(caminho).convert("L")   # "L" = escala de cinza
            img = img.resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            X.append(arr.flatten())
            y.append(label)
        except Exception as e:
            print(f"  [aviso] não consegui ler {caminho}: {e}")
    return X, y


# --- Quando for usar seu dataset REAL de vigilância, use esta função: ---
def carregar_imagens_reais(pasta_base):
    """
    Espera a seguinte estrutura de pastas:
        pasta_base/
            risco/   -> frames com assalto/situação de risco
            normal/  -> frames de cenas normais

    Uso:
        X, y = carregar_imagens_reais("meu_dataset_vigilancia")
    """
    X_r, y_r = carregar_imagens(os.path.join(pasta_base, "risco"), 1)
    X_n, y_n = carregar_imagens(os.path.join(pasta_base, "normal"), 0)
    return X_r + X_n, y_r + y_n


# ------------------------------------------------------------------------
# ETAPA 3: TREINAMENTO DO MLP
# ------------------------------------------------------------------------
def treinar_mlp(X_treino, y_treino):
    """
    hidden_layer_sizes=(128, 64) -> duas camadas ocultas, 128 e 64 neurônios
    (equivalente ao Dense(128) + Dense(64) do Keras que vimos antes)
    """
    modelo = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=SEED,
        early_stopping=True,     # para o treino se parar de melhorar (evita overfitting)
        validation_fraction=0.15,
        n_iter_no_change=20,     # dá mais chance ao otimizador antes de parar
        verbose=False,
    )
    print("Treinando o MLP...")
    modelo.fit(X_treino, y_treino)
    print(f"Treino finalizado em {modelo.n_iter_} épocas (early stopping).")
    return modelo


# ------------------------------------------------------------------------
# ETAPA 4: AVALIAÇÃO
# ------------------------------------------------------------------------
def avaliar_modelo(modelo, X_teste, y_teste):
    y_pred = modelo.predict(X_teste)

    acc = accuracy_score(y_teste, y_pred)
    print(f"\nAcurácia no conjunto de teste: {acc*100:.2f}%\n")
    print("Relatório de classificação:")
    print(classification_report(y_teste, y_pred, target_names=["normal", "risco"]))

    # Matriz de confusão
    cm = confusion_matrix(y_teste, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "risco"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusão - MLP")
    plt.tight_layout()
    caminho_cm = os.path.join(PASTA_SAIDA, "matriz_confusao.png")
    plt.savefig(caminho_cm, dpi=150)
    plt.close()
    print(f"Matriz de confusão salva em: {caminho_cm}")

    # Curva de perda durante o treino (loss por época)
    plt.figure(figsize=(6, 4))
    plt.plot(modelo.loss_curve_)
    plt.title("Curva de perda (loss) durante o treino")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    caminho_loss = os.path.join(PASTA_SAIDA, "curva_loss.png")
    plt.savefig(caminho_loss, dpi=150)
    plt.close()
    print(f"Curva de loss salva em: {caminho_loss}")

    return acc


# ------------------------------------------------------------------------
# ETAPA 5: TESTE COM UMA IMAGEM NOVA (simulando uso real)
# ------------------------------------------------------------------------
def prever_imagem(modelo, scaler, caminho_imagem):
    img = Image.open(caminho_imagem).convert("L").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32).flatten() / 255.0
    arr = scaler.transform([arr])   # aplica a MESMA padronização usada no treino
    pred = modelo.predict(arr)[0]
    prob = modelo.predict_proba(arr)[0]
    classe = "RISCO" if pred == 1 else "NORMAL"
    print(f"\nImagem: {caminho_imagem}")
    print(f"Classificação: {classe}  |  Probabilidades -> normal: {prob[0]:.2f}, risco: {prob[1]:.2f}")


# ------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Gera dataset sintético (comente esta linha quando for usar dados reais)
    # if not os.path.exists(PASTA_DATASET):
    #     gerar_dataset_sintetico()
    # else:
    #     print(f"Usando dataset já existente em: {PASTA_DATASET}/")

    # 2. Carrega e pré-processa
    print("\nCarregando imagens...")
    X_risco, y_risco = carregar_imagens(os.path.join(PASTA_DATASET, "risco"), 1)
    X_normal, y_normal = carregar_imagens(os.path.join(PASTA_DATASET, "normal"), 0)

    X = np.array(X_risco + X_normal)
    y = np.array(y_risco + y_normal)
    print(f"Total de imagens carregadas: {len(X)}  |  risco: {len(X_risco)}  normal: {len(X_normal)}")

    # 3. Separa treino/teste (80/20, mantendo proporção das classes)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 3.1 Padroniza os dados (média 0, desvio padrão 1).
    #     Isso ajuda MUITO o otimizador do MLP a não ficar "preso" prevendo
    #     sempre a classe majoritária -- problema comum quando o sinal
    #     relevante da imagem (ex: um objeto pequeno) fica diluído entre
    #     milhares de valores de fundo/ruído após o flatten.
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)

    # 4. Treina
    modelo = treinar_mlp(X_treino, y_treino)

    # 5. Avalia
    avaliar_modelo(modelo, X_teste, y_teste)

    # 6. Salva o modelo treinado E o scaler (para reusar depois sem retreinar)
    caminho_modelo = os.path.join(PASTA_SAIDA, "modelo_mlp.pkl")
    caminho_scaler = os.path.join(PASTA_SAIDA, "scaler.pkl")
    joblib.dump(modelo, caminho_modelo)
    joblib.dump(scaler, caminho_scaler)
    print(f"\nModelo salvo em: {caminho_modelo}")
    print(f"Scaler salvo em: {caminho_scaler}")

    # 7. Exemplo de uso: classificar uma imagem individual
    exemplo = os.path.join(PASTA_DATASET, "risco", "risco_0.png")
    if os.path.exists(exemplo):
        prever_imagem(modelo, scaler, exemplo)

    print("\nConcluído. Veja os gráficos em:", PASTA_SAIDA)