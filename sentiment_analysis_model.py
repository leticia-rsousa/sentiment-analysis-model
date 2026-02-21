# Bibliotecas para manipulação de dados e visualização
import re
import pandas as pd
import numpy as np
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt

# Bibliotecas para pré-processamento e machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Configuração estética dos gráficos
sns.set_style("whitegrid")

# Carregamento do dataset
nome_arquivo_csv = 'dataset.csv'
df = pd.read_csv(nome_arquivo_csv)

print(df.shape)
print(df.head())
print(df.sample(10))
print(df.tail())

# Informações gerais do DataFrame
print(df.info())
print("\nVerificando valores ausentes:\n")
print(df.isnull().sum())

# Distribuição das classes alvo
sns.countplot(x='sentimento', data=df)
plt.title('Distribuição das Classes de Sentimento')
plt.show()

# Remoção de linhas com textos nulos
print(f"\nTamanho original do DataFrame: {len(df)}")
df.dropna(subset=['texto_review'], inplace=True)
print(f"Tamanho do DataFrame após remover nulos: {len(df)}\n")
print(df.shape)
print(df.head())

# Função para limpeza textual
def limpa_texto(texto):
    """
    Limpa o texto realizando:
    - padronização para minúsculas
    - remoção de acentos
    - remoção de caracteres não alfabéticos
    - eliminação de espaços extras
    """
    if not isinstance(texto, str):
        return ""

    texto_sem_acentos = ''.join(
        c for c in unicodedata.normalize('NFKD', texto)
        if unicodedata.category(c) != 'Mn'
    )

    texto_limpo = texto_sem_acentos.lower()
    texto_limpo = re.sub(r'[^a-z\s]', '', texto_limpo)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()

    return texto_limpo

# Aplicação da limpeza no DataFrame
df['texto_limpo'] = df['texto_review'].apply(limpa_texto)
print(df.head())

# Conversão das classes para valores numéricos
df['sentimento_label'] = df['sentimento'].map({'positivo': 1, 'negativo': 0})

print("\nDataFrame após a limpeza e mapeamento:\n")
print(df[['texto_limpo', 'sentimento_label']].head())

# Separação entre variáveis independentes e dependente
X = df['texto_limpo']
y = df['sentimento_label']

# Divisão treino/teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Pipeline com TF-IDF, normalização e modelo
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um'])),
    ('scaler', StandardScaler(with_mean=False)),
    ('logreg', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)),
])

# Hiperparâmetros para Grid Search
parametros_grid = {
    'tfidf__max_features': [500, 1000, 2000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'logreg__C': [0.01, 0.1, 1, 10]
}

# Otimização dos hiperparâmetros
grid_search = GridSearchCV(
    pipeline,
    parametros_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

print("\nIniciando o treinamento do modelo com otimização de hiperparâmetros...\n")
print(grid_search.fit(X_treino, y_treino))

print("\nMelhores hiperparâmetros encontrados:\n")
print(grid_search.best_params_)

# Seleção do melhor modelo
melhor_modelo = grid_search.best_estimator_

# Avaliação no conjunto de teste
y_pred = melhor_modelo.predict(X_teste)

acuracia = accuracy_score(y_teste, y_pred)
report = classification_report(y_teste, y_pred, target_names=['Negativo', 'Positivo'])

print(f"\nAcurácia no Modelo: {acuracia:.2%}\n")
print("Relatório de Classificação:\n")
print(report)

# Matriz de confusão
cm = confusion_matrix(y_teste, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Negativo', 'Positivo'],
    yticklabels=['Negativo', 'Positivo']
)
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

# Salvando modelo treinado
joblib.dump(melhor_modelo, 'modelo_sentimento_v1.joblib')

# Remove o modelo da memória
del melhor_modelo

# Carrega o modelo salvo para simulação de deploy
modelo_deploy = joblib.load('modelo_sentimento_v1.joblib')

# Exemplos de textos para previsão
novos_reviews = [
    "A bateria do celular não dura nada, péssima compra.",
    "Chegou antes do prazo e o produto é de ótima qualidade! Estou muito feliz.",
    "O serviço de atendimento foi rápido e eficiênte.",
    "Não recomendo, veio faltando peças e a cor estava errada."
]

# Função de previsão usando o modelo carregado
def prever_sentimento(review):
    """
    Recebe uma lista de reviews e retorna a classificação
    utilizando o modelo salvo.
    """
    previsoes = modelo_deploy.predict(review)
    sentimentos = ['Negativo' if p == 0 else 'Positivo' for p in previsoes]

    for review, sentimento in zip(review, sentimentos):
        print(f"\nReview: '{review}\nSentimento Previsto: '{sentimento}'\n")

# Execução da previsão em modo "deploy"
print("\n--- Iniciando Classificação de Novos Reviews (Deploy com Pipeline Completo) ---\n")
prever_sentimento(novos_reviews)
