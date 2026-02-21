## Sentiment Analysis Model
**Descrição Geral** 📄<br>
Este projeto apresenta um **modelo de classificação de sentimentos**, desenvolvido com **Python** e técnicas de **Processamento de Linguagem Natural (NLP)**. O sistema realiza **limpeza de texto, transformação utilizando TF-IDF, treinamento de modelo supervisionado**, além de exibir métricas de avaliação e gerar previsões para novos reviews.
O projeto demonstra conceitos essenciais de **pré-processamento de texto, machine learning, otimização de hiperparâmetros e avaliação de modelos de classificação.**

---
**Objetivo** 🎯 <br> 
O objetivo principal do projeto é construir uma **ferramenta prática para classificação automática de sentimentos** (positivo ou negativo), aplicando métodos de NLP e machine learning para analisar textos de reviews.
O modelo resultante pode ser utilizado em cenários como análise de feedbacks, monitoramento de reputação ou suporte ao cliente.

---
**Tecnologias Utilizadas** 💻 <br>
* ***Python*** - linguagem principal.
* ***Pandas*** - manipulação e análise do dataset.
* ***NumPy*** - operações auxiliares.
* ***Matplotlib / Seaborn*** - visualização de gráficos.
* ***Scikit-learn*** - machine learning, pré-processamento e grid search.
* ***Joblib*** - salvamento e carregamento do modelo treinado.

---
**Arquitetura e Estrutura do Código** 🧱 <br><br>
***1. Script Principal (sentiment_analysis_model.py)*** <br>
Responsável por:
* ***Carregar o dataset e verificar estrutura e valores ausentes.*** 
* ***Realizar limpeza textual completa (remoção de acentos, símbolos e normalização para minúsculas).***
* ***Criar nova coluna com texto pré-processado.***
* ***Transformar texto em vetores TF-IDF.***
* ***Montar pipeline de treinamento usando Logistic Regression.***
* ***Otimizar parâmetros via GridSearchCV.***
* ***Avaliar o modelo (acurácia, relatório de classificação, matriz de confusão).***
* ***Salvar o modelo final em arquivo .joblib.***
* ***Carregar o modelo salvo e fazer previsões em novos reviews.***

---
**Conceitos e Funcionalidades Demonstradas** 🔍 <br><br>
✅ ***Pré-processamento de texto (NLP):*** <br>
Conversão para minúsculas, remoção de acentos, símbolos, números e espaços extras.

✅***Vetorização de texto:*** <br>
Conversão dos reviews em vetores numéricos usando **TfidfVectorizer.**

✅***Pipeline de Machine Learning:*** <br>
Encadeamento das etapas de preparação + modelo dentro de um único fluxo.

✅***Otimização de hiperparâmetros:*** <br>
Busca dos melhores valores via **GridSearchCV.**

✅***Avaliação do modelo:*** <br>
Acurácia, matriz de confusão e relatório de classificação.

✅***Deploy simples:*** <br>
Carregamento do modelo treinado e previsão em novos textos.

---
**Como Executar o Projeto** ▶️ <br><br>
***1. Instale as dependências (recomendado via requirements.txt):*** <br>
```pip install -r requirements.txt```

***2. Certifique-se de que o dataset está no mesmo diretório:*** <br>
```dataset.csv```

***3. Execute o script principal:*** <br>
```python sentiment_analysis_model.py```

***4. Veja as métricas e as previsões geradas.*** <br>

***Exemplo de saída:*** <br>
```
Tamanho do DataFrame original: 5000
Tamanho após limpeza de nulos: 4980

Melhores hiperparâmetros:
{'tfidf__max_features': 2000, 'tfidf__ngram_range': (1, 2), 'logreg__C': 1}

Acurácia no Modelo: 89.75%

Relatório de Classificação:
              precision    recall  f1-score   support
Negativo        0.87       0.91       0.89      1200
Positivo        0.92       0.88       0.90      1250
```

---
**Conclusão** 📌 <br>
Este projeto demonstra como desenvolver um **modelo completo de análise de sentimentos**, desde o pré-processamento do texto até a avaliação e deploy do modelo.
Ele integra **NLP, vetorização de texto, machine learning supervisionado e otimização automática**, oferecendo uma estrutura robusta e reutilizável para aplicações reais de classificação de sentimentos.
