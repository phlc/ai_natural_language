# Tutorial from https://www.kaggle.com/code/leandrodoze/sentiment-analysis-in-portuguese/notebook

import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

# Primeiro, vamos contar a quantidade total de registros
dataset = pd.read_csv('ignore_tweets.csv',encoding='utf-8')
print(dataset.count())

# Agora, apenas os classificados como neutro
print(dataset[dataset.Classificacao == 'Neutro'].count())

# Os classificados como positivo
print(dataset[dataset.Classificacao == 'Positivo'].count())

# E finalmente, os classificados como negativo
print(dataset[dataset.Classificacao == 'Negativo'].count())

# OK, vamos dar uma olhada rápida no conteúdo do dataset para finalizar esse passo
print(dataset.head())

# Próximo passo, vamos separar os tweets e suas classes
tweets = dataset["Text"].values
print(tweets)

classes = dataset["Classificacao"].values
print(classes)

# Agora, vamos treinar o modelo usando a abordagem Bag of Words e o algoritmo Naive Bayes Multinomial
#    - Bag of Words, na prática, cria um vetor com cada uma das palavras do texto completo da base,
#      depois, calcula a frequência em que essas palavras ocorrem em uma data sentença, para então
#      classificar/treinar o modelo
#    - Exemplo HIPOTÉTICO de três sentenças vetorizadas "por palavra" e classificadas baseada na
#      frequência de suas palavras:
#         {0,3,2,0,0,1,0,0,0,1, Positivo}
#         {0,0,1,0,0,1,0,1,0,0, Negativo}
#         {0,1,1,0,0,1,0,0,0,0, Neutro}
#    - Olhando para esses vetores, meu palpite é que as palavras nas posições 2 e 3 são as com maior
#      peso na determinação de a que classe pertence cada uma das três sentenças avaliadas
#    - A função fit_transform faz exatamente esse processo: ajusta o modelo, aprende o vocabulário,
#      e transforma os dados de treinamento em feature vectors, a.k.a. vetor com frequêcia das palavras

vectorizer = CountVectorizer(analyzer = "word")
freq_tweets = vectorizer.fit_transform(tweets)

modelo = MultinomialNB()
modelo.fit(freq_tweets, classes)

# Vamos usar algumas frases de teste para fazer a classificação com o modelo treinado
testes = ["Esse governo está no início, vamos ver o que vai dar",
          "Estou muito feliz com o governo de São Paulo esse ano",
          "O estado de Minas Gerais decretou calamidade financeira!!!",
          "A segurança desse país está deixando a desejar",
          "O governador de Minas é do PT",
          "O prefeito de São Paulo está fazendo um ótimo trabalho"]

freq_testes = vectorizer.transform(testes)
print(modelo.predict(freq_testes))


# Validação cruzada do modelo. Neste caso, o modelo é dividido em 10 partes, treinado em 9 e testado em 1
resultados = cross_val_predict(modelo, freq_tweets, classes, cv = 10)
print(resultados)

# Quão acurada é a média do modelo?
print(metrics.accuracy_score(classes, resultados))

# Medidas de validação do modelo
sentimentos = ["Positivo", "Negativo", "Neutro"]
print(metrics.classification_report(classes, resultados, labels=sentimentos))

# Lembrando que:
#    : precision = true positive / (true positive + false positive)
#    : recall    = true positive / (true positive + false negative)
#    : f1-score  = 2 * ((precision * recall) / (precision + recall))

# Vamos fazer uma matriz de confusão -- What?!?!
print(pd.crosstab(classes, resultados, rownames = ["Real"], colnames=["Predito"], margins=True))

# Lembrando que:
#    - Predito = O que o programa classificou como Negativo, Neutro, Positivo e All
#    - Real    = O que é de fato Negativo, Neutro, Positivo e All
#
# Ou seja, somente 9 tweets eram de fato negativos e o programa classificou como positivos. Já os
# positivos que o programa classificou como negativos foram 45, muito mais

# Com o modelo de Bigrams, em lugar de vetorizar o texto "por palavra", vamos vetoriza-lo por cada
# "duas palavras", tipo: Eu gosto de São Paulo => { eu gosto, gosto de, de são, são paulo }
vectorizer = CountVectorizer(ngram_range = (1, 2))
freq_tweets = vectorizer.fit_transform(tweets)

modelo = MultinomialNB()
modelo.fit(freq_tweets, classes)

# Nova predição bigramada
resultados = cross_val_predict(modelo, freq_tweets, classes, cv = 10)
print(resultados)

# Qual foi a acuracidade desse novo modelo?
print(metrics.accuracy_score(classes, resultados))

# As novas medidas de validação do modelo, um pouquinho melhor que o anterior
print(metrics.classification_report(classes, resultados, labels=sentimentos))

# E a nova matriz de confusão
print(pd.crosstab(classes, resultados, rownames = ["Real"], colnames = ["Predito"], margins = True))

# Mudanças em relação ao modelo anterior:
#
#    - Negativo-negativo = 2275 vs 2265 (piorou)
#    - Negativo-neutro   = 162  vs 179  (piorou)
#    - Negativo-positivo = 9    vs 2    (melhorou)
#
#    - Positivo-positivo = 2899 vs 2900 (melhorou)
#    - Positivo-neutro   = 356  vs 357  (piorou)
#    - Positivo-negativo = 45   vs 43   (melhorou)
#
#    - Neutro-neutro     = 2067 vs 2177 (melhorou)
#    - Neutro-negativo   = 240  vs 181  (melhorou)
#    - Neutro-positivo   = 146  vs 95   (melhorou)
#
# Tabela anterior para referência:
#
#    Predito   Negativo  Neutro  Positivo   All
#    Real                                      
#    Negativo      2275     162         9  2446
#    Neutro         240    2067       146  2453
#    Positivo        45     356      2899  3300
#    All           2560    2585      3054  8199

# Vamos reinicializar nosso bag of words com um parâmetro de máximo de features
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                             stop_words = None, max_features = 5000)

# Treinar o modelo, aprender o vocabulário e transformar nossos dados de treinamento em feature vectors
train_data_features = vectorizer.fit_transform(tweets)

# Hora de iniciar um classificador Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)

# Separar os sentimentos do dataset de tweets
class_sentimentos = dataset["Classificacao"].values

# Ajusta a forest ao dataset de treinamento usando a bag of words como feature e os sentimentos
# como a resposta variável
forest = forest.fit(train_data_features, class_sentimentos)


# Criar a bag of words de teste
test_data_features = vectorizer.transform(testes)

# Fazer um predição
resultados = forest.predict(test_data_features)
print(resultados)

# Resultado que tivemos com o primeiro modelo:
# array(['Neutro', 'Neutro', 'Negativo', 'Negativo', 'Neutro', 'Positivo'], dtype='<U8')
#
# Meh.

# Que tal gerar uma tabelinha Pandas?
testes_id = [1, 2, 3, 4, 5, 6]

data_frame = pd.DataFrame(data = { "id": testes_id, "texto": testes, "sentimento": resultados })
print(data_frame)