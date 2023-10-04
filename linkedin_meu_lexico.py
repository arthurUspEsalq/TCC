import numpy as np
import matplotlib.pyplot as plt
#Importando o comando app da library google-play-scraper
from google_play_scraper import app

#Baixando todos os reviews do app.
from google_play_scraper import Sort, reviews_all

#Comando para todas as reviews
Reviews = reviews_all( 'com.cornershopapp.android', lang = 'pt', country = 'br', sort = Sort.MOST_RELEVANT, sleep_milliseconds = 0)

#Importando pandas.
import pandas as pd

#Transformando os dados em um DataFrame para trabalharmos as análises.
reviews_app = pd.DataFrame(Reviews)
contagem = reviews_app.groupby('score').size()


#Importando o nltk e salvando os corpus necessários
import nltk
nltk.download('wordnet')
nltk.download('punkt')

#Aplicando uma função para tokenizar por palavra
reviews_app['content2'] = reviews_app.apply(lambda row: nltk.word_tokenize(row['content']), axis=1) # Tokenização dos dados

import re
from nltk.corpus import stopwords
import unidecode 
nltk.download('stopwords')
language = 'portuguese'

#editando lista de stop word do NLTK - passando para minusculo, removendo acentos e 'nao'
lista_stop = nltk.corpus.stopwords.words('portuguese')
#minusculo
lista_stop = [x.lower() for x in lista_stop]
#acentos
for i in range(len(lista_stop)):     
    lista_stop[i] = unidecode.unidecode(lista_stop[i]) 
#'nao'
lista_stop.remove('nao')
lista_stop.remove('muito')
#duplicados
lista_stop = list(set(lista_stop))
lista_stop.sort()

def remove_stopwords(words):
    """Remover as Stopwords das palavras tokenizadas"""
    new_words = []
    for word in words:
        if word not in lista_stop:
            new_words.append(word)
    return new_words

def to_lowercase(words):
    """converter todos os caracteres para lowercase"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

words = 'a maçã está ótimo'
def remove_acentos(words):
    # Define os caracteres acentuados e seus equivalentes sem acentos
    acentos = {'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a', 'é': 'e', 'ê': 'e', 'í': 'i', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ú': 'u', 'ü': 'u', 'ç': 'c'}
    # Substitui cada caractere acentuado por seu equivalente sem acento usando a função re.sub()
    new_words = []
    for word in words:
        new_word = re.sub('[{}]'.format(''.join(acentos.keys())), lambda m: acentos[m.group(0)], word)
        new_words.append(new_word)

    return new_words

def normalize(words):
    words = to_lowercase(words)
    words = remove_acentos(words)
    words = remove_stopwords(words) 
    return ' '.join(words)

reviews_app['content2'] = reviews_app.apply(lambda row: normalize(row['content2']), axis=1)


#dicionario
#lexico de todas palavras
lexico = open(r'C:\Users\totvt\OneDrive\Área de Trabalho\pos\tcc\codigos\codigo_pronto\lexico\lexico.txt')
df_lexico = pd.read_csv(r'C:\Users\totvt\OneDrive\Área de Trabalho\pos\tcc\codigos\codigo_pronto\lexico\lexico.txt', delimiter=',', header=None)
df_lexico = df_lexico.rename(columns={0: 'text', 1: 'class', 2: 'value', 3: 'type'})

# Criar um dicionário vazio para armazenar os resultados
dicionario = {}

# Loop sobre cada linha do DataFrame e adicionar um elemento ao dicionário
for i in range(len(df_lexico)):
    chave = df_lexico.iloc[i]['text']
    valor = df_lexico.iloc[i]['value']
    dicionario[chave] = valor



#Criando uma função chamada "Score de Sentimento" para determinar os #sentimentos associados
def Score_sentimento(frase):
    frase = frase.lower()
    l_sentimento = []
    for p in frase.split():
        l_sentimento.append(int(dicionario.get(p, 0)))
    score = sum(l_sentimento)
    if score > 0:
        return 'Pos {} '.format(score)
    elif score == 0:
        return 'Neu {} '.format(score)
    else:        
        return 'Neg {}'.format(score)    

#Criando uma função para aplicar um score de sentimento para cada um dos comentários, a partir das palavras positivas e negativas.
reviews_app['sentimento'] = reviews_app.apply(lambda row: Score_sentimento(row['content2']), axis=1)

#Reorganizando o resultado em colunas para posteriormente lançar no modelo
reviews_app['Score_Sentimento'] = reviews_app['sentimento'].str.slice(-2)
reviews_app['Score_Sentimento'] = reviews_app['Score_Sentimento'].astype(int)
reviews_app['Sent'] = reviews_app['sentimento'].str.slice(0,-3)

#Verificando como ficou a distribuição de comentários a partir do Score de Sentimento Criado.
linha_maior_valor = reviews_app.loc[reviews_app['Score_Sentimento'].idxmax()]
resumo = reviews_app.groupby('score').count()

#comparando a classificao do dicionario com avaliacao dos usuarios
reviews_app['classificacao'] = reviews_app["score"].replace([1,2,3,4,5], [-1,-1,0,1,1])
reviews_app['classificacao_senti'] = reviews_app["classificacao"].replace([-1,0,1], ["Negativo","Neutro","Positivo"])


reviews_app['classificacao2'] = np.sign(reviews_app['Score_Sentimento']).replace({-1: -1, 0: 0, 1: 1})

#grafico com distribuição
contagem2 = reviews_app.groupby('classificacao_senti').size()
contagem2 = contagem2.reset_index()
contagem2.columns = ['Classificacao', 'Valores']

# Criar o gráfico de colunas
plt.bar(contagem2['Classificacao'], contagem2['Valores'])

# Adicionar rótulos aos eixos
plt.xlabel('Classificacao')
plt.ylabel('Valores')

# Adicionar título ao gráfico
plt.title('Avaliações por Classificação')

# Exibir o gráfico
plt.show()
#grafico com distribuição


#criando classificao da nota dos usuario em pos, neg e neu

reviews_app.to_excel(r'C:\Users\totvt\OneDrive\Área de Trabalho\pos\tcc\codigos\lexicos\codigo_corne.xlsx')

reviews_app['comparacao_avaliacao'] = (reviews_app['classificacao'] == reviews_app['classificacao2']).astype(int)

counts = reviews_app['comparacao_avaliacao'].value_counts()
acertos = counts[1]
erros = counts[0]


########## matriz de confusao ##########



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Calculando a matriz de confusão
conf_matrix = confusion_matrix(reviews_app['classificacao_senti'], reviews_app['Sent'], labels=['Neg', 'Neu', 'Pos'])

# Renomeando as colunas
conf_matrix_df = pd.DataFrame(conf_matrix, columns=['pred_neg', 'pred_neu', 'pred_pos'], index=['true_neg', 'true_neu', 'true_pos'])

# Exibindo a matriz de confusão
print(conf_matrix_df)

# Gerando o heatmap da matriz de confusão
sns.heatmap(conf_matrix_df, annot=True, fmt='g', cmap='Blues')

# Exibindo o gráfico
plt.show()

##################### nuvem palavras #####################
#criando um objeto somente com os comentários
content = reviews_app['content2']

#juntando todos eles para construir a wordcloud - ela tem que estar todo contido numa string
all_content = "".join(c for c in content)

#importando as libraries necessárias para o wordcloud
from wordcloud import WordCloud

def nuvem_palavras(reviews):
    wordcloud = WordCloud(width = 3000, height = 2000, background_color = 'white').generate(str(reviews))
    fig = plt.figure(figsize = (0,6))
    plt.lmshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show

df_pos = reviews_app.iloc[reviews_app['score'] == 5]['content2']
nuvem_palavras(df_pos)

valores_unicos = reviews_app["score"].unique()

##################### aplicando modelo na base #####################

########### quantidade de dados por score ###########
import seaborn as sns
sns.countplot('score', data=reviews_app);


########### quantidade de dados por score ###########



from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range = (1,1))

#aplicar modelo em base balanceada na mao
contagem = reviews_app['classificacao'].value_counts()

# Definir o número de linhas que devem ser mantidas para cada valor
num_linhas_por_valor = 529

# Agrupar as linhas da tabela de acordo com o valor da coluna1 e amostrar aleatoriamente um número fixo de linhas de cada grupo
tabela_amostrada = reviews_app.groupby('classificacao', group_keys=False).apply(lambda x: x.sample(num_linhas_por_valor)).reset_index(drop=True)



#mapeamento da palavra, frequencia
vect.fit(reviews_app['content2'])

#print palavra
vect.vocabulary_
text_vect = vect.transform(reviews_app['content2'])

from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(text_vect, reviews_app['classificacao'], test_size=0.3, random_state=42)
print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

#iniciando modelo de regressao
model = LogisticRegression(random_state=0, solver='newton-cg')
#model2 = BernoulliNB()

#treinando modelo
model = model.fit(X_train, Y_train)    
#model2 = model2.fit(X_train, Y_train)

from sklearn.metrics import f1_score

#avaliacao do modelo
y_prediction_train = model.predict(X_train)
f1 = f1_score(y_prediction_train, Y_train, average='weighted')
print(f1)

#avaliacao do teste
y_prediction_test = model.predict(X_test)
f1 = f1_score(y_prediction_test, Y_test, average='weighted')
print(f1)

# plotar a matrix de confusão
cm = confusion_matrix(Y_test, y_prediction_test, labels=model.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
disp.plot()
plt.show()

############## aplicando modelo em base balanceada ##############
from imblearn.under_sampling import RandomUnderSampler


# usar técnica under-sampling
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, Y_train)
# ver o balanceamento das classes
print(pd.Series(y_res).value_counts())
# plotar a nova distribuição de classes
sns.countplot(y_res);

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score



# instanciar e treinar um modelo de Regressão Logística
model_res = LogisticRegression()
model_res.fit(X_res, y_res)
# fazer as previsões em cima dos dados de teste
y_pred_res = model_res.predict(X_test)
y_proba_res = model_res.predict_proba(X_test)
# plotar a matrix de confusão
cm = confusion_matrix(Y_test, y_pred_res, labels=model_res.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_res.classes_)
disp.plot()
plt.show()


# imprimir relatório de classificação
print("Relatório de Classificação:\n", classification_report(Y_test, y_pred_res, digits=4))
# imprimir a acurácia do modelo
print("Acurácia: {:.4f}\n".format(accuracy_score(Y_test, y_pred_res)))
# imprimir a área sob da curva
print("AUC: {:.4f}\n".format(roc_auc_score(Y_test, y_pred_res)))





############## aplicando modelo em base balanceada ##############




##################### aplicando modelo na base #####################



########### testando em outro app
reviews_mimo = reviews_all( 'com.getmimo', lang = 'pt', country = 'br', sleep_milliseconds = 0)

#Transformando os dados em um DataFrame para trabalharmos as análises.
df_mimo = pd.DataFrame(reviews_mimo)

contagem2 = df_mimo.groupby('score').size()

#Aplicando uma função para tokenizar por palavra
df_mimo['content2'] = df_mimo.apply(lambda row: nltk.word_tokenize(row['content']), axis=1) # Tokenização dos dados

#app mimo corrigindo frases
df_mimo['content2'] = df_mimo.apply(lambda row: normalize(row['content2']), axis=1)

#aplicando modelo na base
df_mimo['predicao'] = model.predict(vect.transform(df_mimo['content2']))
#aplicando modelo balanceado
df_mimo['predicao_balanc'] = model_res.predict(vect.transform(df_mimo['content2']))


#criando classificacao pos, neg, neu do score e predicao
df_mimo['class_score'] = df_mimo["score"].replace([1,2,3,4,5], ["Neg","Neg","Neu","Pos","Pos"])
df_mimo['class_pred'] = df_mimo["predicao"].replace([-1,0,1], ["Neg","Neu","Pos"])
df_mimo['class_pred_balanc'] = df_mimo["predicao_balanc"].replace([-1,0,1], ["Neg","Neu","Pos"])


#comparando classes do score e predicao
filtro = (df_mimo['class_pred'] == 'Neg') & (df_mimo['class_score'] == 'Pos')
tabela_filtrada = df_mimo[filtro]

contagem = df_mimo['class_score'].value_counts()
print(contagem)

###### aplicando modelo

# Calculando a matriz de confusão na tabela mimo
conf_matrix = confusion_matrix(df_mimo['class_score'], df_mimo['class_pred'], labels=['Neg', 'Neu', 'Pos'])

# Renomeando as colunas
conf_matrix_df = pd.DataFrame(conf_matrix, columns=['pred_neg', 'pred_neu', 'pred_pos'], index=['true_neg', 'true_neu', 'true_pos'])

# Exibindo a matriz de confusão
print(conf_matrix_df)

# Gerando o heatmap da matriz de confusão
sns.heatmap(conf_matrix_df, annot=True, fmt='g', cmap='Blues')

# Exibindo o gráfico
plt.show()


    