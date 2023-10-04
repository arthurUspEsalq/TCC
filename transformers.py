import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from pprint import pprint

#Folder path containing the fine-tuned model files
model_path = r'C:\Users\totvt\OneDrive\Área de Trabalho\pos\tcc\codigos\codigo_pronto\codigos_final\bert'

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)


classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)

threshold = 0.3

inputs = [
	'😄 Estou muito feliz hoje!',
    'Tô num mar de pensamentos positivos! 🌊✨'
]

output = classifier(inputs)

predictions = []

for prediction in output:
    max_score_pred = max(prediction, key=lambda x: x['score'])
    predictions.append(max_score_pred)

# Criar uma lista para cada coluna do DataFrame
phrases = []
labels = []
scores = []

for idx, pred in enumerate(predictions):
    phrases.append(inputs[idx])
    labels.append(pred['label'])
    scores.append(pred['score'])

# Criar o DataFrame usando as listas
data = {
    'Frase': phrases,
    'Label': labels,
    'Score': scores
}

df = pd.DataFrame(data)
df


####################################### modelo alice #######################################
df_alice = pd.read_excel(r'C:\Users\totvt\OneDrive\Área de Trabalho\pos\tcc\bases_twitter\base_alice_2.xlsx')

frase_list = df_alice['tweet'].tolist()

output = classifier(frase_list)

predictions = []

for prediction in output:
    max_score_pred = max(prediction, key=lambda x: x['score'])
    predictions.append(max_score_pred)

# Criar uma lista para cada coluna do DataFrame
phrases = []
labels = []
scores = []

for idx, pred in enumerate(predictions):
    phrases.append(frase_list[idx])
    labels.append(pred['label'])
    scores.append(pred['score'])


# Criar o DataFrame usando as listas
data = {
    'tweet': phrases,
    'Label': labels,
    'Score': scores
}

df_pred_alice = pd.DataFrame(data)
df_pred_alice = pd.merge(df_pred_alice,df_alice[['tweet','data']],on='tweet', how='left')


####################################### modelo lewis #######################################
df_lewis = pd.read_excel(r'C:\Users\totvt\OneDrive\Área de Trabalho\pos\tcc\bases_twitter\hamilton_2.xlsx')
df_lewis.drop_duplicates()

frase_list = df_lewis['tweet'].tolist()

output = classifier(frase_list)

predictions = []

for prediction in output:
    max_score_pred = max(prediction, key=lambda x: x['score'])
    predictions.append(max_score_pred)

# Criar uma lista para cada coluna do DataFrame
phrases = []
labels = []
scores = []

for idx, pred in enumerate(predictions):
    phrases.append(frase_list[idx])
    labels.append(pred['label'])
    scores.append(pred['score'])


# Criar o DataFrame usando as listas
data = {
    'tweet': phrases,
    'Label': labels,
    'Score': scores
}

df_pred_lewis = pd.DataFrame(data)
df_pred_lewis = pd.merge(df_pred_lewis,df_lewis[['tweet','data']],on='tweet', how='left')

df_pred_lewis.drop_duplicates()
