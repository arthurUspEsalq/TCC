import pandas as pd
from datetime import datetime, timedelta

# Função para converter a data e hora da segunda base para o formato aaaa-mm-dd hh:mm:ss
def converter_data_hora(data, hora_inicio):
    data_formatada = datetime.strptime(data, '%d/%m/%Y')
    hora_formatada = datetime.strptime(hora_inicio, '%H:%M')
    data_hora_formatada = data_formatada.replace(hour=hora_formatada.hour, minute=hora_formatada.minute, second=0)
    return data_hora_formatada

# Função para verificar se dois eventos ocorrem no mesmo horário
def eventos_ocorrem_no_mesmo_horario(inicio_evento1, fim_evento1, inicio_evento2, fim_evento2,):   
    inicio_depois = (inicio_evento1 > inicio_evento2) and (fim_evento1 > fim_evento2) and (inicio_evento1 < fim_evento2)
    inicio_antes = (inicio_evento1 < inicio_evento2) and (fim_evento1 < fim_evento2) and (fim_evento1 > inicio_evento2)
    durante = (inicio_evento1 > inicio_evento2) and (fim_evento1 < fim_evento2)

    return inicio_depois, inicio_antes, durante

#row_base1['Data_Hora_Inicio'], row_base1['Data_Hora_Fim'], row_base2['Hora_Inicio'], row_base2['Hora_Fim']

#row_base1['Data_Hora_Inicio'] < row_base2['Hora_Inicio']

# Exemplo de dados da primeira base
dados_base1 = {
    'Data_Hora_Fim': ["2023-10-05 14:00:00", "2023-10-06 15:30:00", "2023-10-07 16:45:00", "2023-10-08 10:00:00"],
    'Duracao_Segundos': [3600, 3600, 3600, 3600]
}

# Exemplo de dados da segunda base
dados_base2 = {
    'Data': ["05/10/2023", "06/10/2023", "07/10/2023", "08/10/2023", "01/10/2023"],
    'Hora_Inicio': ["12:50", "14:15", "16:30", "10:01", "05:59"],
    'Hora_Fim': ["14:20", "14:45", "18:00", "10:02", "22:01"]
    }

# Crie DataFrames para as duas bases
df_base1 = pd.DataFrame(dados_base1)
df_base2 = pd.DataFrame(dados_base2)

# Adicione duas colunas no DataFrame da primeira base para marcar as condições
df_base1['inicio_depois'] = False
df_base1['inicio_antes'] = False
df_base1['durante'] = False

# Adicione uma coluna 'Data_Hora_Inicio' ao DataFrame da primeira base
df_base1['Data_Hora_Inicio'] = ""

# Verifique as condições para cada par de eventos
for index_base1, row_base1 in df_base1.iterrows():
    data_hora_evento1 = datetime.strptime(row_base1['Data_Hora_Fim'], '%Y-%m-%d %H:%M:%S')
    duracao_evento1 = row_base1['Duracao_Segundos']

    data_hora_inicio_evento1 = data_hora_evento1 - timedelta(seconds=duracao_evento1)
    df_base1.at[index_base1, 'Data_Hora_Inicio'] = data_hora_inicio_evento1.strftime('%Y-%m-%d %H:%M:%S')

# Adicione uma coluna 'Data_Hora_Inicio' ao DataFrame da primeira base
df_base2['Data_Hora_Inicio'] = ""
df_base2['Data_Hora_Fim'] = ""

for index_base2, row_base2 in df_base2.iterrows():
    data_hora_evento2 = converter_data_hora(row_base2['Data'], row_base2['Hora_Inicio'])
    data_hora_fim_evento2 = converter_data_hora(row_base2['Data'], row_base2['Hora_Fim'])

    df_base2.at[index_base2, 'Data_Hora_Inicio'] = data_hora_evento2.strftime('%Y-%m-%d %H:%M:%S')
    df_base2.at[index_base2, 'Data_Hora_Fim'] = data_hora_fim_evento2.strftime('%Y-%m-%d %H:%M:%S')
    

# Verifique as condições para cada par de eventos
for index_base1, row_base1 in df_base1.iterrows():
    
    for index_base2, row_base2 in df_base2.iterrows():
        
        inicio_depois, inicio_antes, durante = eventos_ocorrem_no_mesmo_horario(row_base1['Data_Hora_Inicio'], row_base1['Data_Hora_Fim'], row_base2['Data_Hora_Inicio'], row_base2['Data_Hora_Fim'])
        
        if inicio_depois:
            df_base1.at[index_base1, 'inicio_depois'] = True
        if inicio_antes:
            df_base1.at[index_base1, 'inicio_antes'] = True
        if durante:
            df_base1.at[index_base1, 'durante'] = True
        
print(df_base1)
