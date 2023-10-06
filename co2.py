import pandas as pd
from datetime import datetime, timedelta

# Função para converter a data e hora da segunda base para o formato aaaa-mm-dd hh:mm:ss
def converter_data_hora(data, hora_inicio):
    data_formatada = datetime.strptime(data, '%d/%m/%Y')
    hora_formatada = datetime.strptime(hora_inicio, '%H:%M')
    data_hora_formatada = data_formatada.replace(hour=hora_formatada.hour, minute=hora_formatada.minute, second=0)
    return data_hora_formatada

# Função para verificar se dois eventos ocorrem no mesmo horário
def eventos_ocorrem_no_mesmo_horario(data_hora_evento1, duracao_evento1, data_hora_evento2, duracao_evento2):
    data_hora_evento1_fim = data_hora_evento1 + timedelta(seconds=duracao_evento1)
    data_hora_evento2_fim = data_hora_evento2 + timedelta(minutes=duracao_evento2)
    
    return data_hora_evento1 <= data_hora_evento2_fim and data_hora_evento1_fim >= data_hora_evento2
    return data_hora_evento1_fim <= data_hora_evento2_fim and data_hora_evento1 >= data_hora_evento2

# Exemplo de dados da primeira base
dados_base1 = {
    'Data_Hora': ["2023-10-05 14:00:00", "2023-10-05 15:30:00", "2023-10-05 16:45:00", "2023-10-06 10:00:00"],
    'Duracao_Segundos': [3600, 2700, 1800, 3600]
}

# Exemplo de dados da segunda base
dados_base2 = {
    'Data': ["05/10/2023", "05/10/2023", "05/10/2023", "06/10/2023", "06/10/2023"],
    'Hora_Inicio': ["14:00", "15:15", "16:30", "09:50", "10:01"],
    'Duracao_Minutos': [60, 45, 60, 9, 1]
}

# Crie DataFrames para as duas bases
df_base1 = pd.DataFrame(dados_base1)
df_base2 = pd.DataFrame(dados_base2)

# Adicione uma coluna no DataFrame da primeira base para marcar se o evento ocorre durante um evento da segunda base
df_base1['Ocorre_Durante_Evento_Base2'] = False

# Verifique se os eventos da primeira base ocorrem durante os eventos da segunda base
for index_base1, row_base1 in df_base1.iterrows():
    data_hora_evento1 = datetime.strptime(row_base1['Data_Hora'], '%Y-%m-%d %H:%M:%S')
    duracao_evento1 = row_base1['Duracao_Segundos']
    
    for index_base2, row_base2 in df_base2.iterrows():
        data_hora_evento2 = converter_data_hora(row_base2['Data'], row_base2['Hora_Inicio'])
        duracao_evento2 = row_base2['Duracao_Minutos']
        
        if eventos_ocorrem_no_mesmo_horario(data_hora_evento1, duracao_evento1, data_hora_evento2, duracao_evento2):
            df_base1.at[index_base1, 'Ocorre_Durante_Evento_Base2'] = True
            
            

print(df_base1)
