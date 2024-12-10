from id3 import id3
from RandomForest import rf
import matplotlib.pyplot as plt
import pandas as pd
#from NeuralNetwork import nn

def plot(k_values, accuracy_results):
    data = []
    for k in k_values:
        for i, acc in enumerate(accuracy_results[k], start=1):
            data.append({'k': k, 'Iteração': i, 'Acurácia': acc})

    combined_table = pd.DataFrame(data)

    # Reorganizar os dados para criar uma tabela com k como índice e iteração como colunas
    pivot_table = combined_table.pivot(index='k', columns='Iteração', values='Acurácia')

    pivot_table['Média'] = pivot_table.mean(axis=1)

    # Formatar os rótulos das linhas como "k = {valor}"
    row_labels = [f"k = {k}" for k in pivot_table.index]
    col_labels = [f"Partição {col}" if col != 'Média' else col for col in pivot_table.columns]
    formatted_values = pivot_table.applymap(
        lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "-").values


    # Gerar a tabela com matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')

    # Criar a tabela com os rótulos formatados
    table = ax.table(cellText=formatted_values, 
            colLabels=col_labels, 
            rowLabels=row_labels, loc='center', 
            colColours=["#f0f0f0"] * len(pivot_table.columns), 
            rowColours=["#f0f0f0"] * len(row_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    plt.title("Tabela de Acurácia: k x Iteração")
    plt.show()


filepath = './treino_modificado.txt'

k_values = [3, 10]

"""
accuracy_results = {
    k_values[0]: id3(filepath, k_values[0], 'm'),
    k_values[1]: id3(filepath, k_values[1], 'm'),
    k_values[2]: id3(filepath, k_values[2], 'm')
}

plot(k_values, accuracy_results) #ID3, K=3,7,10,MEDIAN SPLIT

accuracy_results = {
    k_values[0]: id3(filepath, k_values[0], 'a'),
    k_values[1]: id3(filepath, k_values[1], 'a'),
    k_values[2]: id3(filepath, k_values[2], 'a')
}
#plot(k_values, accuracy_results) #ID3, K=3,7,10,BEST SPLIT
"""
accuracy_results = {
    k_values[0]: rf(filepath, k_values[0],40 ,6),
    k_values[1]: rf(filepath, k_values[1],40 ,6),
    k_values[0]: rf(filepath, k_values[0],120 ,6),
    k_values[1]: rf(filepath, k_values[1],120 ,6),
    k_values[0]: rf(filepath, k_values[0],40 ,18),
    k_values[1]: rf(filepath, k_values[1],40 ,18),
    k_values[0]: rf(filepath, k_values[0],120 ,18),
    k_values[1]: rf(filepath, k_values[1],120 ,18),
    
}
plot(k_values, accuracy_results) #RF, K=3,7,10, 20 arvores

#k = 10 #5
#trees = 10
#rf(filepath, k, trees)

##################################################
#k=4
#epochs=42
#batch_size=6
#nn(filepath, k, epochs, batch_size)