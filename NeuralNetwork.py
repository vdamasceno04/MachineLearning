import numpy as np #Biblioteca para ajudar em cálculos
import pandas as pd #Biblioteca pandas para mexer com DF(tabelas)

from sklearn.model_selection import KFold #Divir conjunto de dados para Validação Cruzada
from sklearn.preprocessing import StandardScaler#Normalizar dados

#Modelos de redes Neurais
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#data -> Matris com dados de Entrada
#labels -> Vetor com as classificações possiveis
#k -> n Numero de subgrupos que será dividido o grupo de dados inicial
#epochs -> Numero de vezes que passara pelos dados de treinamento
#batch_size -> Intervalo de  iterações para atualizar os pessos da Rede

def cross_validation(data, labels, k, epochs, batch_size):
    kf = KFold(n_splits=k, shuffle=True, random_state=69)#numero de subgrupos, emparalha os dados, seed de aleatoridade
    accuracies = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model = create_model(X_train.shape[1])#Cria modelos passando quantidade de categorias existentes
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)#Treina modelo com os parametros  

        _, accuracy = model.evaluate(X_test, y_test, verbose=0) #Testa o modelo e devolve acuracia

        accuracies.append(accuracy)
        print(f'Acurácia da partição: {accuracy * 100:.2f}%')

    return np.mean(accuracies)#Devolve a media das acuracias

def create_model(input):
    model = Sequential()#Cria um modelo sequencial

    model.add(Dense(16, input_dim=input, activation='relu'))#Coloca uma camada de 16 neuronios depois da entrada, 
    #relu -> negativo vira zero e positivo não acntece nada   
    model.add(Dense(24, activation='relu'))#Coloca uma camada de 24 neuronios depois da  de 16
    model.add(Dense(labels.shape[1], activation='softmax'))#Coloca uma camada de saida onde o numero de neuronios é igual ao numero de categorias
    #softmax -> transforma sida em probabilidade   
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #Adam -> Algoritmo esponsável por ajustar pesos e vies
    #categorical_crossentropy -> tipo de função que medeo erro entre a previsão e o valor esperado
    #metodo de avalização de desempenho 

    return model

def nn(filepath, k, epochs, batch_size):
    data = pd.read_csv(filepath, index_col=0)#Carrega dados em um DF

    features = data[['qualidade_pressao_arterial', 'pulso', 'respiracao']].values
    labels = data['rotulo'].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)#Normaliza dados

    labels = to_categorical(labels) # Converte as categorias para formato one hot

    average_accuracy = cross_validation(features, labels, k, epochs, batch_size)
    print(f'Acurácia média do Neural Network: {average_accuracy * 100:.2f}%')