from id3 import id3
from RandomForest import rf
#from NeuralNetwork import nn

filepath = './treino_modificado.txt'

##################################################
k = 10
split = 'a'
#id3(filepath, k, split)

##################################################
k = 10 #5
trees = 20
rf(filepath, k, trees)

##################################################
k=4
epochs=42
batch_size=6
#nn(filepath, k, epochs, batch_size)