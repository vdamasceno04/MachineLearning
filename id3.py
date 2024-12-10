import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.model_selection import KFold

class Node:
    def __init__(self, attribute=None, threshold=None, left=None, right=None, label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

def entropy(labels):
    label_counts = Counter(labels)
    total_labels = len(labels)
    entropy_value = 0.0

    for label_count in label_counts.values():
        probability = label_count / total_labels
        entropy_value -= probability * math.log2(probability)

    return entropy_value

def information_gain(parent_labels, left_labels, right_labels):
    parent_entropy = entropy(parent_labels)
    left_entropy = entropy(left_labels)
    right_entropy = entropy(right_labels)
    
    left_weight = len(left_labels) / len(parent_labels)
    right_weight = len(right_labels) / len(parent_labels)
    
    return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

def best_split(data, labels):
    best_gain = 0
    best_attribute = None
    best_split_point = None
    n_features = data.shape[1]

    for feature in range(n_features):
        unique_values = np.unique(data[:, feature])
        for split_point in unique_values:
            left_indices = data[:, feature] <= split_point
            right_indices = data[:, feature] > split_point

            left_labels = labels[left_indices]
            right_labels = labels[right_indices]
            
            gain = information_gain(labels, left_labels, right_labels)
            
            if gain > best_gain:
                best_gain = gain
                best_attribute = feature
                best_split_point = split_point

    return best_attribute, best_split_point

def median_split(data, labels):
    best_gain = 0
    best_attribute = None
    best_split_point = None
    n_features = data.shape[1]

    for feature in range(n_features):
        # Calcula a mediana do atributo
        split_point = np.median(data[:, feature])

        # Divide os dados com base na mediana
        left_indices = data[:, feature] <= split_point
        right_indices = data[:, feature] > split_point

        left_labels = labels[left_indices]
        right_labels = labels[right_indices]

        # Calcula o ganho de informação para essa divisão
        gain = information_gain(labels, left_labels, right_labels)

        if gain > best_gain:
            best_gain = gain
            best_attribute = feature
            best_split_point = split_point

    return best_attribute, best_split_point

def insert_tree(data, label, split):
    if len(set(label)) == 1: #se só há 1 rótulo, ele é raiz
        return Node(label=label[0])
    
    if len(data) == 0: 
        return Node(label=Counter(label).most_common(1)[0][0]) #se não, escolhe o mais comum

    if split == 'm':
        best_attribute, best_split_point = median_split(data, label)
    elif split == 'a':
        best_attribute, best_split_point = best_split(data, label)

    if best_attribute is None:
        return Node(label=Counter(label).most_common(1)[0][0])

    left_indices = data[:, best_attribute] <= best_split_point
    right_indices = data[:, best_attribute] > best_split_point
    
    left_subtree = insert_tree(data[left_indices], label[left_indices], split)
    right_subtree = insert_tree(data[right_indices], label[right_indices], split)
    
    return Node(attribute=best_attribute, threshold=best_split_point, left=left_subtree, right=right_subtree)

def predict(node, sample):
    if node.label is not None:
        return node.label
    if sample[node.attribute] <= node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)

def id3(filepath, k, split):
    data = pd.read_csv(filepath)
    features = data[['qualidade_pressao_arterial', 'pulso', 'respiracao']].values
    label = data['rotulo'].values

    kf = KFold(n_splits=k)
    accuracies = []

    for train_i, test_i in kf.split(features):
        x_train, x_test = features[train_i], features[test_i]
        y_train, y_test = label[train_i], label[test_i]
        
        tree = insert_tree(x_train, y_train, split)

        correct_predictions = 0
        for i in range(len(x_test)):
            prediction = predict(tree, x_test[i])
            if prediction == y_test[i]:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(y_test)
        
        print(f"Acurácia da partição: {accuracy * 100:.2f}%\n")
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    print(f"Acurácia média: {average_accuracy * 100:.2f}%")
