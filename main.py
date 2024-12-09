import pandas as pd
import numpy as np
import math

from collections import Counter

from sklearn.model_selection import KFold

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

class DecisionNode:
    def __init__(self, attribute=None, threshold=None, left=None, right=None, label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

def build_tree(data, labels):
    if len(set(labels)) == 1: 
        return DecisionNode(label=labels[0])
    
    if len(data) == 0: 
        return DecisionNode(label=Counter(labels).most_common(1)[0][0])

    best_attribute, best_split_point = best_split(data, labels)
    
    if best_attribute is None:
        return DecisionNode(label=Counter(labels).most_common(1)[0][0])

    left_indices = data[:, best_attribute] <= best_split_point
    right_indices = data[:, best_attribute] > best_split_point
    
    left_subtree = build_tree(data[left_indices], labels[left_indices])
    right_subtree = build_tree(data[right_indices], labels[right_indices])
    
    return DecisionNode(attribute=best_attribute, threshold=best_split_point, left=left_subtree, right=right_subtree)

def predict(node, sample):
    if node.label is not None:
        return node.label
    if sample[node.attribute] <= node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)
import os
print('pasta =', os.getcwd())
data = pd.read_csv('/media/victor/SHARED/UTFPR/SI/T2/conke/Machine-Learning/GravityScale/ID3/dados_treinamento.csv')

features = data[['qualidade_pressao_arterial', 'pulso', 'respiracao']].values
labels = data['rotulo'].values

kf = KFold(n_splits=7)
accuracy_list = []

for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    tree = build_tree(X_train, y_train)

    correct_predictions = 0
    for i in range(len(X_test)):
        prediction = predict(tree, X_test[i])
        if prediction == y_test[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(y_test)
    
    print(f"Acurácia da partição: {accuracy * 100:.2f}%\n")
    accuracy_list.append(accuracy)

average_accuracy = np.mean(accuracy_list)
print(f"Acurácia média: {average_accuracy * 100:.2f}%")