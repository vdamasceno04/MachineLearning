import math
from collections import Counter
import numpy as np

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