import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):
    # Сортируем признаки и целевые значения в порядке возрастания признаков
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    # Находим уникальные значения признаков
    unique_values, value_counts = np.unique(sorted_features, return_counts=True)

    # Вычисляем индексы, где происходят изменения в признаках
    change_indices = np.cumsum(value_counts)[:-1]

    # Рассчитываем критерий Джини для каждого порога
    ginis = []
    for index in change_indices:
        left_target = sorted_targets[:index]
        right_target = sorted_targets[index:]

        p1_left = np.sum(left_target == 1) / len(left_target)
        p0_left = 1 - p1_left
        gini_left = 1 - p1_left ** 2 - p0_left ** 2

        p1_right = np.sum(right_target == 1) / len(right_target)
        p0_right = 1 - p1_right
        gini_right = 1 - p1_right ** 2 - p0_right ** 2

        gini = (len(left_target) * gini_left + len(right_target) * gini_right) / len(sorted_targets)
        ginis.append(gini)

    # Находим порог с минимальным значением критерия Джини
    best_threshold_index = np.argmin(ginis)
    threshold_best = (unique_values[change_indices[best_threshold_index]] + unique_values[change_indices[best_threshold_index] - 1]) / 2
    gini_best = ginis[best_threshold_index]

    # Возвращаем пороги, значения критерия Джини, оптимальный порог и соответствующее значение критерия Джини
    return unique_values[change_indices - 1], ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=2):
        # Инициализация параметров дерева
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None  # Структура дерева будет храниться здесь

    def _gini(self, y):
        # Рассчет критерия Джини для узла
        total_samples = len(y)
        class_counts = Counter(y)
        gini = 1.0
        for class_label in class_counts:
            class_probability = class_counts[class_label] / total_samples
            gini -= class_probability ** 2
        return gini

    def _find_best_split(self, X, y, valid_features):
        # Рассчет критерия Джини для всех возможных разбиений и поиск наилучшего разбиения
        best_gini = float('inf')
        best_split = None
        
        for feature_index in valid_features:
            feature_values = np.unique(X[:, feature_index])
            for value in feature_values:
                left_mask = X[:, feature_index] == value
                right_mask = ~left_mask
                if np.sum(left_mask) >= self.min_samples_split and np.sum(right_mask) >= self.min_samples_split:
                    left_gini = self._gini(y[left_mask])
                    right_gini = self._gini(y[right_mask])
                    weighted_gini = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / len(y)
                    if weighted_gini < best_gini:
                        best_gini = weighted_gini
                        best_split = (feature_index, value)
                        
        return best_split




    def _fit_node(self, X, y, depth, valid_features):
        # Рекурсивное построение дерева
        if depth is None or depth == 0 or len(set(y)) == 1:
            # Критерии останова: достигнута максимальная глубина или все объекты в узле одного класса
            return {'type': 'leaf', 'class': Counter(y).most_common(1)[0][0]}
        
        best_split = self._find_best_split(X, y, valid_features)
        if best_split is None:
            # Не удалось найти разбиение, создаем листовой узел
            return {'type': 'leaf', 'class': Counter(y).most_common(1)[0][0]}
        
        feature_index, value = best_split
        left_mask = X[:, feature_index] == value
        right_mask = ~left_mask
        
        left_node = self._fit_node(X[left_mask], y[left_mask], depth, valid_features)
        right_node = self._fit_node(X[right_mask], y[right_mask], depth, valid_features)
        
        return {'type': 'split', 'feature_index': feature_index, 'value': value,
                'left': left_node, 'right': right_node}





    def _predict_node(self, x, node):
        # Рекурсивное предсказание для одного объекта
        if node['type'] == 'leaf':
            return node['class']
        
        feature_index = node['feature_index']
        value = node['value']
        
        if x[feature_index] == value:
            return self._predict_node(x, node['left'])
        else:
            return self._predict_node(x, node['right'])

 


    def fit(self, X, y):
        # Подготовка данных и запуск рекурсивного построения дерева
        valid_features = np.arange(X.shape[1])  # Все признаки допустимы для разбиения на первом уровне
        if self.max_depth is None:
            self.max_depth = float('inf')
        self.tree = self._fit_node(X, y, self.max_depth, valid_features)

    def predict(self, X):
        # Проверка, было ли дерево обучено
        if self.tree is None:
            raise RuntimeError("Дерево не было обучено. Сначала выполните метод fit.")
        
        # Предсказание для всех объектов
        predictions = []
        for x in X:
            predictions.append(self._predict_node(x, self.tree))
        return np.array(predictions)
    