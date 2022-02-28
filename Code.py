import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

def first_graph(n):
    xlist = []
    ylist = []

    for x in np.arange(0, 1, n):
        y = (math.sin(6*x)/6) + 0.6
        xlist.append(x)
        ylist.append(y)

    plt.plot(xlist, ylist)
    plt.show()


def first_dataset(n, rounding_factor):
    list_of_elements = [['', 'X', 'Y', 'Class']]

    for x in np.arange(0, 1, n):
        for y in np.arange(0, 1, n):
            if y - ((math.sin(6 * x) / 6) + 0.6) < 0:
                klasa = "negative"
                attributes = ['', round(x, rounding_factor), round(y, rounding_factor), klasa]
                list_of_elements.append(attributes)
            else:
                klasa = "positive"
                attributes = ['', round(x, rounding_factor), round(y, rounding_factor), klasa]
                list_of_elements.append(attributes)

    data = np.array(list_of_elements)
    return pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])


def second_dataset(n, rounding_factor):
    list_of_elements = [['', 'X', 'Y', 'Class']]
    for x in np.arange(0, 1, n):
        for y in np.arange(0, 1, n):
            if (x < 0.25 and y < 0.25) or (x < 0.25 and y < 0.75 and y > 0.5) or (
                    x < 0.5 and x > 0.25 and y > 0.25 and y < 0.5) or (x < 0.5 and x > 0.25 and y > 0.75) or (
                    x < 0.75 and x > 0.5 and y < 0.25) or (x < 0.75 and x > 0.5 and y < 0.75 and y > 0.5) or (
                    x > 0.75 and y < 0.5 and y > 0.25) or (x > 0.75 and y > 0.75):
                klasa = "black"
                attributes = ['', round(x, rounding_factor), round(y, rounding_factor), klasa]
                list_of_elements.append(attributes)
            else:
                klasa = "white"
                attributes = ['', round(x, rounding_factor), round(y, rounding_factor), klasa]
                list_of_elements.append(attributes)
    data = np.array(list_of_elements)
    return pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])


def knn_classifier(n, training_features, training_klasa, test_features, test_klasa):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(training_features, training_klasa)

    pred_klasa = knn.predict(test_features)
    print('KNN Accuracy: ', metrics.accuracy_score(test_klasa, pred_klasa))


def decision_tree_classifier(criterion, splitter, max_depth, training_features, training_klasa, test_features, test_klasa):
    clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
    clf = clf.fit(training_features, training_klasa)

    pred_klasa = clf.predict(test_features)
    print("DT Accuracy: ", metrics.accuracy_score(test_klasa, pred_klasa))


def synthetic_dataset_classifier(dataset, n, criterion, splitter, max_depth):
    dataset = shuffle(dataset)
    training_data = dataset[:int(len(dataset) * 0.8)]
    test_data = dataset[int(len(dataset) * 0.8):]

    training_features = training_data[['X', 'Y']]
    training_klasa = training_data['Class']

    test_features = test_data[['X', 'Y']]
    test_klasa = test_data['Class']

    knn_classifier(n, training_features, training_klasa, test_features, test_klasa)
    decision_tree_classifier(criterion, splitter, max_depth, training_features, training_klasa, test_features, test_klasa)


def iris_cross_validation(dataset, n, criterion, splitter, max_depth):
    features = dataset.iloc[:, :-1]
    klasa = dataset.iloc[:, -1:]

    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    knn = KNeighborsClassifier(n_neighbors=n)
    dt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
    scores_knn = cross_val_score(knn, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
    scores_dt = cross_val_score(dt, features, klasa, scoring='accuracy', cv=cv, n_jobs=-1)
    print('KNN Accuracy: ', (np.mean(scores_knn)))
    print('DT Accuracy: ', (np.mean(scores_dt)))


def letter_recognition_classifier(letter_recognition, n, criterion, splitter, max_depth):
    training_data = letter_recognition[:int(len(letter_recognition) * 0.66)]
    test_data = letter_recognition[int(len(letter_recognition) * 0.66):]

    training_features = training_data.iloc[:, 1:]
    training_klasa = training_data[0]

    test_features = test_data.iloc[:, 1:]
    test_klasa = test_data[0]

    knn_classifier(n, training_features, training_klasa, test_features, test_klasa)
    decision_tree_classifier(criterion, splitter, max_depth, training_features, training_klasa, test_features, test_klasa)

sin_function1       = first_dataset(0.1, 2)
sin_function2       = first_dataset(0.01, 4)
sin_function3       = first_dataset(0.001, 6)
chess_board1        = second_dataset(0.1, 2)
chess_board2        = second_dataset(0.01, 4)
chess_board3        = second_dataset(0.001,6)
iris                = pd.read_csv('iris.csv', header=None)
letter_recognition  = pd.read_csv('letter-recognition.csv', header=None)


print(sin_function1)
print("============================================")
print(sin_function2)
print("============================================")
print(sin_function3)
print("============================================")

print(chess_board1)
print("============================================")
print(chess_board2)
print("============================================")
print(chess_board3)
print("============================================")

print(iris)
print(letter_recognition)

print("Sine Function with 100 samples:")
print("KNN classifier with 1 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=best, max_depth=None")
synthetic_dataset_classifier(sin_function1, 1, "gini", "best", None)
print("KNN classifier with 3 nearest neightbor and Decision Tree classifier with criterion=entropy, splitter=best, max_depth=3")
synthetic_dataset_classifier(sin_function1, 3, "entropy", "best", 3)
print("KNN classifier with 7 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=random, max_depth=4")
synthetic_dataset_classifier(sin_function1, 7, "gini", "random", 3)
print("========================================================================================================================")

print("Sine Function with 10000 samples:")
print("KNN classifier with 1 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=best, max_depth=None")
synthetic_dataset_classifier(sin_function2, 1, "gini", "best", None)
print("KNN classifier with 3 nearest neightbor and Decision Tree classifier with criterion=entropy, splitter=best, max_depth=3")
synthetic_dataset_classifier(sin_function2, 3, "entropy", "best", 3)
print("KNN classifier with 7 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=random, max_depth=4")
synthetic_dataset_classifier(sin_function2, 7, "gini", "random", 3)
print("========================================================================================================================")

print("Sine Function with 1000000 samples:")
print("KNN classifier with 1 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=best, max_depth=None")
synthetic_dataset_classifier(sin_function3, 1, "gini", "best", None)
print("KNN classifier with 3 nearest neightbor and Decision Tree classifier with criterion=entropy, splitter=best, max_depth=3")
synthetic_dataset_classifier(sin_function3, 3, "entropy", "best", 3)
print("KNN classifier with 7 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=random, max_depth=4")
synthetic_dataset_classifier(sin_function3, 7, "gini", "random", 3)
print("========================================================================================================================")

print("Chessboard with 100 samples:")
print("KNN classifier with 1 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=best, max_depth=None")
synthetic_dataset_classifier(chess_board1, 1, "gini", "best", None)
print("KNN classifier with 3 nearest neightbor and Decision Tree classifier with criterion=entropy, splitter=best, max_depth=3")
synthetic_dataset_classifier(chess_board1, 3, "entropy", "best", 3)
print("KNN classifier with 7 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=random, max_depth=4")
synthetic_dataset_classifier(chess_board1, 7, "gini", "random", 3)
print("========================================================================================================================")

print("Chessboard with 10000 samples:")
print("KNN classifier with 1 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=best, max_depth=None")
synthetic_dataset_classifier(chess_board2, 1, "gini", "best", None)
print("KNN classifier with 3 nearest neightbor and Decision Tree classifier with criterion=entropy, splitter=best, max_depth=3")
synthetic_dataset_classifier(chess_board2, 3, "entropy", "best", 3)
print("KNN classifier with 7 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=random, max_depth=4")
synthetic_dataset_classifier(chess_board2, 7, "gini", "random", 3)
print("========================================================================================================================")

print("Chessboard with 1000000 samples:")
print("KNN classifier with 1 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=best, max_depth=None")
synthetic_dataset_classifier(chess_board3, 1, "gini", "best", None)
print("KNN classifier with 3 nearest neightbor and Decision Tree classifier with criterion=entropy, splitter=best, max_depth=3")
synthetic_dataset_classifier(chess_board3, 3, "entropy", "best", 3)
print("KNN classifier with 7 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=random, max_depth=4")
synthetic_dataset_classifier(chess_board3, 7, "gini", "random", 3)
print("========================================================================================================================")

print("Iris Dataset 10 Fold Cross-Validation:")
print("KNN classifier with 1 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=best, max_depth=None")
iris_cross_validation(iris, 1, "gini", "best", None)
print("KNN classifier with 3 nearest neightbor and Decision Tree classifier with criterion=entropy, splitter=best, max_depth=3")
iris_cross_validation(iris, 3, "entropy", "best", 3)
print("KNN classifier with 7 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=random, max_depth=4")
iris_cross_validation(iris, 7, "gini", "random", 3)
print("========================================================================================================================")

print("Letter Recognition Classification:")
print("KNN classifier with 1 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=best, max_depth=None")
letter_recognition_classifier(letter_recognition, 1, "gini", "best", None)
print("KNN classifier with 3 nearest neightbor and Decision Tree classifier with criterion=entropy, splitter=best, max_depth=3")
letter_recognition_classifier(letter_recognition, 3, "entropy", "best", 3)
print("KNN classifier with 7 nearest neightbor and Decision Tree classifier with criterion=gini, splitter=random, max_depth=4")
letter_recognition_classifier(letter_recognition, 7, "gini", "random", 3)

