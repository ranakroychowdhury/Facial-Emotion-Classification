# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 19:42:43 2020

@author: Ranak Roy Chowdhury
"""
import numpy as np
import math
import random


def sigmoid(data, weight):
    n, d = data.shape
    activation = np.matmul(data, weight.T)
    activation = np.exp(activation)
    summation = activation.sum(axis = 1)
    for i in range(n):
        activation[i] /= summation[i]
    return activation
    

def cross_entropy(target, sigmoid_result):
    error_matrix = np.multiply(target, np.log(sigmoid_result))
    error = sum(sum(error_matrix))
    return -error


def evaluation(data, weight, target_label, classes):
    inst, dim = data.shape
    sigmoid_result = sigmoid(data, weight)
    idx = np.argmax(sigmoid_result, axis=1)
    predicted_label = np.zeros((inst, classes))
    for i in range(inst):
        predicted_label[i, idx[i]] = 1
    
    correct = 0
    for i in range(inst):
        if np.array_equal(target_label[i], predicted_label[i]):
            correct += 1
    acc = correct / inst
    error = cross_entropy(target_label, sigmoid_result)
    return acc, error


def softmax_reg_batch(train_data, val_data, test_data, train_label, val_label, test_label, epoch, classes, learning_rate):
    n, d = train_data.shape
    weight = np.zeros((classes, d))
    min_error = math.inf
    best_weight = np.zeros((classes, d))
    training_error = []
    validation_error = []
    validation_acc = []
    test_error = []
    test_acc = []
    for t in range(1, epoch + 1):
        sigmoid_result = sigmoid(train_data, weight)
        weight += learning_rate * np.matmul((train_label - sigmoid_result).T, train_data)
        train_error = cross_entropy(train_label, sigmoid_result)
        val_acc, val_error = evaluation(val_data, weight, val_label, classes)
        t_acc, t_error = evaluation(test_data, weight, test_label, classes)
        
        # print(str(train_error) + ' ' + str(val_error) + ' ' + str(val_acc))
        training_error.append(train_error)
        validation_error.append(val_error)
        validation_acc.append(val_acc)
        test_error.append(t_error)
        test_acc.append(t_acc)
        if val_error < min_error:
            best_t = t
            min_error = val_error
            best_weight = np.copy(weight)
    print(best_t)
    TEST_ACCURACY, TEST_ERROR = evaluation(test_data, best_weight, test_label, classes)
    return training_error, validation_error, validation_acc, test_error, test_acc, TEST_ERROR, TEST_ACCURACY


def split(data, target, train_split, val_split, test_split):
    n, d = data.shape
    train_data = data[ : int(train_split * n)]
    train_label = target[ : int(train_split * n)]
    val_data = data[int(train_split * n) : int((train_split + val_split) * n)]
    val_label = target[int(train_split * n) : int((train_split + val_split) * n)]
    test_data = data[int((train_split + val_split) * n) : ]
    test_label = target[int((train_split + val_split) * n) : ]
    return train_data, val_data, test_data, train_label, val_label, test_label


if __name__ == '__main__':
    n = 100
    d = 3
    epoch = 10
    classes = 6
    learning_rate = 0.1
    
    # generate data
    data = np.random.random((n, d))
    
    # generate labels for six class
    target = np.zeros((n, classes))
    for i in range(n):
        index = random.randrange(0, classes, 1)
        target[i, index] = 1
    
    # split the data into train, validation and test set
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    train_data, val_data, test_data, train_label, val_label, test_label = split(data, target, train_split, val_split, test_split)

    # pass a numpy array of train, val and test data along with their target labels
    # epoch, classes and learning rate
    # This function will be in a loop for cross validation
    # will run 10 times for 10-fold cross validation
    # this function is line 8 - 12 of Training_Procedure function
    training_error, validation_error, validation_acc, test_error, test_acc, TEST_ERROR, TEST_ACCURACY = softmax_reg_batch(train_data, val_data, test_data, train_label, val_label, test_label, epoch, classes, learning_rate)
    print('Training Error: ' + str(training_error))
    print('Validation Error: ' + str(validation_error))
    print('Validation Accuracy: ' + str(validation_acc))
    print('Test Error: ' + str(test_error))
    print('Test Accuracy: ' + str(test_acc))
    print('Actual best test error: ' + str(TEST_ERROR))
    print('Actual best test accuracy: ' + str(TEST_ACCURACY))    
    