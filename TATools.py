from pandas import read_csv
from pennylane import numpy as np
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def features_to_broadcast(features, pattern, windings):
    params = []

    if pattern=='all_to_all':
        for i in range(len(features)):
            for j in range(i+1,len(features)):
                params.append(windings[1]*np.pi*(1-features[i])*(1-features[j]))

    if pattern=='ring':
        for i in range(len(features)):
            params.append(windings[1]*np.pi*(1-features[i])*(1-features[(i+1)%len(features)]))

    return params

def ansatz_V(params, wires, reps, last_layer=False):
    for layer in range(reps):
        for wire in wires:
            qml.RY(params[layer*len(wires)*2+wire*2], wire)
            qml.RZ(params[layer*len(wires)*2+wire*2+1], wire)
        qml.broadcast(qml.CZ, wires, pattern='all_to_all')
    if last_layer:
        for wire in wires:    
            qml.RY(params[reps*len(wires)*2+wire*2], wire)
            qml.RZ(params[reps*len(wires)*2+wire*2+1], wire)
        

def ansatz_U(features, windngs, wires, reps):
    for layer in range(reps):
        qml.broadcast(qml.Hadamard, wires, pattern='single')
        qml.broadcast(qml.RZ, wires, pattern='single', parameters=windngs[0]*np.pi*features)####winding 1
        if len(features)>2:
            qml.broadcast(qml.MultiRZ, wires, pattern='all_to_all', parameters=features_to_broadcast(features, 'all_to_all', windngs))####winding 2
        else:
            qml.broadcast(qml.MultiRZ, wires, pattern='all_to_all', parameters=[windngs[1]*np.pi*(1-features[0])*(1-features[1])])####winding 2

def ansatz_mixed(params, features, windings, wires, reps=2):
    for layer in range(reps):
        for wire in wires:
            qml.RY(params[layer*len(wires)*2+wire], wire)
            qml.RZ(params[layer*len(wires)*2+wire+1], wire)
        qml.broadcast(qml.CZ, wires, pattern='all_to_all')
    
        qml.broadcast(qml.Hadamard, wires, pattern='single')
        qml.broadcast(qml.RZ, wires, pattern='single', parameters=windings[0]*np.pi*features)####winding 1
        if len(features)>2:
            qml.broadcast(qml.MultiRZ, wires, pattern='all_to_all', parameters=features_to_broadcast(features, 'all_to_all', windings))####winding 2
        else:
            qml.broadcast(qml.MultiRZ, wires, pattern='all_to_all', parameters=[windings[1]*np.pi*(1-features[0])*(1-features[1])])####winding 2

def kernel_matrix(X, params, windings, kernel_circ, assume_normalized_kernel=False):
    states = np.array([kernel_circ(x, params, windings) for x in X])
    
    N = len(X)
    matrix = [0] * N**2

    for i in range(N):
        for j in range(i, N):
            if assume_normalized_kernel and i == j:
                matrix[N * i + j] = 1.0
            else:
                matrix[N * i + j] = np.abs(np.dot(states[i],np.conj(states[j])))**2
                matrix[N * j + i] = matrix[N * i + j]

    return np.array(matrix).reshape((N, N))

def target_alignment(X, Y, params, windings, kernel_circ, assume_normalized_kernel=False, rescale_class_labels=True):
    
    K = kernel_matrix(X, params, windings, kernel_circ, assume_normalized_kernel=assume_normalized_kernel)

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product

def target_alignment_matrix(ker_matrix, Y, rescale_class_labels=True):
    
    K = ker_matrix

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product

def test_matrix(X_train, X_test, params, windings, kernel_circ):
    states_train = np.array([kernel_circ(x, params, windings) for x in X_train])
    states_test = np.array([kernel_circ(x, params, windings) for x in X_test])
    
    M = len(X_train)
    N = len(X_test)
    matrix = [0] * N*M

    for i in range(N):
        for j in range(i, M):
            matrix[M * i + j] = np.abs(np.dot(states_train[j],np.conj(states_test[i])))**2
            matrix[N * j + i] = matrix[M * i + j]

    return np.array(matrix).reshape((N, M))