import random
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
import time
from sklearn.svm import SVC

#data matrices
q1_dataset = sio.loadmat('q1_dataset.mat')
hog_features_train = q1_dataset['hog_features_train']
hog_features_test = q1_dataset['hog_features_test']
inception_features_train = q1_dataset['inception_features_train']
inception_features_test = q1_dataset['inception_features_test']
superclass_labels_train = q1_dataset['superclass_labels_train']
superclass_labels_test = q1_dataset['superclass_labels_test']
subclass_labels_train = q1_dataset['subclass_labels_train']
subclass_labels_test = q1_dataset['subclass_labels_test']

#upon inspection hog features are standardized but inception is not. Thus Standardizing inception
inception_features_train = (inception_features_train - np.mean(inception_features_train, axis=0)) / np.std(inception_features_train, axis=0)
inception_features_test = (inception_features_test - np.mean(inception_features_test, axis=0)) / np.std(inception_features_test, axis=0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def hyp(w, X):
    return sigmoid(np.dot(X, w))

def gradAscent(w, X, Y, batchSize=1, iters=1000, alpha=0.001, gType='mini'):
    dataSet = np.c_[X,Y]
    for i in range(iters):
        if (gType == 'batch' and (i+1) % 100 == 0):
            print("Weights at i = ", i+1, " :", w)
        startLoc = 0
        if (batchSize < X.shape[0]):
            dataSet = dataSet[np.random.permutation(dataSet.shape[0]),:]
        for j in range(int(X.shape[0] / batchSize)):
            x = dataSet[startLoc:startLoc + batchSize, :-1]
            y = dataSet[startLoc:startLoc + batchSize, -1]
            startLoc = (startLoc + batchSize) % X.shape[0]
            w = w + alpha * np.dot(x.T, y - hyp(w, x))  
    return w

def initialW(mean, sigma, size):
    return np.random.normal(mean, sigma, size)

def trainLogistic(X,y, batchSize=1, alpha=0.0001, gType='mini'):
    biasedX = np.c_[np.ones((X.shape[0], 1)), X]
    w = initialW(0, 0.01, biasedX.shape[1])
    w = gradAscent(w, biasedX, y, batchSize, alpha=alpha, gType=gType)
    return w

def testLogistic(w, X, y):
    biasedX = np.c_[np.ones((X.shape[0], 1)), X]
    predictions = ((hyp(w, biasedX) >= 0.5).astype(int)).flatten()
    print("Accuracy: ", np.mean(predictions == y) * 100)
    TP = sum(predictions[np.where(predictions == y)] == 1)
    TN = sum(predictions[np.where(predictions == y)] == 0)
    FP = sum(predictions[np.where(predictions != y)] == 1)
    FN = sum(predictions[np.where(predictions != y)] == 0)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    NPV = TN / (FN + TN)
    FPR = FP / (TP + FP)
    FDR = FP / (FN + TN)
    F1 = 2 * precision * recall / (precision + recall)
    F2 = 5 * precision * recall / ((4 * precision) + recall)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("NPV: ", NPV)
    print("FPR: ", FPR)
    print("FDR: ", FDR)
    print("F1: ", F1)
    print("F2: ", F2)
    print("----Confusion Matrix----")
    print("TP: ", TP, "FP: ", FP)
    print("FN: ", FN, "TN: ", TN)
# Q1.1 
# mini batch with NN features
print("---------Mini Batch NN Features---------")
start = time.time()
w = trainLogistic(inception_features_train, superclass_labels_train.flatten(), 25, 0.0001)
testLogistic(w, inception_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

# stochastic with NN features
print("---------Stochastic NN Features---------")
start = time.time()
w = trainLogistic(inception_features_train, superclass_labels_train.flatten(), 1, 0.0001)
testLogistic(w, inception_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

# mini batch with HOG features
print("---------Mini Batch HOG Features---------")
start = time.time()
w = trainLogistic(hog_features_train, superclass_labels_train.flatten(), 25, 0.0001)
testLogistic(w, hog_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

# stochastic with HOG features
print("---------Stochastic HOG Features---------")
start = time.time()
w = trainLogistic(hog_features_train, superclass_labels_train.flatten(), 1, 0.0001)
testLogistic(w, hog_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

# Q 1.2
# batch with NN features
print("---------Batch NN Features---------")
start = time.time()
w = trainLogistic(inception_features_train, superclass_labels_train.flatten(), inception_features_train.shape[0], 0.001, gType='batch')
ind = np.argpartition(w, -10)[-10:]
print("Indices of 10 most important features: ", ind)
testLogistic(w, inception_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

# batch with HOG features
print("---------Batch HOG Features---------")
start = time.time()
w = trainLogistic(hog_features_train, superclass_labels_train.flatten(), hog_features_train.shape[0], 0.0001, gType='batch')
ind = np.argpartition(w, -10)[-10:]
print("Indices of 10 most important features: ", ind)
testLogistic(w, hog_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")


#Q 1.3 Stratified K fold
def stratified_k_fold(X, Y, k):
    dataSet = np.c_[X,Y]
    labels = np.unique(Y)
    data_labels = []
    for label in labels:
        data_label_set = dataSet[np.asarray(np.where(dataSet[:, -1] == label))].squeeze()
        data_labels.append(np.random.permutation(data_label_set))
    
    data_K = []
    for i in range(k):
        data = np.ones((1,dataSet.shape[1]))
        for data_label in data_labels:
            fold_cut = int(math.ceil(data_label.shape[0] / k))
            data = np.vstack([data, data_label[i * fold_cut:((i+1) * fold_cut), :]])
        data_K.append(np.random.permutation(data[1:, :]))
        
    return data_K


#SVM model helper methods
def getLeftOut(data_K, i):
    data_K = np.asarray(data_K)
    train = [x for x in range(len(data_K)) if x != i]
    return data_K[np.asarray(train)], data_K[i]
    
def SVM_train_k_fold(data_K, C=1.0, kernel='linear', degree=3, gamma=2):
    accuracy = []
    for i in range (len(data_K)):
        train, test = getLeftOut(data_K, i)
        train = train.reshape((-1, test.shape[1]))
        svclassifier = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        svclassifier.fit(train[:, :-1], train[:, -1])
        final_prediction = svclassifier.predict(test[:, :-1])
        accuracy.append(np.mean(final_prediction == test[:,-1]) * 100)
    return np.mean(accuracy)
        
def SVM_train(X, Y, C=None, kernel='linear', degree=None, gamma=None, k_folds=5):
    data_K = stratified_k_fold(X, Y, k_folds)
    accuracies = []
    if kernel is 'linear':
        for c in C:
            accuracies.append(SVM_train_k_fold(data_K, C=c, kernel='linear'))
            print("C: ", c, " Accuracy: ", accuracies[-1])
        C_final = C[np.argmax(accuracies)]
        print("fin_C: ", C_final)
        svclassifier = SVC(C=C_final, kernel='linear')
    elif kernel is 'rbf':
        for c in C:
            for g in gamma:
                accuracies.append(SVM_train_k_fold(data_K, C=c, kernel='rbf', gamma=g))
                print("C: ", c, " G: ", g, " Accuracy: ", accuracies[-1])
        loc = np.argmax(accuracies)
        C_final = C[(int)(loc / len(gamma))]
        if len(C) is 1:
            gamma_final = gamma[loc]
        else:
            gamma_final = gamma[loc % len(C)]
        print("fin_C: ", C_final, " fin_gamma: ", gamma_final)
        svclassifier = SVC(C=C_final, kernel='rbf', gamma=gamma_final)
    elif kernel is 'poly':
        for d in degree:
            for g in gamma:
                accuracies.append(SVM_train_k_fold(data_K, C=C[0], kernel='poly', gamma=g, degree=d))
                print("C: ", C[0], " G: ", g, " D: ", d, " Accuracy: ", accuracies[-1])
        loc = np.argmax(accuracies)
        degree_final = degree[(int)(loc / len(gamma))]
        gamma_final = gamma[loc % len(degree)]
        print("fin_degree: ", degree_final, " fin_gamma: ", gamma_final, " fin_C: ", C[0])
        svclassifier = SVC(C=C[0], kernel='poly', gamma=gamma_final, degree=degree_final)
    svclassifier.fit(X, Y)
    return svclassifier

def con_mat(final_prediction, y):
    u = np.unique(y)
    confusion_matrix = np.zeros((u.shape[0],u.shape[0]))
    for i in range(u.shape[0]**2):
        predicted = (int)(i / u.shape[0])
        actual = i % u.shape[0]
        pred_loc = np.array(np.where(final_prediction == (u.shape[0] - 1 - predicted))).flatten()
        act_loc = np.array(np.where(y == (u.shape[0] - 1 - actual))).flatten()
        val = (np.array(np.intersect1d(pred_loc, act_loc)).flatten()).shape[0]
        confusion_matrix[predicted][actual] = val
    return confusion_matrix

def SVM_test(classifier, X, y):
    final_prediction = classifier.predict(X)
    u = np.unique(y)
    confusion_matrix = con_mat(final_prediction, y)
    
    
    print("----Confusion Matrix----")
    if u.shape[0] == 2:
        print(confusion_matrix)
        print("Accuracy: ", np.mean(final_prediction == y) * 100)
        precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
        print("Precision: ", precision)
        print("Recall: ", recall)
    else:
        print(np.flipud(np.fliplr(confusion_matrix)))
        print('Accuracy, avg: ', np.sum(confusion_matrix.diagonal() * 100/confusion_matrix.sum(axis=0))/u.shape[0])
        y_class = np.zeros((u.shape[0], y.shape[0]))
        pred_class = np.zeros((u.shape[0], y.shape[0]))
        TP = []
        FP = []
        FN = []
        precisions = []
        recalls = []
        F1s = []
        print('Accuracy, class:', confusion_matrix.diagonal() * 100/confusion_matrix.sum(axis=0))
        for i in range(u.shape[0]):
            y_class[i][np.array(np.where(y == u[i]))] = 1
            pred_class[i][np.array(np.where(final_prediction == u[i]))] = 1
            TP.append(sum(pred_class[i][np.where(pred_class[i] == y_class[i])] == 1))
            FP.append(sum(pred_class[i][np.where(pred_class[i] != y_class[i])] == 1))
            FN.append(sum(pred_class[i][np.where(pred_class[i] != y_class[i])] == 0))
            precisions.append(TP[-1] / (TP[-1] + FP[-1]))
            recalls.append(TP[-1] / (TP[-1] + FN[-1]))
            F1s.append(2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1]))
        precision_macro = np.sum(np.array(precisions)) / len(precisions)
        recall_macro = np.sum(np.array(recalls)) / len(recalls)
        F1_macro = np.sum(np.array(F1s)) / len(F1s)
        precision_micro = np.sum(np.array(TP)) / (np.sum(np.array(TP)) + np.sum(np.array(FP)))
        recall_micro = np.sum(np.array(TP)) / (np.sum(np.array(TP)) + np.sum(np.array(FN)))
        F1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        print("Precision_micro: ", precision_micro)
        print("Precision_macro: ", precision_macro)
        print("Recall_micro: ", recall_micro)
        print("Recall_macro: ", recall_macro)
        print("F1_micro: ", F1_micro)
        print("F1_macro: ", F1_macro)
        
#Superclass Classification
# Q 1.4 Soft Margin SVM Linear kernel
print("---------Soft Margin SVM Linear Kernel Hog---------")
start = time.time()
classifier = SVM_train(hog_features_train, superclass_labels_train.flatten(), C=[0.01, 0.1, 1, 10, 100])
SVM_test(classifier, hog_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

print("---------Soft Margin SVM Linear Kernel NN---------")
start = time.time()
classifier = SVM_train(inception_features_train, superclass_labels_train.flatten(), C=[0.01, 0.1, 1, 10, 100])
SVM_test(classifier, inception_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

# Q 1.5 Hard Margin SVM RBF kernel
print("---------Hard Margin SVM RBF Kernel Hog---------")
start = time.time()
classifier = SVM_train(hog_features_train, superclass_labels_train.flatten(), C=[1e10], kernel='rbf', gamma=[2**(-4), 2**(-3), 2**(-2), 2**(-1), 2**(0), 2**(1), 2**(6)])
SVM_test(classifier, hog_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

print("---------Hard Margin SVM RBF Kernel NN---------")
start = time.time()
classifier = SVM_train(inception_features_train, superclass_labels_train.flatten(), C=[1e10], kernel='rbf', gamma=[2**(-4), 2**(-3), 2**(-2), 2**(-1), 2**(0), 2**(1), 2**(6)])
SVM_test(classifier, inception_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

# Q 1.6 Soft Margin SVM RBF kernel
print("---------Soft Margin SVM RBF Kernel Hog---------")
start = time.time()
classifier = SVM_train(hog_features_train, superclass_labels_train.flatten(), C=[10**(-2), 1, 10**(2)], kernel='rbf', gamma=[2**(-2), 2, 2**(6)])
SVM_test(classifier, hog_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

print("---------Soft Margin SVM RBF Kernel NN---------")
start = time.time()
classifier = SVM_train(inception_features_train, superclass_labels_train.flatten(), C=[10**(-2), 1, 10**(2)], kernel='rbf', gamma=[2**(-2), 2, 2**(6)])
SVM_test(classifier, inception_features_test, superclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")


# Subclass Classification
# Q 1.7 Soft Margin SVM RBF one vs all kernel
print("---------Soft Margin SVM RBF Kernel one vs all Hog---------")
start = time.time()
classifier = SVM_train(hog_features_train, subclass_labels_train.flatten(), C=[10**(-2), 1, 10**(2)], kernel='rbf', gamma=[2**(-2), 2, 2**(6)])
SVM_test(classifier, hog_features_test, subclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

print("---------Soft Margin SVM RBF Kernel one vs all NN---------")
start = time.time()
classifier = SVM_train(inception_features_train, subclass_labels_train.flatten(), C=[10**(-2), 1, 10**(2)], kernel='rbf', gamma=[2**(-2), 2, 2**(6)])
SVM_test(classifier, inception_features_test, subclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

# Q 1.8 Hard Margin SVM Poly one vs all kernel
print("---------Hard Margin SVM poly Kernel one vs all Hog---------")
start = time.time()
classifier = SVM_train(hog_features_train, subclass_labels_train.flatten(), C=[1e10], degree=[3, 5, 7], kernel='poly', gamma=[2**(-2), 2, 2**(6)])
SVM_test(classifier, hog_features_test, subclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

print("---------Hard Margin SVM poly Kernel one vs all NN---------")
start = time.time()
classifier = SVM_train(inception_features_train, subclass_labels_train.flatten(), C=[1e10], degree=[3, 5, 7], kernel='poly', gamma=[2**(-2), 2, 2**(6)])
SVM_test(classifier, inception_features_test, subclass_labels_test.flatten())
end = time.time()
print("took ", end - start, " seconds to finsh.")

#Q2
def pltImg(X, label, fig, j):
    for i in range(j,j+5):
        img = (X[i%5,:]).reshape((85,125))
        img1 = fig.add_subplot(3, 5, i+1) 
        img1.set_title(label+" "+str(i%5+1))
        plt.imshow(img)
    return i + 1

q2_dataset = sio.loadmat('q2_dataset.mat')
X = q2_dataset['data']
X = X.reshape((150, 10625))
fig = plt.figure(figsize=(85, 125))
j = pltImg(X,"Original", fig, 0)

#2.1 SVD based
start = time.time()
U, S, VT = np.linalg.svd(X)
end = time.time()

Sigma = np.zeros((X.shape[0], X.shape[1]))
Sigma[:, :X.shape[0]] = np.diag(S)
print("---------SVD based---------")
print("MSE: ", ((X - np.dot(U, np.dot(Sigma, VT)))**2).mean())
print("took ", end - start, " seconds to finsh.")
j = pltImg(np.dot(U, np.dot(Sigma, VT)),"SVD", fig, j)

#2.2 cov based
start = time.time()
m = np.mean(X, axis=0)
X_mean = X - m
X_cov = np.cov(X_mean.transpose())
eig_val, eig_vec = np.linalg.eig(X_cov)
end = time.time()

reduced_X = np.dot(X_mean, eig_vec.transpose())
reconstructed_X = (np.dot(reduced_X, eig_vec)).real + m
print("---------Covariance based---------")
print("MSE: ", ((X - reconstructed_X)**2).mean())
print("took ", end - start, " seconds to finsh.")
j = pltImg(reconstructed_X,"COV", fig, j)

plt.show()
