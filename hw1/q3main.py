import random
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

K = 8
aminoAcids = ['G', 'P', 'A', 'V', 'L', 'I', 'M', 'C', 'F', 'Y', 'W', 'H', 'K', 'R', 'Q', 'N', 'E', 'D', 'S', 'T']

#get the data from the file
testSet = np.loadtxt("q2_test_set.txt", delimiter=',')
trainSet = np.loadtxt("q2_train_set.txt", delimiter=',')
gagSeq = np.array(())
gag = np.array(())
with open("q2_gag_sequence.txt") as fileobj:
    gagString = (fileobj.read().replace('\n', '')).upper()
    lenGag = len(gagString)

for ch in gagString:
    oneHot = np.zeros((len(aminoAcids)))
    oneHot[aminoAcids.index(ch)] = 1
    gag = np.append(gag, oneHot)

for i in range(lenGag - K + 1):
    gagSeq = np.append(gagSeq, gag[(i * len(aminoAcids)): ((i * len(aminoAcids) + (K * len(aminoAcids))))])

gagSeq = np.reshape(gagSeq, ((lenGag - K + 1),(K * len(aminoAcids))))

#turn the sets into X and Y
testSetX = testSet[:,:-1]
testSetY = testSet[:,-1]
trainSetX = trainSet[:,:-1]
trainSetY = trainSet[:,-1]
#training
class estimators:
    def __init__(self, theta0, theta1, pi0, pi1):
        self.theta0 = theta0
        self.theta1 = theta1
        self.pi0 = pi0
        self.pi1 = pi1

def train(X, Y, alpha=0, k=160):
    X1 = X[np.where(Y == 1), :k]
    X0 = X[np.where(Y == 0), :k]

    pi0 = X0.shape[1] / X.shape[0]
    pi1 = X1.shape[1] / X.shape[0]

    theta0 = (np.sum(X0, axis=1) + alpha) / (X0.shape[1] + (2 * alpha))
    theta1 = (np.sum(X1, axis=1) + alpha) / (X1.shape[1] + (2 * alpha))
    return estimators(theta0, theta1, pi0, pi1)

def predict(X, est):
    X_minus = X.copy()
    X_minus = np.where(X_minus==1, 2, X_minus) 
    X_minus = np.where(X_minus==0, 1, X_minus)
    X_minus = np.where(X_minus==2, 0, X_minus)

    theta0_minus = np.multiply((1 - est.theta0), X_minus)
    theta0 = np.multiply(est.theta0, X)
    theta1_minus = np.multiply((1 - est.theta1), X_minus)
    theta1 = np.multiply(est.theta1, X)
    old_settings = np.seterr(all='ignore')
    y0 = math.log(est.pi0) + np.sum(np.log(theta0 + theta0_minus), axis=1)
    y1 = math.log(est.pi1) + np.sum(np.log(theta1 + theta1_minus), axis=1)
    np.seterr(**old_settings)
    return np.argmax(np.vstack((y0, y1)), axis=0)

def accuracy(predicted, original):
    return np.mean(predicted == original) * 100

def getIndices(predicted, gagSeq):
    indices = np.asarray(np.where(predicted == 1))
    first = indices + (K / 2) - 1
    second = indices + (K / 2)
    return np.vstack((first, second))

def getMers(trained):
    theta0 = trained.theta0.copy()
    theta1 = trained.theta1.copy()

    theta0 = np.reshape(theta0, (K, len(aminoAcids)))
    theta1 = np.reshape(theta1, (K, len(aminoAcids)))

    acid = np.asarray(aminoAcids)
    print(acid[np.argmax(theta1, axis=1)])
    print(acid[np.argmin(theta0, axis=1)])

def getMutualInfo(X, Y):
    X1 = np.asarray(X[np.where(Y == 1)])
    X0 = np.asarray(X[np.where(Y == 0)])

    P_C0 = X0.shape[0] / X.shape[0]
    P_C1 = X1.shape[0] / X.shape[0]
    
    P_U0 = (X.shape[0] - np.sum(X, axis=0)) / X.shape[0]
    P_U1 = np.sum(X, axis=0) / X.shape[0]
    
    P_U0_C0 = (X0.shape[0] - np.sum(X0, axis=0)) / X.shape[0]
    P_U0_C1 = (X1.shape[0] - np.sum(X1, axis=0)) / X.shape[0]
    P_U1_C0 = np.sum(X0, axis=0) / X.shape[0]
    P_U1_C1 = np.sum(X1, axis=0) / X.shape[0]
    old_settings = np.seterr(all='ignore')
    PUC0 = P_U0_C0 * np.log2(P_U0_C0 / (P_U0 * P_C0))
    PUC1 = P_U0_C1 * np.log2(P_U0_C1 / (P_U0 * P_C1))
    PUC2 = P_U1_C0 * np.log2(P_U1_C0 / (P_U1 * P_C0))
    PUC3 = P_U1_C1 * np.log2(P_U1_C1 / (P_U1 * P_C1))
    np.seterr(**old_settings)
    MI = PUC0 + PUC1 + PUC2 + PUC3
    return MI

#Q3.1 training the normal set to get accuracy
trained = train(trainSetX, trainSetY)
testPrediction = predict(testSetX, trained)
print("Accuracy for testSet: ",accuracy(testPrediction, testSetY))
#Q3.2 predict where the gag sequence is breaking
gagPrediction = predict(gagSeq, trained)
theIndices = getIndices(gagPrediction, gagSeq).transpose()
print("Indices for breaking of GagSeq: \n", theIndices.astype(int))
#Q3.3 get the mers which are most and least likely to break
getMers(trained)
#Q3.4 experiment of full train set with upto 10 alphas
alphas = np.arange((11))
alphasAccuracies = []
for i in range(11):
    alphasAccuracies.append(accuracy(predict(testSetX, train(trainSetX, trainSetY, alpha=i)), testSetY))

plt.plot(alphas, alphasAccuracies, label='accuracy', lw=2)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy of full trained model with different alpha values')
plt.legend()
plt.show()
# print(alphasAccuracies)
#Q3.4 experiment of 75 size train set with upto 10 alphas
smallAlphasAccuracies = []
for i in range(11):
    smallAlphasAccuracies.append(accuracy(predict(testSetX, train(trainSetX[:75], trainSetY[:75], alpha=i)), testSetY))

plt.plot(alphas, smallAlphasAccuracies, label='accuracy', lw=2)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy of 75 size trained model with different alpha values')
plt.legend()
plt.show()
# print(smallAlphasAccuracies)

#Q3.5
M_I = getMutualInfo(trainSetX, trainSetY)
MI_testX = testSetX[:,M_I.argsort()[::-1]]
MI_trainX = trainSetX[:,M_I.argsort()[::-1]]
M_IAccuracies = []
# print(M_I)
for i in range(len(M_I)):
    MITrained = train(MI_trainX, trainSetY, k=i)
    MITestPrediction = predict(MI_testX[:,:i], MITrained)
    M_IAccuracies.append(accuracy(MITestPrediction, testSetY))

# kVals = np.arange((len(M_I))).astype(int) + 1
# print(np.vstack((kVals, M_IAccuracies)).transpose())
# highest Accuracy at k
print("highest Accuracy at k", M_IAccuracies.index(max(M_IAccuracies)), " , ", max(M_IAccuracies))

#Q3.6
covarianceMatrix = np.cov(trainSetX[:,:].transpose())
eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)
sortedEigenValues = eigenValues.argsort()[::-1]
reducedPCA = eigenVectors[:, sortedEigenValues[:3]]
PVE = eigenValues[sortedEigenValues[:3]] * 100/ np.sum(eigenValues)
print("PVE of first 3 PCA: ", np.sum(PVE.real))
PCATrainX = (trainSetX.dot(reducedPCA)).real
PCATrainX0 = np.squeeze(PCATrainX[np.asarray(np.where(trainSetY == 0))])
PCATrainX1 = np.squeeze(PCATrainX[np.asarray(np.where(trainSetY == 1))])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(PCATrainX0[:,0], PCATrainX0[:,1], PCATrainX0[:,2], label='Will not Cleave', c='b', lw=0)
ax.scatter(PCATrainX1[:,0], PCATrainX1[:,1], PCATrainX1[:,2], label='Will Cleave', c='r', lw=0)
ax.set_title('Projection of AminoAcid using first 3 Principal Components')
ax.set(xlabel='Principal Component 1', ylabel='Principal Component 2', zlabel='Principal Component 3')
plt.legend()
plt.show()

trained = train(PCATrainX, trainSetY, k=3)
testPrediction = predict((testSetX.dot(reducedPCA)).real, trained)
print("Accuracy for PCA: ",accuracy(testPrediction, testSetY))