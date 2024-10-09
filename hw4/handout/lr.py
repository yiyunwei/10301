import numpy as np
import argparse
from matplotlib import pyplot as plt


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> np.ndarray:
    # TODO: Implement `train` using vectorization
    for j in range(num_epoch):
        grad = [0] * len(theta)
        for i in range(len(y)):
            grad = (sigmoid(np.dot(theta,X[i]))-y[i])*X[i]
            theta = theta - learning_rate*grad
        #    for k in range(len(grad)):
        #        grad[k] += (sigmoid(np.dot(theta,X[i]))-y[i])*X[i][k]
        #const = np.array(len(y))
        #grad = grad/const
        
    return theta


def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    res = [0]*len(X)
    for i in range(len(X)):
        temp = np.dot(theta,X[i])
        if(temp > 0):
            res[i] = 1
        else:
            res[i] = 0
    return res

def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    count = 0
    for i in range(len(y)):
        if(y[i] != y_pred[i]):
            count += 1
    return count/len(y)

def main():
    f1 = open(args.train_input, "r")
    f2 = open(args.validation_input, "r")
    f3 = open(args.test_input, "r")
    f4 = open(args.train_out, "w+")
    f5 = open(args.test_out, "w+")
    f6 = open(args.metrics_out, "w+")

    trainData = []
    for line in f1:
        l = line.split('\t')
        trainData.append(l)
    for i in range(len(trainData)):
        trainData[i][len(trainData[0])-1] = trainData[i][len(trainData[0])-1].strip()
    theta = [0] * (len(trainData[0]))

    y = [0] * len(trainData)
    for i in range(len(trainData)):
        y[i] = float(trainData[i][0])

    X = trainData
    for i in range(len(trainData)):
        X[i].pop(0)
    intercept = [1]*len(X)
    X = np.column_stack((X, intercept))

    theta = np.array(theta, dtype=float)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    valData = []
    for line in f2:
        l = line.split('\t')
        valData.append(l)
    for i in range(len(valData)):
        valData[i][len(valData[0])-1] = valData[i][len(valData[0])-1].strip()
    theta = [0] * (len(valData[0]))

    valy = [0] * len(valData)
    for i in range(len(valData)):
        valy[i] = float(valData[i][0])

    valX = valData
    for i in range(len(valData)):
        valX[i].pop(0)
    intercept = [1]*len(valX)
    valX = np.column_stack((valX, intercept))

    valX = np.array(valX, dtype=float)
    valy = np.array(valy, dtype=float)

    xaxis = list(range(1, 1001))
    yaxis = [0] * 1000
    yaxis2 = [0] * 1000
    yaxis3 = [0] * 1000
    for i in range(1000):
        theta1 = [0] * (len(trainData[0])+1)
        theta2 = [0] * (len(trainData[0])+1)
        theta3 = [0] * (len(trainData[0])+1)
        theta1 = train(theta1, X, y, i, 0.1)
        theta2 = train(theta2, X, y, i, 0.01)
        theta3 = train(theta3, X, y, i, 0.001)
        temp = 0
        temp2 = 0
        temp3 = 0
        for j in range(len(X)):
            if (y[j] == 0):
                temp += np.log(1-sigmoid(np.dot(theta1,X[j])))
                temp2 += np.log(1-sigmoid(np.dot(theta2,X[j])))
                temp3 += np.log(1-sigmoid(np.dot(theta3,X[j])))
            else:
                temp += np.log(sigmoid(np.dot(theta1,X[j])))
                temp2 += np.log(sigmoid(np.dot(theta2,X[j])))
                temp3 += np.log(sigmoid(np.dot(theta3,X[j])))
    #     for m in range(len(valX)):
    #         if (valy[m] == 0):
    #             temp2 += np.log(1-sigmoid(np.dot(theta1,valX[m])))
    #         else:
    #             temp2 += np.log(sigmoid(np.dot(theta1,valX[m])))
        yaxis[i] = (-1/len(X))*temp
        yaxis2[i] = (-1/len(X))*temp2
        yaxis3[i] = (-1/len(X))*temp3
    #     yaxis2[i] = (-1/len(valX))*temp2
    plt.plot(xaxis, yaxis)
    plt.plot(xaxis, yaxis2)
    plt.plot(xaxis, yaxis3)
    plt.show()

    
    theta = train(theta, X, y, args.num_epoch, args.learning_rate)
    trainPred = predict(theta, X)
    trainErr = compute_error(trainPred, y)



    testData = []
    for line in f3:
        l = line.split('\t')
        testData.append(l)
    for i in range(len(testData)):
        testData[i][len(testData[0])-1] = testData[i][len(testData[0])-1].strip()

    y = [0] * len(testData)
    for i in range(len(testData)):
        y[i] = float(testData[i][0])
    y = np.array(y, dtype=float)

    X = testData
    for i in range(len(testData)):
        X[i].pop(0)
    intercept = [1]*len(X)
    X = np.column_stack((X, intercept))
    X = np.array(X, dtype=float)

    testPred = predict(theta, X)
    #print(testPred, y)
    testErr = compute_error(testPred, y)

    for i in trainPred:
        f4.write(str(i) + "\n")

    for i in testPred:
        f5.write(str(i) + "\n")
    
    #print(trainErr, testErr)
    
    f6.write("error(train): " + str(trainErr) + "\n")
    f6.write("error(test): " + str(testErr) + "\n")

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    main()
