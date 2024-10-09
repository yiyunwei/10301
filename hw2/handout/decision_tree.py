import argparse
import numpy
import math

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.ones = None
        self.zeroes = None
    
def findMajority(data, col):
    zeroes, ones = 0, 0
    for i in data:
        if ((col == len(i) - 1 and i[col] == '0\n') or i[col] == '0'):
            zeroes += 1
        elif ((col == len(i) - 1 and i[col] == '1\n') or i[col] == '1'):
            ones += 1
    # total = zeroes + ones

    majority, majorcount = 0, 0
    if zeroes > ones:
        majority = 0
        majorcount = zeroes
    elif ones > zeroes:
        majority = 1
        majorcount = ones
    else:
        majority = 1
        majorcount = ones

    return (zeroes, ones, majorcount, majority)

def findEntropy(data):
    if(len(data) == 0):
        return 0
    
    zeroes, ones, majorcount, majority = findMajority(data, len(data[0])-1)
    total = zeroes + ones

    majorfrac = majorcount/total
    minorfrac = (total-majorcount)/total
    if(majorfrac == 1):
        ent = 0
    else:
        ent = -1*(majorfrac*math.log2(majorfrac))-(minorfrac*math.log2(minorfrac))

    return ent

def findMutInfo(totEnt, data, col):
    (zeroes, ones, majorcount, majority) = findMajority(data, col)
    total = zeroes + ones

    temp1, temp0 = [], []
    for row in data:
        if(row[col] == '1'):
            temp1.append(row)
        elif(row[col] == '0'):
            temp0.append(row)

    return totEnt - (ones/total)*(findEntropy(temp1)) - (zeroes/total)*(findEntropy(temp0))

def findMax(entData):
    highest, ind = 0, 0
    for i in range(len(entData)):
        if (entData[i] > highest):
            highest = entData[i]
            ind = i
    return ind

def majorityVote(data):
    zeroes, ones, majorcount, majority = findMajority(data, len(data[0])-1)
    return zeroes, ones, majorcount, majority

def makeNode(data, entData, depth):
    col = findMax(entData)
    newNode = Node()
    newNode.zeroes, newNode.ones, majorcount, majority = findMajority(data, len(data[0])-1)

    #if (depth == args.max_depth or len(entData) == 0 or entData[col] == 0):
    if (depth == args.max_depth or len(entData) == 0 or findEntropy(data) == 0):
        #if (len(data) == 0):
            #print("hiiii\n")
        newNode.vote = majority
        return newNode
    else:
        #print("yoyo\n")
        newNode.attr = data[0][col]
        temp1, temp0 = [], []
        temp1.append(numpy.delete(data[0], col, None))
        temp0.append(numpy.delete(data[0], col, None))
        for row in data:
            if(row[col] == '1'):
                temp1.append(numpy.delete(row, col, None))
            elif(row[col] == '0'):
                temp0.append(numpy.delete(row, col, None))

        # newNode.ones, newNode.zeroes = len(temp1)-1, len(temp0)-1
        totEnt1 = findEntropy(temp1)
        totEnt0 = findEntropy(temp0)
        newEntData1, newEntData0 = [], []

        for x in range(len(temp1[0])-1):
            newEntData1.append(findMutInfo(totEnt1, temp1, x))
            newEntData0.append(findMutInfo(totEnt0, temp0, x))
        
        #print("temp1 len: " + str(len(temp1)) + " temp0 len: " + str(len(temp0)) + "\n")
        #newNode.left = makeNode(temp1, numpy.delete(entData, col, None), depth+1)
        newNode.left = makeNode(temp1, newEntData1, depth+1)
        #newNode.right = makeNode(temp0, numpy.delete(entData, col, None), depth+1)
        newNode.right = makeNode(temp0, newEntData0, depth+1)
    
    return newNode

def search(node, array, row):
    if(node.vote != None):
        return node.vote
    else:
        i, val = 0, 0
        for att in array[0]:
            if att == node.attr:
                val = array[row][i]
                break
            else: i+=1
        if(val == '1'):
            return search(node.left, array, row)
        else:
            return search(node.right, array, row)

def predict(node, array):
    result = [0]*(len(array)-1)
    for x in range(1, len(array)):
        result[x-1] = search(node, array, x)
    return result
        
def printfunc(node, depth):
    if(node.vote == None):
        print("[" + str(node.zeroes) + " 0/" + str(node.ones) + " 1]")
        print("| " * depth, end="")
        print(node.attr + " = 1: ", end="")
        if(node.left != None):
            printfunc(node.left, depth + 1)
        print("| " * depth, end="")
        print(node.attr + " = 0: ", end="")
        if(node.right != None):
            printfunc(node.right, depth + 1)
    else:
        return print("[" + str(node.zeroes) + " 0/" + str(node.ones) + " 1]")

def main():
    f1 = open(args.train_input, "r")
    f2 = open(args.test_input, "r")
    f3 = open(args.train_out, "w+")
    f4 = open(args.test_out, "w+")
    f5 = open(args.metrics_out, "w+")

    data = []
    for line in f1:
        l = line.split('\t')
        data.append(l)

    testData = []
    for line in f2:
        l = line.split('\t')
        testData.append(l)

    totEnt = findEntropy(data)
    entData = []

    for x in range(len(data[0])-1):
        entData.append(findMutInfo(totEnt, data, x))

    root = makeNode(data, entData, 0)

    trainRes = predict(root, data)
    testRes = predict(root, testData)

    trainErrs, testErrs = 0, 0

    for i in range(len(trainRes)):
        f3.write(str(trainRes[i]) + "\n")
        if ((str(trainRes[i])+"\n") != data[i+1][len(data[0])-1]):
            trainErrs += 1

    for j in range(len(testRes)):
        f4.write(str(testRes[j]) + "\n")
        if ((str(testRes[j])+"\n") != testData[j+1][len(testData[0])-1]):
            testErrs += 1

    trainError = trainErrs/len(trainRes)
    testError = testErrs/len(testRes)
    f5.write("error(train): " + str(trainError) + "\n")
    f5.write("error(test): " + str(testError) + "\n")

    printfunc(root, 1)

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    args = parser.parse_args()

    main()