#################################################
# majority_vote.py
# name: Yiyun Wei
# andrew id: yiyunwei
#################################################

import math
import sys

if __name__ == '__main__':
    trin = sys.argv[1]
    tein = sys.argv[2]
    trout = sys.argv[3]
    teout = sys.argv[4]
    metrics = sys.argv[5]

def main():
    array1 = []
    f1 = open(trin, "r")
    for line in f1:
        l = line.split('\t')
        array1.append(l)

    zeroes, ones = 0, 0
    for i in array1:
        if i[len(i) - 1] == '0\n':
            zeroes += 1
        elif i[len(i) - 1] == '1\n':
            ones += 1

    f2 = open(tein, "r")
    f3 = open(trout, "w+")
    f4 = open(teout, "w+")
    f5 = open(metrics, "w+")

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

    for i in array1:
        if(i[0] == "0" or i[0] == "1"):
            f3.write(str(majority) + "\n")

    array2 = []

    for line in f2:
        l = line.split('\t')
        array2.append(l)
        if(l[0] == "0" or l[0] == "1"):
            f4.write(str(majority) + "\n")

    testzeroes, testones, testcount = 0, 0, 0
    for i in array2:
        if i[len(i) - 1] == '0\n':
            testzeroes += 1
        elif i[len(i) - 1] == '1\n':
            testones += 1
    
    if majority == 1:
        testcount = testones
    else:
        testcount = testzeroes

    trerr = 1 - (majorcount/(zeroes + ones))
    teerr = 1 - (testcount/(testzeroes + testones))
    f5.write("error(train): " + str(trerr) + "\n")
    f5.write("error(test): " + str(teerr) + "\n")


if __name__ == '__main__':
    main()
