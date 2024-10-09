#################################################
# inspection.py
# name: Yiyun Wei
# andrew id: yiyunwei
#################################################

import math
import sys

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]

def main():
    array1 = []
    f1 = open(infile, "r")
    f2 = open(outfile, "w+")
    for line in f1:
        l = line.split('\t')
        array1.append(l)

    zeroes, ones = 0, 0
    for i in array1:
        if i[len(i) - 1] == '0\n':
            zeroes += 1
        elif i[len(i) - 1] == '1\n':
            ones += 1
    total = zeroes + ones

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

    majorfrac = majorcount/total
    minorfrac = (total-majorcount)/total
    ent = -1*(majorfrac*math.log2(majorfrac))-(minorfrac*math.log2(minorfrac))
    err = 1 - (majorcount/total)
    f2.write("entropy: " + str(ent) + "\n")
    f2.write("error: " + str(err) + "\n")

if __name__ == '__main__':
    main()