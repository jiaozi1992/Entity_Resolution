# -*- coding: utf-8 -*-
from myComparison import exact
from myComparison import tokenset
from myComparison import longcommonSeq
from myComparison import Editex
from myComparison import QGram
from myComparison import Jaro
from myComparison import bagDist
from myComparison import compress
from myComparison import compressbz2
from myComparison import editDistance
from myComparison import KeyDiff
from myComparison import PosQGram
from myComparison import OntoLCS
from myComparison import SWDist
from myComparison import ContainsString
# from myComparison import EncodeString
from myComparison import TruncateString
# from myComparison import SyllAlDist
from myComparison import NumericAbs
from myComparison import NumericPerc
from myComparison import NumericPerc
from myComparison import winkler
from nltk import ngrams
import timeit
import sys
import getopt
import re

# sys.setdefaultencoding() does not exist, here!
reload(sys)
sys.setdefaultencoding('UTF8')



def encodeingHelper(text):
    try:
        res = text.decode('utf-8')
    except UnicodeDecodeError:
        res = text.decode('latin-1')
    return res

def usage():
    print "<program.py> <afile> <bfile> -o <PairwiseFeatureFile> -f <BlockFeature>"
    print "BlockFeature: the column name that you want to use for blocking. If none, the third column will be used for blocking"

if (len(sys.argv[1:]) == 0 or sys.argv[1] == "-h"):
    usage()
    sys.exit(2)

test_argv = sys.argv[1:]
if len(test_argv) < 3:
    usage()
    sys.exit(2)

try:
    f1_name = sys.argv[1]
    a = open(f1_name)
except IOError:
    print "Could Not Read First File"
    sys.exit(2)

try:
    f2_name = sys.argv[2]
    b = open(f2_name)
except IOError:
    print "Could Not Read Second File B"
    sys.exit(2)


argv = sys.argv[3:]
if len(argv) !=2 and len(argv) != 4:
    usage()
    sys.exit(2)

try:
    opts, args = getopt.getopt(argv, "o:f:", ["output=,BlockFeature="])

except getopt.GetoptError:
    usage()
    sys.exit(2)

block_key_name = ""

for opt, arg in opts:
    if opt in ("-o", "--output"):
        f3_name = arg

    elif opt in ("-f", "--BlockFeature"):
        block_key_name = arg
        # block_key_index = int(arg) - 1

numericalFeature = raw_input("Enter the column name of which the value is numerical. If multiple,split by \",\". If none, enter -1: ")

LongFeature = raw_input("Enter the column name of which the value that you think are too long(normally more than 300 characters. If multiple, split by \",\".If none, enter -1: ")


# blocking part
block = {} # store all block pairs, key:3gram, value:id
n = 3 # which gram to use for tokenizing the string
header_list = open(f1_name).readlines()[0].strip().split(",")
header_list2 = open(f1_name).readlines()[0].strip().split(",")

if header_list!=header_list2:
    print "Please Use Same Schema for File 1 and File 2"
    sys.exit()

numericalFeatureList = []
if numericalFeature == "-1":
    numericalFeatureList = []
else:
    # numericalFeatureList = [int(i)-1 for i in numericalFeature.split(",")]
    numericalFeatureNameList = numericalFeature.split(",")
    for i in range(0, len(header_list)):
        if header_list[i] in numericalFeatureNameList:
            numericalFeatureList.append(i)
    if numericalFeatureList == []:
        print "Wrong Numerical Column Name"
        sys.exit()

# print "numericalFeatureList",numericalFeatureList

LongFeatureList = []
if LongFeature == "-1":
    LongFeatureList = []
else:
    LongFeatureNameList = LongFeature.split(",")
    for i in range(0, len(header_list)):
        if header_list[i] in LongFeatureNameList:
            LongFeatureList.append(i)

    if LongFeatureList == []:
        print "Wrong Long Column Name"
        sys.exit()

# print "LongFeatureList",LongFeatureList

block_key_index = -1

if block_key_name != "":
    for i in range(0,len(header_list)):
        if header_list[i] == block_key_name:
            block_key_index = i
    if block_key_index == -1:
        print "Wrong Column Name for Blocking"
        sys.exit()

af = open(f1_name).readlines()[1:]
a_firstId = sys.maxint

if block_key_index < 0 and len(header_list)>2:
    block_key_index = 3

if block_key_index < 0 and len(header_list)==2:
    block_key_index = 2

for l in af:
    # if block_key_index < 0:
    #     l = l.strip()
    #     ll = list(ngrams(l, n))
    l = l.strip().split(",")
    a_firstId = min(a_firstId, int(l[0]))
    l[block_key_index] = list(ngrams(l[block_key_index], n))
    ll = l[block_key_index]

    tmp = []
    for i in range(0,len(ll)):
        item = ll[i]
        item = "".join(item)
        tmp.append(item)
        if item not in block:
            block[item] = []
        block[item].append(l[0])

bf = open(f2_name).readlines()[1:]
b_firstId = sys.maxint

print "Block Key: ", header_list[block_key_index]
for l in bf:
    l = l.strip().split(",")
    b_firstId = min(b_firstId, int(l[0]))
    l[block_key_index] = list(ngrams(l[block_key_index], n))
    ll = l[block_key_index]

    tmp = []
    for i in range(0, len(ll)):
        item = ll[i]
        item = "".join(item)
        tmp.append(item)
        if item not in block:
            block[item] = []
        block[item].append(l[0])

all_pairs_num = len(af)*len(bf)

block_size = int(raw_input("The Largest Size You want for Each Block(the larger, the more paris will be compared(33 is recommended)): "))

print "Start Blocking..."
block2 = {}
for k in block:
    if len(block[k]) <= block_size:
        block2[k] = block[k]
#
pairs = set()

split_num = max(a_firstId,b_firstId)
# print "split_num",split_num

for k in block2:
    for i in range(0,len(block2[k])):
        for j in range(i+1,len(block2[k])):
            if block2[k][i].isdigit() and block2[k][j].isdigit():
                inum = int(block2[k][i])
                jnum = int(block2[k][j])
                # print "inum",inum
                # print "jnum",jnum
                if inum < split_num and jnum >= split_num or (inum >= split_num and jnum < split_num):
                    small = min(int(block2[k][i]),int(block2[k][j]))
                    large = max(int(block2[k][i]),int(block2[k][j]))
                    pairs.add((small,large))


if len(pairs) == 0:
    print "No Pairs to Be Compared. Please Increase the Block Size"
    sys.exit()

print "Blocking Has Been Finished"
# sim part
sim_fun = [tokenset,longcommonSeq,Editex,QGram,Jaro,bagDist,compress,
           compressbz2,editDistance,KeyDiff,PosQGram,OntoLCS,SWDist,
           ContainsString,TruncateString,winkler]

sim_fun2 = [Jaro,winkler,ContainsString]


num_fun = [NumericAbs,NumericPerc]

sim_funString = ["tokenset","longcommonSeq","Editex","QGram","Jaro","bagDist","compress",
           "compressbz2","editDistance","KeyDiff","PosQGram","OntoLCS","SWDist",
           "ContainsString","TruncateString","winkler"]

sim_fun2String = ["Jaro","winkler","ContainsString"]

num_funString = ['NumericAbs','NumericPerc']

if a_firstId > b_firstId:
    tmp_name = f2_name
    f2_name = f1_name
    f1_name = tmp_name
    tmp_id = b_firstId
    b_firstId = a_firstId
    a_firstId = tmp_id

af = open(f1_name).readlines()[1:]
asim = {} # for calculating the sim
for l in af:
    l = l.strip().split(",")
    asim[int(l[0])] = l

bf = open(f2_name).readlines()[1:]
bsim = {} # for calculating the sim
for l in bf:
    l = l.strip().split(",")
    if l[0] == '':
        continue
    bsim[int(l[0])] = l

# write HEADER
# l = len(asim[0])
header= []
header.append("rec_id1")
header.append("rec_id2")
header.append("Str-Exact-class-class")

for attr_index in range(2,len(header_list)):
    if attr_index not in LongFeatureList and attr_index not in numericalFeatureList:
        for sf in sim_funString:
            s = str(sf) + '-' + str(header_list[attr_index]) + '-' + str(header_list[attr_index])
            header.append(s)
    if attr_index in LongFeatureList:
        for sf in sim_fun2String:
            s = str(sf) + '-' + str(header_list[attr_index]) + '-' + str(header_list[attr_index])
            header.append(s)
    if attr_index in numericalFeatureList:
        for sf in num_funString:
            s = str(sf) + '-' + str(header_list[attr_index]) + '-' + str(header_list[attr_index])
            header.append(s)

pairs2 = []
for i in pairs:
    small = min(i[0],i[1])
    large = max(i[0],i[1])
    pairs2.append([small,large])

# generate sim file
start = timeit.default_timer()

cnt = 0
pairs_completed = 0

# no class, -1
with open(f3_name,"w") as f:
    f.write(",".join(header))
    f.write("\n")
    for p in pairs2:
        if cnt % (int(len(pairs2)/10.0)) == 0:
            now = timeit.default_timer()
            print "It takes",round((now - start)/60.0,2),"minutes to complete pairwise feature generation for ", round(cnt/(len(pairs2)/10.0),0)*10,"% of all pairs"
        aindex = p[0]
        bindex = p[1]
        cur = []
        al = asim[aindex]
        bl = bsim[bindex]
        cur.append(str(al[0]))
        cur.append(str(bl[0]))
        if len(al[1]) == 0 or len(bl[1]) == 0:
            cur.append("-1.0")
        else:
            cur.append(str(exact(al[1],bl[1])))
        for attr_index in range(2,len(header_list)):
            if attr_index in numericalFeatureList:
                for sf2 in num_fun:
                    av = al[attr_index]
                    bv = bl[attr_index]
                    av = re.sub(r'\D', "", av)
                    bv = re.sub(r'\D', "", bv)
                    cur.append(str(sf2(av, bv)))

            if attr_index not in LongFeatureList and attr_index not in numericalFeatureList:
                av = al[attr_index]
                bv = bl[attr_index]
                for sf in sim_fun:
                    simv = sf(encodeingHelper(av),encodeingHelper(bv))
                    cur.append(str(simv))

            if attr_index in LongFeatureList:
                av = al[attr_index]
                bv = bl[attr_index]
                for sf in sim_fun2:
                    simv = sf(encodeingHelper(av),encodeingHelper(bv))
                    cur.append(str(simv))

        cnt += 1
        f.write(",".join(cur))
        f.write("\n")