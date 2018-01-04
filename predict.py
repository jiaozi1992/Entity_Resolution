import logging
from sklearn.model_selection import train_test_split
from random import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import sys, getopt
from numpy import array
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier

def match_status_funct(red_id1,rec_id2,w_vec):
    return (w_vec[0] == 1.0)

def usage():
    print '<program.py> <file1> <file2> <PairwiseFeatureFile> -o1 <outputFile1> -o2 <outputFile2> -f <featureSelectionModel> -m <clusterModel>'
    print "<-o1 outputFile1> is the predicted file for file1"
    print "<-o2 outputFile2> is the predicted file for file2"
    print "featureSelectionModel Option: Complete, Random, VarianceThreshold,SelectKBestFeatures,SelectFromModel_LassoCV. If none, Complete will be used"
    print "clusterModel Option: LogisticRegressionCV, RandomForest. If none, RandomForest model will be used"


def find_root(i,dic):
    while dic[i] != i:
        i = dic[i]
    return i

# only use the pair with label for modeling
# id_map = {}, key: child_id, value: root_id
# apply the model to all paris in block file, if pred == 1, union find update id_map
# used_id = set(),cntID = 0
# iterate id_map, if apply class + root_id, used_id.add(root_id)
# To do: merge class_w_vec_dic/ w_vec_dic
# how to select labeled_data from data, generate train_data


if __name__ == "__main__":
    if (len(sys.argv[1:]) == 0 or sys.argv[1] == "-h"):
        usage()
        sys.exit(2)

    test_argv = sys.argv[1:]
    if len(test_argv) < 3:
        usage()
        sys.exit(2)

    try:
        a = open(sys.argv[1]).readlines()
    except IOError:
        print "Could Not Read First File"
        sys.exit(2)

    try:
        b = open(sys.argv[2]).readlines()
    except IOError:
        print "Could Not Read Second File"
        sys.exit(2)

    try:
        WeightVectorFileName = sys.argv[3]
        open(WeightVectorFileName)
    except IOError:
        print "Could Not Read Pairwise Feature File"
        sys.exit(2)

    feature = "Complete"
    clusterModel = "RandomForest"

    argv = sys.argv[4:]
    try:
        opts, args = getopt.getopt(argv, "a:b:f:m:", ["outputFilea=","outputFileb=","featureSelectionModel=","clusterModel="])

    except getopt.GetoptError:
        usage()
        sys.exit(2)

    if len(argv) < 4:
        usage()
        sys.exit()

    output1 = ""
    output2 = ""

    for opt, arg in opts:
        if opt in ("-a", "--outputFilea"):
            output1 = arg

        elif opt in ("-b", "--outputFileb"):
            output2 = arg

        elif opt in ("-f", "--featureSelectionModel"):
            feature = arg
            if feature not in ["Complete", "Random", "VarianceThreshold","SelectKBestFeatures","SelectFromModel_LassoCV"]:
                print "Wrong Feature Selection Model,Please Follow:"
                usage()
                sys.exit()
        elif opt in ("-m", "--clusterModel"):
            clusterModel = arg
            if clusterModel not in ["LogisticRegressionCV","RandomForest","Kmeans"]:
                print "Wrong Cluster Model, Please Follow:"
                usage()
                sys.exit()

    if output1 == "":
        print "No Output File1"
        sys.exit(2)

    try:
        open(output1,"w")
    except IOError:
        print "Could Not Open Output File"
        sys.exit(2)

    if output2 == "":
        print "No Output File2"
        sys.exit(2)
    try:
        open(output2,"w")
    except IOError:
        print "Could Not Open Output File"
        sys.exit(2)

    print "It may take several minutes to finish this part(usually less than 5 minutes if you have 4 million pairs between first and second file)"
    a_ = []
    b_ = []
    all_pairs = {}
    all_pairs_set = set()
    for al in a[1:]:
        al = al.strip().split(",")
        a_.append(al)

    for bl in b[1:]:
        bl = bl.strip().split(",")
        b_.append(bl)

    # for others
    p = 0
    n = 0
    for i in range(0, len(a_)):
        for j in range(0, len(b_)):
            if a_[i][1] == b_[j][1] and a_[i][0] != b_[i][0]:
                # print "a_[i][1]",a_[i][1]
                p += 1
                all_pairs[((a_[i][0], b_[j][0]))] = 1
            else:
                n += 1
                all_pairs[((a_[i][0], b_[j][0]))] = -1
            all_pairs_set.add((a_[i][0], b_[j][0]))

    all_tp = p
    assert len(all_pairs_set) == p + n
    if p > 0:
        ratio = n / p

    w = open(WeightVectorFileName).readlines()
    field_names_list = w[0].strip().split(",")[2:]
    w_vec_dict = {}
    # class_w_vec_dict = {}
    tm_set = set()
    tnm_set = set()
    for l in w[1:]:
        l = l.strip().split(",")
        if len(l) < len(field_names_list):
            print "The Weight File Is Not Compelte. You May Want To Delete The Last Line."
            sys.exit()
        pair = (l[0], l[1])
        if float(l[2]) > 0:
            tm_set.add(pair)
        elif float(l[2]) < 0:
            tnm_set.add(pair)
        w_vec_dict[pair] = [float(i) for i in l[2:]]
        # class_w_vec_dict[pair] = [float(i) for i in l[3:]]

    pairNum2 = len(field_names_list)-1
    if float(all_tp) > 0:
        pc = min(1.0,len(tm_set)/float(all_tp))
    else:
        pc = 0
    print "Pairs Completeness:", pc
    print "Reduction Ratio:",1 - (len(tm_set) + len(tnm_set))/float(len(all_pairs))

    # blokd_pairs = {}  # remain after blocking
    # blokd_pairs_set = set()
    #
    # for k in w_vec_dict:
    #     blokd_pairs[k] = 1
    #     blokd_pairs_set.add(k)
    #
    # assert len(blokd_pairs_set) == len(tm_set) + len(tnm_set)
    # assert len(blokd_pairs) == len(tm_set) + len(tnm_set)
    #
    # # unbloked test+preds
    # unbloked_test = []
    # unbloked_preds = []
    # unbloked_set = set()
    # for i in all_pairs_set:
    #     if (i[0], i[1]) not in blokd_pairs and (i[1], i[0]) not in blokd_pairs:
    #         unbloked_test.append(all_pairs[i])
    #         unbloked_preds.append(-1)
    #
    # assert len(unbloked_test) + len(blokd_pairs) == len(all_pairs_set)

    data = [] # store labeled data in the block file for training data
    labels = [] # store labeled data' labels in the block file for training data
    # ids = []
    all_data = [] # stroe all data in the block file
    all_pairs_list = []

    for (rec_id_tuple, w_vec) in w_vec_dict.iteritems():
        if w_vec[0] > -0.5:
            data.append(w_vec[1:])
            # ids.append(rec_id_tuple)
            if (rec_id_tuple in tm_set):
                labels.append(1.0)  # Match class
            else:
                labels.append(-1.0)  # Non-match class
        all_data.append(w_vec[1:])
        all_pairs_list.append(rec_id_tuple)

    # assert len(data) == len(tm_set) + len(tnm_set)
    assert len(data) == len(labels)
    assert len(all_data) == len(w_vec_dict)

    balance_input = raw_input("Enter Whether You Want to Balance the Training Data.Enter 'Y' Or 'N'(It will decrease the data size): ")

    try:
        if balance_input == "Y":
            balance = True
        if balance_input == "N":
            balance = False
    except:
        print "Enter 'Y' Or 'N'"
        sys.exit()

    data_ = array(data)
    labels_ = array(labels)
    all_data_ = array(all_data)

    if balance:
    #balance data
    # data_ is numpy array of all blocking data(as sklearn only accept numpy array)
        tptotal = []
        other = []
        tptotal_L = []
        other_L = []
        for i in range(0, len(data_)):
            if labels_[i] > 0:
                tptotal.append(data_[i])
                tptotal_L.append(labels_[i])
            else:
                other.append(data_[i])
                other_L.append(labels_[i])
        assert len(tptotal) + len(other) == len(data)
        assert len(tptotal) == len(tm_set)

        cnt3 = 0
        data2_ = []  # data2_ is the array after balancing the data
        labels2_ = []
        for i in range(0, len(data_)):
            if labels_[i] < 0 and cnt3 <= 0.3 * len(data):
                data2_.append(data_[i])
                labels2_.append(labels_[i])
                cnt3 += 1

            if labels_[i] > 0:
                data2_.append(data_[i])
                labels2_.append(labels_[i])
                cnt3 += 1

    elif not balance:
        data2_ = data_
        labels2_ = labels_

    random_num = randint(1, 1000000)
    train_size = raw_input("Enter The Percentage You Want to Use For Training Data (For example: 0.3): ")
    try:
        train_size = float(train_size)
    except:
        print "Enter a Float Number for Training Data Percentage"
        sys.exit()

    X_train2, X_test2, y_train2, y_test2 = train_test_split(data2_, labels2_, test_size= 1-train_size, random_state=random_num)
    print "Training Dataset Size: ",len(X_train2)

    if feature == "Complete":  # for diagram title
        data_ = data_
        X_train2 = X_train2
        all_data_ = all_data_

    if feature == "Random":
        featureForColumn = int(raw_input("How Many features do you want for each column(If exceed the total number of features for a column, the total number of features for the column will be used): "))
        ranges = []
        pre = field_names_list[1].split("-")[1]
        f = True
        for i in range(1, len(field_names_list[1:])):
            if field_names_list[i].split("-")[1] != pre:
                ranges.append(i - 1)
                pre = field_names_list[i].split("-")[1]
        ranges.append(len(field_names_list) - 1)

        columnNumList = []
        preNum = 0
        for i in ranges:
            num  = 0
            while num < featureForColumn and num <= (i - 1 - preNum):
                tmp = randint(preNum, i - 1)
                if tmp not in columnNumList:
                    num += 1
                    columnNumList.append(tmp)
            preNum = i
        data_ = data_[:, columnNumList]
        all_data_ = all_data_[:, columnNumList]
        X_train2 = array(X_train2)
        y_train2 = array(y_train2)
        X_train2 = X_train2[:, columnNumList]

    if feature == "VarianceThreshold":
        thold = float(raw_input("Enter Threshold: "))
        sel = VarianceThreshold(threshold=thold)
        try:
            X_train2 = sel.fit_transform(X_train2)
        except:
            print "No feature in training dataset meets the variance threshold. Please try a smaller threshold."
            sys.exit()
        data_ = sel.fit_transform(data_)
        all_data_ = sel.fit_transform(all_data_)

    if feature == "SelectKBestFeatures":
        kv = int(raw_input("Enter Number of Features: "))
        SK = SelectKBest(k=kv)
        SK.fit(X_train2, y_train2)
        data_ = SK.transform(data_)
        all_data_ = SK.transform(all_data_)
        X_train2 = SK.transform(X_train2)

    if feature == "SelectFromModel_LassoCV":
        max_iterNum = int(raw_input("Enter Max Interation Number: "))
        ls = LassoCV(max_iter=max_iterNum)
        sfm = SelectFromModel(ls, threshold=0.25)
        sfm.fit(X_train2, y_train2)
        data_ = sfm.transform(data_)
        all_data_ = sfm.transform(all_data_)
        X_train2 = sfm.transform(X_train2)

    if clusterModel == "LogisticRegressionCV":
        clf = LogisticRegressionCV().fit(X_train2, y_train2)
        # all_predicted is the predicts of all pairs in blocking file
        # all_predicted = [[probability of class 1,probability of class 0]....]
        test_predicted = clf.predict_proba(data_) # to calculate the precision,recall of the model on labeled data
        all_predicted = clf.predict_proba(all_data_)


    if clusterModel == "RandomForest":
        maxDepth = int(raw_input("Enter maxDepth for Random Forest Model: "))
        nEstimators = int(raw_input("Enter the number of trees in the forest: "))
        clf = RandomForestClassifier(n_estimators = nEstimators,max_depth=100, random_state=2000).fit(X_train2, y_train2)
        test_predicted = clf.predict_proba(data_) # to calculate the precision,recall of the model on labeled data
        all_predicted = clf.predict_proba(all_data_)

    # if clusterModel == "Kmeans":
    #     iterNum = (raw_input("Enter Maximum number of iterations of the k-means algorithm for a single run: "))
    #     try:
    #         iterNum = int(iterNum)
    #     except:
    #         print "Enter An Integer"
    #         sys.exit()
    #     KM = KMeans(n_clusters=2, max_iter=iterNum,n_jobs=10).fit(X_train2)
    #     test_predicted = KM.predict(data_)  # to calculate the precision,recall of the model on labeled data
    #     all_predicted = KM.predict(all_data_)

    # make all_predicted = [probability of class 1]
    if len(clf.classes_) > 1:
        if clf.classes_[1] > 0:
            all_predicted = [i[1] for i in all_predicted]
            test_predicted = [i[1] for i in test_predicted]
            # assert i[0] + i[1] == 1
        else:
            all_predicted = [i[0] for i in all_predicted]
            test_predicted = [i[1] for i in test_predicted]
            # assert i[0] + i[1] == 1

    if len(clf.classes_) <= 1:
        if clf.classes_[0] > 0:
            all_predicted = [i[0] for i in all_predicted]
        if clf.classes_[0] < 0:
            all_predicted = [1 -i[0] for i in all_predicted]

    # test model on labeled data
    y_test = labels_
    y_score = test_predicted

    precision, recall, threshold = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)

    print "AUC of " + feature + " Labeled Data Precision-Recall curve: ", average_precision
    for i in range(0,len(threshold)):
        print "Threshold", threshold[i]
        print "Precision of " + feature + " Labeled Data ", precision[i]
        print "Recall of " + feature + " Labeled Data", recall[i]
        print

    your_thold = raw_input("Enter The Threshold You Want to Use For Predicting Data (For example: 0.3): ")
    try:
        your_thold = float(your_thold)
    except:
        print "Enter A Float For Threshold"
        sys.exit()

    # predict all the data and union find the cluster -generate the dictionary res
    res = {}  # key : id, value : root_id
    for pair in all_pairs_list:
        id1 = int(pair[0])
        id2 = int(pair[1])
        if id1 not in res:
            # print "id1",id1
            res[id1] = id1
        if id2 not in res:
            # print "id2", id2
            res[id2] = id2

    print "len(res)",len(res)
    for i in range(0,len(all_pairs_list)):
        pair = all_pairs_list[i]
        id1 = int(pair[0])
        id2 = int(pair[1])
        predictV = all_predicted[i]
        if float(predictV) >= your_thold:
            root1 = find_root(id1,res)
            root2 = find_root(id2,res)
            if root1 != root2:
                res[root2] = root1

    with open(output1,"w") as f:
        f.write(a[0])
        for l in a[1:]:
            l = l.strip().split(",")
            cur_id = int(l[0])
            if cur_id in res:
                l[1] = "cluster" + str(res[cur_id])#l1 is class ID
            else:
                l[1] = "cluster" + str(cur_id)
            f.write(",".join(l))
            f.write("\n")

    with open(output2, "w") as f:
        f.write(b[0])
        for l in b[1:]:
            l = l.strip().split(",")
            cur_id = int(l[0])
            if cur_id in res:
                l[1] = "cluster" + str(res[cur_id])  # l1 is class ID
            else:
                l[1] = "cluster" + str(cur_id)
            f.write(",".join(l))
            f.write("\n")

