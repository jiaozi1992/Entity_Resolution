from sklearn.model_selection import train_test_split
from random import *
from sklearn.linear_model import LogisticRegressionCV
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
    print '<program.py> <file1> <file2> <PairwiseFeatureFile> -o <outputFile> -f <featureSelectionModel> -m <clusterModel>'
    print "featureSelectionModel Option: Complete, Random, VarianceThreshold,SelectKBestFeatures,SelectFromModel_LassoCV. If none, Complete will be used"
    print "clusterModel Option: LogisticRegressionCV, RandomForest. If none, RandomForest model will be used"

if __name__ == "__main__":
    if (len(sys.argv[1:]) == 0 or sys.argv[1] == "-h"):
        usage()
        sys.exit(2)

    test_argv = sys.argv[1:]
    if len(test_argv) < 3:
        usage()
        sys.exit(2)

    try:
        a = open(sys.argv[1])
    except IOError:
        print "Could Not Read First File"
        sys.exit(2)

    try:
        b = open(sys.argv[2])
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
        opts, args = getopt.getopt(argv, "o:f:m:", ["outputFile=","featureSelectionModel=","clusterModel="])

    except getopt.GetoptError:
        usage()
        sys.exit(2)

    if len(argv) < 2:
        usage()
        sys.exit()

    output = ""
    for opt, arg in opts:
        if opt in ("-o", "--outputFile"):
            output = arg

        elif opt in ("-f", "--featureSelectionModel"):
            feature = arg
            if feature not in ["Complete", "Random", "VarianceThreshold","SelectKBestFeatures","SelectFromModel_LassoCV"]:
                print "Wrong Feature Selection Model,Please Follow:"
                usage()
                sys.exit()
        elif opt in ("-m", "--clusterModel"):
            clusterModel = arg
            if clusterModel not in ["LogisticRegressionCV","RandomForest"]:
                print "Wrong Cluster Model, Please Follow:"
                usage()
                sys.exit()

    if output == "":
        print "No Output File"
        sys.exit(2)

    try:
        output = output + '.csv'
        open(output,"w")
    except IOError:
        print "Could Not Open Output File"
        sys.exit(2)

    print "It may take several minutes to finish this part(usually less than 5 minutes if you have 4 million pairs between first and second file)"
    a_ = []
    b_ = []
    all_pairs = {}
    all_pairs_set = set()
    for al in a:
        al = al.strip().split(",")
        a_.append(al)

    for bl in b:
        bl = bl.strip().split(",")
        b_.append(bl)

    a_ = a_[1:]
    b_ = b_[1:]
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
    class_w_vec_dict = {}
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
        else:
            tnm_set.add(pair)
        w_vec_dict[pair] = [float(i) for i in l[2:]]
        class_w_vec_dict[pair] = [float(i) for i in l[3:]]

    pairNum2 = len(field_names_list)-1
    if float(all_tp) > 0:
        pc = min(1.0,len(tm_set)/float(all_tp))
    else:
        pc = 0
    print "Pairs Completeness:", pc
    print "Reduction Ratio:",1 - (len(tm_set) + len(tnm_set))/float(len(all_pairs))

    blokd_pairs = {}  # remain after blocking
    blokd_pairs_set = set()

    for k in w_vec_dict:
        blokd_pairs[k] = 1
        blokd_pairs_set.add(k)

    assert len(blokd_pairs_set) == len(tm_set) + len(tnm_set)
    assert len(blokd_pairs) == len(tm_set) + len(tnm_set)

    # unbloked test+preds
    unbloked_test = []
    unbloked_preds = []
    unbloked_set = set()
    for i in all_pairs_set:
        if (i[0], i[1]) not in blokd_pairs and (i[1], i[0]) not in blokd_pairs:
            unbloked_test.append(all_pairs[i])
            unbloked_preds.append(-1)

    assert len(unbloked_test) + len(blokd_pairs) == len(all_pairs_set)

    data = []
    labels = []
    ids = []

    for (rec_id_tuple, w_vec) in class_w_vec_dict.iteritems():
        if w_vec[0] > -0.5:
            data.append(w_vec)
            ids.append(rec_id_tuple)
            if (rec_id_tuple in tm_set):
                labels.append(1.0)  # Match class
            else:
                labels.append(-1.0)  # Non-match class

    assert len(data) == len(tm_set) + len(tnm_set)
    assert len(data) == len(labels)
    assert len(data) == len(w_vec_dict)

    #balance data
    # data_ is numpy array of all blocking data(as sklearn only accept numpy)
    data_ = array(data)
    labels_ = array(labels)

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

    random_num = randint(1, 1000000)
    train_size = raw_input("Enter The Percentage You Want to Use For Training Data (For example: 0.5 for this small dataset): ")
    try:
        train_size = float(train_size)
    except:
        print "Enter a Float Number for Training Data Percentage"
        sys.exit()
    X_train2, X_test2, y_train2, y_test2 = train_test_split(data2_, labels2_, test_size=1-train_size, random_state=random_num)
    print "Training Dataset Size: ",len(X_train2)

    if feature == "Complete":  # for diagram title
        data_ = data_
        X_train2 = X_train2

    if feature == "Random":
        featureForColumn = int(raw_input("How Many features do you want for each column(If exceed the total number of features for a column, the total number of features for the column will be used): "))
        if featureForColumn <= 0:
            print "Features do you want for each column must be more than 0"
            sys.exit()
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

    if feature == "SelectKBestFeatures":
        kv = int(raw_input("Enter Number of Features: "))
        SK = SelectKBest(k=kv)
        SK.fit(X_train2, y_train2)
        data_ = SK.transform(data_)
        X_train2 = SK.transform(X_train2)

    if feature == "SelectFromModel_LassoCV":
        max_iterNum = int(raw_input("Enter Max Interation Number: "))
        ls = LassoCV(max_iter=max_iterNum)
        sfm = SelectFromModel(ls, threshold=0.25)
        sfm.fit(X_train2, y_train2)
        data_ = sfm.transform(data_)
        X_train2 = sfm.transform(X_train2)

    if clusterModel == "LogisticRegressionCV":
        clf = LogisticRegressionCV().fit(X_train2, y_train2)
        # all_predicted is the predicts of all pairs in blocking file
        # all_predicted = [[probability of class 1,probability of class 0]....]
        all_predicted = clf.predict_proba(data_)


    if clusterModel == "RandomForest":
        maxDepth = int(raw_input("Enter maxDepth for Random Forest Model: "))
        nEstimators = int(raw_input("Enter the number of trees in the forest: "))
        clf = RandomForestClassifier(n_estimators = nEstimators,max_depth=100, random_state=2000).fit(X_train2, y_train2)
        all_predicted = clf.predict_proba(data_)

    # make all_predicted = [probability of class 1]
    if len(clf.classes_) > 1:
        if clf.classes_[1] > 0:
            all_predicted = [i[1] for i in all_predicted]
            # assert i[0] + i[1] == 1
        else:
            all_predicted = [i[0] for i in all_predicted]
            # assert i[0] + i[1] == 1

    if len(clf.classes_) <= 1:
        if clf.classes_[0] > 0:
            all_predicted = [i[0] for i in all_predicted]
        if clf.classes_[0] < 0:
            all_predicted = [1 -i[0] for i in all_predicted]

    y_test = labels + unbloked_test
    y_score = all_predicted + unbloked_preds


    precision, recall, threshold1 = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)

    print "AUC of " + feature + " Precision-Recall curve: ", average_precision

    threshold2 = set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    threshold1 = set(threshold1)
    threshold3 = threshold2.union(threshold1)
    threshold3 = sorted(list(threshold3))
    threshold3 = [i for i in threshold3 if i >= 0]


    csv_matrix = []
    for threshold in threshold3:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(0, len(all_predicted)):
            if (all_predicted[i] >= threshold):  # Match prediction
                pred_match = True
                # print "all_predicted[i]",all_predicted[i]
            else:
                pred_match = False

            if (pred_match == True):
                if (int(labels[i]) > 0):
                    tp += 1
                else:
                    fp += 1
            else:  # Non-match prediction
                if (int(labels[i]) <= 0):
                    tn += 1
                else:
                    fn += 1

        if tp + fp + tn + fn <= 0:
            acc = 0
        else:
            acc = float(tp + tn) / (tp + fp + tn + fn)

        if tp + fp <= 0:
            prec = 0
        else:
            prec = tp / float((tp + fp))

        if tp + fn <= 0:
            recall = 0
        else:
            recall = tp / float(all_tp)
            # recall = min(1.0, tp / float(len(tm_set))

        if prec + recall <= 0:
            fm = 0
        else:
            fm = 2 * prec * recall / (prec + recall)

        cur = [threshold,acc,recall,fm]
        csv_matrix.append(cur)

    with open(output, "w") as f1:
        f1.write("Threshold,Accuracy,Precision,F-measure")
        f1.write("\n")
        for i in csv_matrix:
            for j in i:
                f1.write(str(j))
                f1.write(",")
            f1.write("\n")
