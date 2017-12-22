# Entity_Resolution

Entity_Resolution compares different feature selection methods and clustering models in entity linking problem.

## Usage

### Part 1, generate the pairwise features file.
Example Usage

$ python myCalSim.py ./dataset/test1.csv ./dataset/test2.csv -o ./test -f name

Full Command List
The full list of command line options is available with $ python myCalSim.py -h

#### 1,./dataset/test1.csv ./dataset/test2.csv
The input files should be in .csv format with column names as the header. The two files should have same schema(same column name and order). The first column should be record id(int) and the second column should be class id, which "class" means the record belong to. Id should be integers and any id from one file must be larger than all ids from another file.

#### 2, -o ./outputName
It will output a pairwise features file with the outputName. The row of the pairwise feature file will be a potential pair from the two different data sources and the features of the pair. The features are generated by string similarity functions.

#### 3, -f name
It is to clarify the column you want to use for jaccard blocking. Enter the column name of which the value you want to use for blocking. If none, the third column will be used for blocking.

### Part 2, compares different feature selection methods and clustering model for clustering.
Example Usage

$ python model_part.py ./dataset/test1.csv ./dataset/test2.csv ./test -o ./res -f Random -m RandomForest

Full Command List
The full list of command line options is available with $ python model_part.py -h

#### 1, ./dataset/test1.csv ./dataset/test2.csv
The same two files with part 1

#### 2,./test
The pairwise feature file from part 1.

#### 3, -o <outputFile>
OutputFile will contain Accuracy,Precision,Recall,F-measure for different thresholds of probability(being the matched pairs). Thresholds are from 0.0 to 1.0(0.1 as unit of measurement). Besides,the distribution of thresholds of probability in the output file will follow the distribution of thresholds of probability in predictions.

#### 4,-f <featureSelectionModel>
featureSelectionModel Option: Complete, Random, VarianceThreshold,SelectKBestFeatures,SelectFromModel_LassoCV. If none, Complete will be used.

#### 5, -m <clusterModel>
clusterModel Option: LogisticRegressionCV, RandomForest. If none, RandomForest model will be used.


## Requirements
1,numpy(>= 1.13.3)

2,sklearn(>= 0.19.1)

3,nltk(>= 3.2.5)
