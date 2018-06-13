# Jason Rich
# MSIM 607 Machine Learning
# Project 1
# KNN

getwd()
setwd('/Users/jasonrich/msim607-machine-learning/project1')
################################################################################
#install packages (if missing)
################################################################################
pkgs <- installed.packages()[,1]
pkgs.need <- c('ggplot2','caTools','tidyverse','class')
pkgs.missing <- pkgs.need[!pkgs.need %in% pkgs]
if (length(pkgs.missing) > 0){
    install.packages(pkgs.missing, dep = TRUE)
}

library(ggplot2) # plotting and graphic library
library(caTools) # Toolkit for data preprocessing in R
library(tidyverse) # Loads tidyverse, financial pkgs, used to get data
library(class) # knn functionality


# Generated traiing data
################################################################################

# read in the generated training data
knn.trn <- read.table('project1-submit/data/train.txt', sep = ' ')
colnames(knn.trn)<- c('ftr1', 'ftr2', 'rsp')

# create the feature dataframe
x.trn <- knn.trn[1:2]

# create the response dataframe
# ?factor
y.trn <- factor(knn.trn$rsp)


################################################################################
# Generated test data
################################################################################

# read in the generated training data
knn.tst <- read.table('project1-submit/data/test.txt', sep = ' ')
colnames(knn.tst)<- c('ftr1', 'ftr2', 'rsp')

# create the feature dataframe
x.tst <- knn.tst[1:2]

# create the response dataframe
y.tst <- factor(knn.tst$rsp)


################################################################################
# zip code training data
################################################################################

# read in zip code training data
zip.knn.trn <- read.csv('./project1-submit/data/Valid.ZC.train.Data.csv', header = FALSE, sep = ',')

colnames(zip.knn.trn) <- c('ftr1','ftr2','ftr3','ftr4','ftr5','ftr6','ftr7','ftr8','ftr9'
                          ,'ftr10','ftr11','ftr12','ftr13','ftr14','ftr15','ftr16','rsp')

# create the response dataframe
zip.y.trn <- factor(zip.knn.trn$rsp)

# create the feature dataframe
zip.x.trn <- zip.knn.trn[1:16]

################################################################################
# zip code test data
################################################################################

# read in zip code test data
zip.knn.tst <- read.table('./project1-submit/data/Valid.ZC.test.Data.csv', header =  FALSE, sep = ',')

colnames(zip.knn.tst) <- c('ftr1','ftr2','ftr3','ftr4','ftr5','ftr6','ftr7','ftr8','ftr9'
                          ,'ftr10','ftr11','ftr12','ftr13','ftr14','ftr15','ftr16','rsp')

# create the response dataframe
zip.y.tst <- factor(zip.knn.tst$rsp)

# create the feature dataframe
zip.x.tst <- zip.knn.tst[1:16]


################################################################################
# KNN classifier
################################################################################





################################################################################
# Task 5:
# For both the generated data and the zip code data, design a k-nearest
# neighbor classifier. Try 3 different k values at your choices. Keep in mind
# that k must be odd. Report classification accuracies on the testing data set.
# Turn in all of your codes and results.
################################################################################


# generating data
# set.seed to help R randomly break any ties
set.seed(1)
k <- seq(c(1,3,5))

# ?length
# knn prediction to apply to test response where k = 1
knn.pred.1 <- knn(x.trn, x.tst, y.trn, k = 1)

# cross-validation
table(knn.cv(x.tst, y.tst, k = 1))
#   0   1
# 192 208

# str(y.trn)
# length(x.trn)
# ?knn
# applying our training knn predictions to test response
table(knn.pred.1, y.tst)
#           y.tst
# knn.pred.1   0   1
#          0 170  33
#          1  30 167

# knn prediction to apply to test response where k = 3
knn.pred.3 <- knn(x.trn, x.tst, y.trn, k = 3, prob = TRUE)
# cross-validation
table(knn.cv(x.tst, y.tst, k = 3))
#   0   1
# 197 203


# applying our training knn predictions to test response
table(knn.pred.3, y.tst)
#             y.tst
# knn.pred.3   0   1
#          0 181  27
#          1  19 173



# knn prediction to apply to test response where k = 5
knn.pred.5 <- knn(x.trn, x.tst, y.trn, k = 5, prob = TRUE)
# cross-validation
table(knn.cv(x.tst, y.tst, k = 5))
#   0   1
# 200 200

# applying our training knn predictions to test response
table(knn.pred.5, y.tst)
#             y.tst
# knn.pred.5   0   1
#          0 179  31
#          1  21 169


# zip code data
# set.seed to help R randomly break any ties
set.seed(1)

# knn prediction to apply to test response where k = 1
zip.knn.pred.1 <- knn(zip.x.trn, zip.x.tst, zip.y.trn, k = 1, prob = TRUE)

# applying our training knn predictions to test response
table(zip.knn.pred.1, zip.y.tst)
#                zip.y.tst
# zip.knn.pred.1   1   2   3   4   5   6   7   8   9  10
#             1  271   0   2   2   4   1   8   0   0   1
#             2    0 292   4   1   1   0   0   3   0   0
#             3    2   1 216  18   1   4   3   2   4   2
#             4    3   1  47 248   3  27   0   9   3   7
#             5    2   2   4   4 262   0   4   9   2  11
#             6   16   0   7  19   0 260  10   1   2   8
#             7    4   1  12   1   0   4 273   0  13   0
#             8    0   3   0   2   5   2   0 268   0  10
#             9    2   0   5   3   2   1   2   0 269   4
#             10   0   0   3   2  22   1   0   8   7 257



# knn prediction to apply to test response where k = 3
zip.knn.pred.3 <- knn(zip.x.trn, zip.x.tst, zip.y.trn, k = 3, prob = TRUE)

# applying our training knn predictions to test response
table(zip.knn.pred.3, zip.y.tst)
#             zip.y.tst
# zip.knn.pred.3   1   2   3   4   5   6   7   8   9  10
#             1  276   0   5   2   5   1  11   0   0   2
#             2    0 294   5   6   1   1   0   5   0   0
#             3    2   0 213  10   1   3   1   3   5   0
#             4    7   1  48 247   3  22   1   8   4  12
#             5    1   2   3   4 262   1   4  10   1  11
#             6   12   0   3  22   1 265  14   1   2  11
#             7    2   1  15   3   1   6 266   0  18   1
#             8    0   2   1   3   6   1   0 269   0   4
#             9    0   0   3   1   1   0   3   0 264   4
#             10   0   0   4   2  19   0   0   4   6 255


# knn prediction to apply to test response where k = 5
zip.knn.pred.5 <- knn(zip.x.trn, zip.x.tst, zip.y.trn, k = 5, prob = TRUE)

# applying our training knn predictions to test response
table(zip.knn.pred.5, zip.y.tst)
#               zip.y.tst
#   zip.knn.pred.5   1   2   3   4   5   6   7   8   9  10
#               1  276   0   4   0   6   0   9   0   1   1
#               2    0 294   4   9   1   1   0   5   0   0
#               3    3   0 212   7   1   6   2   2   5   0
#               4    6   1  50 246   3  25   0   8   5  15
#               5    1   3   6   4 256   0   1  10   1   9
#               6   13   0   3  28   2 261  25   1   1  12
#               7    1   0  14   1   0   7 261   0  21   0
#               8    0   2   0   1   7   0   0 270   0   5
#               9    0   0   2   0   1   0   2   0 260   1
#               10   0   0   5   4  23   0   0   4   6 257

# error rate: estimate test MSE for KNN models
bayes.rule <- function(x1, x2){
    x1 + x1^2 + x2 + x2^2
}

sim.test <- data.frame(x.trn
                      ,logodds = bayes.rule(x.tst[1],x.tst[2]) + rnorm(400, 0, .5)
                      ,y = knn.tst$rep)

bayes.err <- mean(sim.test$y != bayes.rule(x.trn$ftr1,x.trn$ftr2))




################################################################################
# Task 6:
# For the generated data set, display the decision boundaries produced by each
# of the above classifiers, using the method we have discussed in class. Show
# them separately.
