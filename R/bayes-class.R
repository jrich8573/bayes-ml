# Jason Rich
# MSIM 607 Machine Learning
# Project 1
# Bayesian Classifier



getwd()
setwd('/Users/jasonrich/msim607-machine-learning/project1')
################################################################################
#install packages (if missing)
################################################################################
pkgs <- installed.packages()[,1]
pkgs.need <- c('ggplot2','caTools','tidyverse')
pkgs.missing <- pkgs.need[!pkgs.need %in% pkgs]
if (length(pkgs.missing) > 0){
    install.packages(pkgs.missing, dep = TRUE)
}

library(ggplot2) # plotting and graphic library
library(caTools) # Toolkit for data preprocessing in R
library(tidyverse) # Loads tidyverse, financial pkgs, used to get data

################################################################################
# Generated traiing data
################################################################################

# read in the generated training data
trn <-  read.table('project1-submit/data/train.txt', sep = ' ')
View(trn)


colnames(trn)<- c('ftr1', 'ftr2', 'rsp')
View(trn)

# create the feature dataframe
x.trn <- trn[,-3]
View(x.trn)
str(x.trn)

# q: are the features the same size?
# a: yes; 400
length(x.trn[,1]) # 400
length(x.trn[,2]) # 400

# create the response dataframe
y.trn <- trn[3]
View(y.trn)
str(y.trn)

# q: is the response vector the same size as the feature df size?
# a: yes; 400
length(y.trn[,1])


################################################################################
# Generated test data
################################################################################

# read in the generated training data
tst <-  read.table('project1-submit/data/test.txt', sep = ' ')
View(tst)


colnames(tst)<- c('ftr1', 'ftr2', 'rsp')
View(tst)

# create the feature dataframe
x.tst <- tst[,-3]
View(x.tst)
str(x.tst)

# q: are the features the same size?
# a: yes; 400
length(x.tst[,1]) # 400
length(x.tst[,2]) # 400

# create the response dataframe
y.tst <- tst[3]
View(y.tst)
str(y.tst)

# q: is the response vector the same size as the feature df size?
# a: yes; 400
length(y.tst[,1])

################################################################################
# plotting the traiing and test data
################################################################################
# Task 1:
# For the generated datasets, scatter plot the training and testing data sets.
# Show class 0 in red and class 1 in blue.

# training scatter plot
ggplot(trn, aes(x=ftr1, y=ftr2, color=rsp)) +
    geom_point(shape = 1) +
    # scale_colour_gradientn(colours=rainbow(2)) +
    scale_color_gradient(low = "#FF033E", high = "#00308F")

# test scatter plot
ggplot(tst, aes(x=ftr1, y=ftr2, color=rsp)) +
    geom_point(shape = 1) +
    # scale_colour_gradientn(colours=rainbow(2)) +
    scale_color_gradient(low = "#FF033E", high = "#00308F")


################################################################################
# Task 2:
# For both the generated data and the zip code data, design a Bayes
# classifier assuming that the data follows a Gaussian distribution. Estimate
# corresponding parameters from the training data (parametric estimation).
# Apply your Bayes classifier to the training and testing data sets,
# respectively. Report training and testing classification accuracies.
# (For the zip code dataset, if the inverse of any covariance  matrix does not
# exist (this can be done by checking if the determinant of the covariance
# matrix is zero), then add a small positive value to all the diagonal
# components and report the value you added in your project report).
################################################################################

# Bayes Classifier
# training:
bayes.rule <- function(x1, x2){
    x1 + x1^2 + x2 + x2^2
}


bayes <- data_frame(x1 = x.trn[,-2],
                    x2 = x.trn[,-1],
                    y = trn$rsp,
                    y.actual = bayes.rule(x1, x2) > .5)

bayes.err <- mean(bayes$y != bayes$y.actual)
bayes.err
# The Bayes error for the training generated data is
# 0.4325





# Bayes Classifier
# test
bayes.rule <- function(x1, x2){
    x1 + x1^2 + x2 + x2^2
}


bayes.test <- data_frame(x1 = x.tst[,-2],
                         x2 = x.tst[,-1],
                         y = tst$rsp,
                         y.actual = bayes.rule(x1, x2) > .5)

bayes.err <- mean(bayes.test$y != bayes.test$y.actual)
bayes.err
# The Bayes error for the test generated data is
# 0.4225






################################################################################
# Zip code training data
################################################################################

# ?read.csv
zip.trn <- read.csv('project1-submit/data/Valid_ZC_train_Data.csv', header = FALSE, sep = ',')
View(zip.trn)

colnames(zip.trn) <- c('ftr1','ftr2','ftr3','ftr4','ftr5','ftr6','ftr7','ftr8','ftr9'
                          ,'ftr10','ftr11','ftr12','ftr13','ftr14','ftr15','ftr16','rsp')


zip.x.trn <- zip.trn[, -17]
zip.y.trn <- zip.trn[17]

################################################################################
# Zip code test data
################################################################################
zip.tst <- read.csv('project1-submit/data/Valid_ZC_test_Data.csv', header = FALSE, sep = ',')
# View(zip.tst)

colnames(zip.tst) <- c('ftr1','ftr2','ftr3','ftr4','ftr5','ftr6','ftr7','ftr8','ftr9'
                          ,'ftr10','ftr11','ftr12','ftr13','ftr14','ftr15','ftr16','rsp')


zip.x.tst <- zip.tst[, -17]
zip.y.trn <- zip.tst[17]


bayes_rule <- function(x1,x2,x3,x4,x5,x6,x7,x8,x9
                       ,x10,x11,x12,x13,x14,x15,x16){
      x1 + x1^2
    + x2 + x2^2
    + x3 + x3^2
    + x4 + x4^2
    + x5 + x5^2
    + x6 + x6^2
    + x7 + x7^2
    + x8 + x8^2
    + x9 + x9^2
    + x10 + x10^2
    + x11 + x11^2
    + x12 + x12^2
    + x13 + x13^2
    + x14 + x14^2
    + x15 + x15^2
    + x16 + x16^2
}



bayes.zip <- data_frame(x1 = zip.x.trn$ftr1,
                        x2 = zip.x.trn$ftr2,
                        x3 = zip.x.trn$ftr3,
                        x4 = zip.x.trn$ftr4,
                        x5 = zip.x.trn$ftr5,
                        x6 = zip.x.trn$ftr6,
                        x7 = zip.x.trn$ftr7,
                        x8 = zip.x.trn$ftr8,
                        x9 = zip.x.trn$ftr9,
                        x10 = zip.x.trn$ftr10,
                        x11 = zip.x.trn$ftr11,
                        x12 = zip.x.trn$ftr12,
                        x13 = zip.x.trn$ftr13,
                        x14 = zip.x.trn$ftr14,
                        x15 = zip.x.trn$ftr15,
                        x16 = zip.x.trn$ftr16,
                        logodds = bayes_rule(x1,
                                             x2,
                                             x3,
                                             x4,
                                             x5,
                                             x6,
                                             x7,
                                             x8,
                                             x9,
                                             x10,
                                             x11,
                                             x12,
                                             x13,
                                             x14,
                                             x15,
                                             x16) + rnorm(200, 0, .5),
                        y = zip.trn$rsp,
                        y.actual = bayes_rule(x1,
                                             x2,
                                             x3,
                                             x4,
                                             x5,
                                             x6,
                                             x7,
                                             x8,
                                             x9,
                                             x10,
                                             x11,
                                             x12,
                                             x13,
                                             x14,
                                             x15,
                                             x16) > .5)
bayes.err <- mean(bayes.zip$y != bayes.zip$y.actual)
bayes.err
# The Bayes error for the zip training data is
# 0.9


# check the determinant of the covariance matrix is not zero
det(cov(bayes.zip))
# 0.1503716


bayes.zip.tst <- data_frame(x1 = zip.x.tst$ftr1,
                        x2 = zip.x.tst$ftr2,
                        x3 = zip.x.tst$ftr3,
                        x4 = zip.x.tst$ftr4,
                        x5 = zip.x.tst$ftr5,
                        x6 = zip.x.tst$ftr6,
                        x7 = zip.x.tst$ftr7,
                        x8 = zip.x.tst$ftr8,
                        x9 = zip.x.tst$ftr9,
                        x10 = zip.x.tst$ftr10,
                        x11 = zip.x.tst$ftr11,
                        x12 = zip.x.tst$ftr12,
                        x13 = zip.x.tst$ftr13,
                        x14 = zip.x.tst$ftr14,
                        x15 = zip.x.tst$ftr15,
                        x16 = zip.x.tst$ftr16,
                        logodds = bayes_rule(x1,
                                             x2,
                                             x3,
                                             x4,
                                             x5,
                                             x6,
                                             x7,
                                             x8,
                                             x9,
                                             x10,
                                             x11,
                                             x12,
                                             x13,
                                             x14,
                                             x15,
                                             x16) + rnorm(200, 0, .5),
                        y = zip.trn$rsp,
                        y.actual = bayes_rule(x1,
                                             x2,
                                             x3,
                                             x4,
                                             x5,
                                             x6,
                                             x7,
                                             x8,
                                             x9,
                                             x10,
                                             x11,
                                             x12,
                                             x13,
                                             x14,
                                             x15,
                                             x16) > .5)

bayes.err.tst <- mean(bayes.zip.tst$y != bayes.zip.tst$y.actual)
bayes.err
# The Bayes error for the zip test data is
# 0.9

# check the determinant of the covariance matrix is not zero
det(cov(bayes.zip.tst))bayes.zip.tst
# 0.5384546
