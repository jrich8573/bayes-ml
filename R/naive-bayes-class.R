# Jason Rich
# Naive Bayes


getwd()
setwd('/Users/jasonrich/msim607-machine-learning/project1/project1-submit/')
################################################################################
#install packages (if missing)
################################################################################

pkgs <- installed.packages()[,1]
pkgs.need <- c('ggplot2','caTools','tidyverse','data.table')
pkgs.missing <- pkgs.need[!pkgs.need %in% pkgs]
if (length(pkgs.missing) > 0){
    install.packages(pkgs.missing, dep = TRUE)
}

library(ggplot2) # plotting and graphic library
library(caTools) # Toolkit for data preprocessing in R
library(tidyverse) # Loads tidyverse, financial pkgs, used to get data
library(data.table) # Used for naive bayes classifier
################################################################################
# Generated traiing data
################################################################################

# read in the generated training data
nb.trn <- data.table(read.table('project1-submit/data/train.txt', sep = ' '))
colnames(nb.trn)<- c('ftr1', 'ftr2', 'rsp')

# create the feature dataframe
x.trn <- data.frame(nb.trn[,1:2])

# create the response dataframe
y.trn <- data.frame(nb.trn[,3])

################################################################################
# Generated test data
################################################################################

# read in the generated training data
nb.tst <- data.table(read.table('project1-submit/data/test.txt', sep = ' '))
colnames(nb.tst)<- c('ftr1', 'ftr2', 'rsp')

# create the feature dataframe
x.tst <- data.frame(nb.tst[,1:2])

# create the response dataframe
y.tst <- data.frame(nb.tst[,3])


################################################################################
# zip code training data
################################################################################

# read in zip code training data
zip.nb.trn <- data.table(read.csv('./project1-submit/data/Valid_ZC_train_Data.csv', header = FALSE, sep = ','))

colnames(zip.nb.trn) <- c('ftr1','ftr2','ftr3','ftr4','ftr5','ftr6','ftr7','ftr8','ftr9'
                          ,'ftr10','ftr11','ftr12','ftr13','ftr14','ftr15','ftr16','rsp')

# create the response dataframe
zip.y.trn <- data.frame(zip.nb.trn[,17])

# create the feature dataframe
zip.x.trn <- data.frame(zip.nb.trn[,1:16])

################################################################################
# zip code test data
################################################################################

# read in zip code test data
zip.nb.tst <- data.table(read.table('./project1-submit/data/Valid_ZC_test_Data.csv', header =  FALSE, sep = ','))

colnames(zip.nb.tst) <- c('ftr1','ftr2','ftr3','ftr4','ftr5','ftr6','ftr7','ftr8','ftr9'
                          ,'ftr10','ftr11','ftr12','ftr13','ftr14','ftr15','ftr16','rsp')

# create the response dataframe
zip.y.tst <- data.frame(zip.nb.tst[,17])

# create the feature dataframe
zip.x.tst <- data.frame(zip.nb.tst[,1:16])

################################################################################
# Task 3:
# Repeat Task 2 by designing a naive Bayes classifier. (A simple way to design a
# naïve Bayes classifier is to make those off-diagonal elements in the estimated
# covariance matrix zero.)
################################################################################

# naive bayes classifier

train <- function(dt){
    return(dt[,list(meanftr1 = mean(ftr1), varftr1 = sd(ftr1),
                    meanftr2 = mean(ftr2), varftr2 = sd(ftr2)), by = rsp])
}

classify <- function (classifier, sample){
    posterior <- function (sample, class_prior, class){
        p.ftr1 <- dnorm(sample$ftr1, class$meanftr1, class$varftr1)
        p.ftr2 <- dnorm(sample$ftr2, class$meanftr2, class$varftr2)
        return(class_prior * p.ftr1* p.ftr2)
    }
    class.0 <- classifier[which(classifier$rsp == 0),]
    class.1 <- classifier[which(classifier$rsp == 1),]

    prior.0 <- 0.5
    prior.1 <- 0.5

    return(list(rsp.0 = posterior(sample, prior.0, class.0),
                rsp.1 = posterior(sample, prior.1, class.1)))
}


# training data
sample <- data.table(ftr1 = 3, ftr2 = 6)
classifier <- train(nb.trn)
# classifier
# rsp  meanftr1   varftr1  meanftr2  varftr2
#  0  0.1204424 0.9499342 0.1360337 2.891274
#  1  2.0439203 0.9515809 1.9988650 2.962921

result = classify(classifier, sample)
cat('posterior(class 0) =', result$rsp.0)
# 3.744878e-05
cat('posterior(class 1) =', result$rsp.1)
# 0.006845928

# from the EDA above
cov(x.trn, y.trn)
#         rsp
# ftr1 0.4820747
# ftr2 0.4668750


# test data
sample <- data.table(ftr1 = 3, ftr2 = 6)
classifier <- train(nb.tst)
# classifier
# rsp   meanftr1   varftr1   meanftr2  varftr2
#  0  0.02468252 0.9043062 -0.1360337 3.106214
#  1  1.98174622 0.9994003  2.0011351 3.044064

result = classify(classifier, sample)
cat('posterior(class 0) =', result$rsp.0)
# 1.795661e-05
cat('posterior(class 1) =', result$rsp.1)
# 0.006568224

# from the EDA above
cov(x.tst, y.tst)
#          rsp
# ftr1 0.4904922
# ftr2 0.5356313

################################################################################
# Zip code Naive Bayes
################################################################################

train <- function(dt){
    return(dt[,list(meanftr1 = mean(ftr1), varftr1 = sd(ftr1),
                    meanftr2 = mean(ftr2), varftr2 = sd(ftr2),
                    meanftr3 = mean(ftr3), varftr3 = sd(ftr3),
                    meanftr4 = mean(ftr4), varftr4 = sd(ftr4),
                    meanftr5 = mean(ftr5), varftr5 = sd(ftr5),
                    meanftr6 = mean(ftr6), varftr6 = sd(ftr6),
                    meanftr7 = mean(ftr7), varftr7 = sd(ftr7),
                    meanftr8 = mean(ftr8), varftr8 = sd(ftr8),
                    meanftr9 = mean(ftr9), varftr9 = sd(ftr9),
                    meanftr10 = mean(ftr10), varftr10 = sd(ftr10),
                    meanftr11 = mean(ftr11), varftr11 = sd(ftr11),
                    meanftr12 = mean(ftr12), varftr12 = sd(ftr12),
                    meanftr13 = mean(ftr13), varftr13 = sd(ftr13),
                    meanftr14 = mean(ftr14), varftr14 = sd(ftr14),
                    meanftr15 = mean(ftr15), varftr15 = sd(ftr15),
                    meanftr16 = mean(ftr16), varftr16 = sd(ftr16)), by = rsp])
}

classify <- function (classifier, sample){
    posterior <- function (sample, class_prior, class){
        p.ftr1 <- dnorm(sample$ftr1, class$meanftr1, class$varftr1)
        p.ftr2 <- dnorm(sample$ftr2, class$meanftr2, class$varftr2)
        p.ftr3 <- dnorm(sample$ftr3, class$meanftr3, class$varftr3)
        p.ftr4 <- dnorm(sample$ftr4, class$meanftr4, class$varftr4)
        p.ftr5 <- dnorm(sample$ftr5, class$meanftr5, class$varftr5)
        p.ftr6 <- dnorm(sample$ftr6, class$meanftr6, class$varftr6)
        p.ftr7 <- dnorm(sample$ftr7, class$meanftr7, class$varftr7)
        p.ftr8 <- dnorm(sample$ftr8, class$meanftr8, class$varftr8)
        p.ftr9 <- dnorm(sample$ftr9, class$meanftr9, class$varftr9)
        p.ftr10 <- dnorm(sample$ftr10, class$meanftr10, class$varftr10)
        p.ftr11 <- dnorm(sample$ftr11, class$meanftr11, class$varftr11)
        p.ftr12 <- dnorm(sample$ftr12, class$meanftr12, class$varftr12)
        p.ftr13 <- dnorm(sample$ftr13, class$meanftr13, class$varftr13)
        p.ftr14 <- dnorm(sample$ftr14, class$meanftr14, class$varftr14)
        p.ftr15 <- dnorm(sample$ftr15, class$meanftr15, class$varftr15)
        p.ftr16 <- dnorm(sample$ftr16, class$meanftr16, class$varftr16)
        return(class_prior * p.ftr1* p.ftr2* p.ftr3* p.ftr4* p.ftr5* p.ftr6* p.ftr7*
                   p.ftr8* p.ftr9* p.ftr10* p.ftr11* p.ftr12* p.ftr13* p.ftr14* p.ftr15*
                   p.ftr16)
    }
    class.1 <- classifier[which(classifier$rsp == 1),]
    class.2 <- classifier[which(classifier$rsp == 2),]
    class.3 <- classifier[which(classifier$rsp == 3),]
    class.4 <- classifier[which(classifier$rsp == 4),]
    class.5 <- classifier[which(classifier$rsp == 5),]
    class.6 <- classifier[which(classifier$rsp == 6),]
    class.7 <- classifier[which(classifier$rsp == 7),]
    class.8 <- classifier[which(classifier$rsp == 8),]
    class.9 <- classifier[which(classifier$rsp == 9),]
    class.10 <- classifier[which(classifier$rsp == 10),]
    class.11 <- classifier[which(classifier$rsp == 11),]
    class.12 <- classifier[which(classifier$rsp == 12),]
    class.13 <- classifier[which(classifier$rsp == 13),]
    class.14 <- classifier[which(classifier$rsp == 14),]
    class.15 <- classifier[which(classifier$rsp == 15),]
    class.16 <- classifier[which(classifier$rsp == 16),]

    prior.1 <- 0.5
    prior.2 <- 0.5
    prior.3 <- 0.5
    prior.4 <- 0.5
    prior.5 <- 0.5
    prior.6 <- 0.5
    prior.7 <- 0.5
    prior.8 <- 0.5
    prior.9 <- 0.5
    prior.10 <- 0.5
    prior.11 <- 0.5
    prior.12 <- 0.5
    prior.13 <- 0.5
    prior.14 <- 0.5
    prior.15 <- 0.5
    prior.16 <- 0.5

    return(list(rsp.1 = posterior(sample, prior.1, class.1),
                rsp.2 = posterior(sample, prior.2, class.2),
                rsp.3 = posterior(sample, prior.3, class.3),
                rsp.4 = posterior(sample, prior.4, class.4),
                rsp.5 = posterior(sample, prior.5, class.5),
                rsp.6 = posterior(sample, prior.6, class.6),
                rsp.7 = posterior(sample, prior.7, class.7),
                rsp.8 = posterior(sample, prior.8, class.8),
                rsp.9 = posterior(sample, prior.9, class.9),
                rsp.10 = posterior(sample, prior.10, class.10),
                rsp.11 = posterior(sample, prior.11, class.11),
                rsp.12 = posterior(sample, prior.12, class.12),
                rsp.13 = posterior(sample, prior.13, class.13),
                rsp.14 = posterior(sample, prior.14, class.14),
                rsp.15 = posterior(sample, prior.15, class.15),
                rsp.16 = posterior(sample, prior.16, class.16)))
}


# training data
sample <- data.table(ftr1= 11 , ftr2= 0.1 , ftr3= 0.33  , ftr4= 13  , ftr5=5 , ftr6=9 , ftr7=4,
                     ftr8=1 , ftr9=3 , ftr10=0.6  , ftr11=0.3 , ftr12=0  , ftr13=0.9  ,
                     ftr14= 7 , ftr15=1.92, ftr16=0.008)

classifier <- train(zip.nb.trn)
# classifier

result = classify(classifier, sample)
cat('posterior(class 1) =', result$rsp.1)
cat('posterior(class 2) =', result$rsp.2)
cat('posterior(class3) =',  result$rsp.3)
cat('posterior(class4) =',  result$rsp.4)
cat('posterior(class5) =',  result$rsp.5)
cat('posterior(class6) =',  result$rsp.6)
cat('posterior(class7) =',  result$rsp.7)
cat('posterior(class8) =',  result$rsp.8)
cat('posterior(class9) =',  result$rsp.9)
cat('posterior(class10) =', result$rsp.10)
cat('posterior(class11) =', result$rsp.11)
cat('posterior(class12) =', result$rsp.12)
cat('posterior(class13) =', result$rsp.13)
cat('posterior(class14) =', result$rsp.14)
cat('posterior(class15) =', result$rsp.15)
cat('posterior(class16) =', result$rsp.16)


# from the EDA above
cov(zip.x.trn, zip.y.trn)
#        rsp
# ftr1  -0.096865622
# ftr2   0.055685228
# ftr3   0.123207736
# ftr4   0.287762588
# ftr5  -0.322607536
# ftr6   0.009336445
# ftr7  -0.231743915
# ftr8   0.652550850
# ftr9   1.497415805
# ftr10  7.013671224
# ftr11  0.359936024
# ftr12 -4.637212404
# ftr13 -3.018475495
# ftr14  0.972782771
# ftr15  1.141931963
# ftr16 -0.363056268




# test data
sample <- data.table(ftr1= 3 , ftr2= 3 , ftr3= 3 , ftr4= 3 , ftr5= 6 , ftr6= 6 , ftr7= 6,
                     ftr8= 6 , ftr9=9  , ftr10= 9 , ftr11= 9 , ftr12= 9 , ftr13= 9 ,
                     ftr14= 1 , ftr15= 1 , ftr16= 1 )

classifier <- train(zip.nb.tst)
# classifier

result = classify(classifier, sample)
cat('posterior(class 1) =', result$rsp.1)
cat('posterior(class 2) =', result$rsp.2)
cat('posterior(class3) =',  result$rsp.3)
cat('posterior(class4) =',  result$rsp.4)
cat('posterior(class5) =',  result$rsp.5)
cat('posterior(class6) =',  result$rsp.6)
cat('posterior(class7) =',  result$rsp.7)
cat('posterior(class8) =',  result$rsp.8)
cat('posterior(class9) =',  result$rsp.9)
cat('posterior(class10) =', result$rsp.10)
cat('posterior(class11) =', result$rsp.11)
cat('posterior(class12) =', result$rsp.12)
cat('posterior(class13) =', result$rsp.13)
cat('posterior(class14) =', result$rsp.14)
cat('posterior(class15) =', result$rsp.15)
cat('posterior(class16) =', result$rsp.16)


# from the eda above
cov(zip.x.tst, zip.y.tst)
#        rsp
# ftr1  -0.11170390
# ftr2   0.08369456
# ftr3   0.10353451
# ftr4   0.31993998
# ftr5  -0.31543848
# ftr6   0.01750584
# ftr7  -0.20773591
# ftr8   0.62970990
# ftr9   1.44223074
# ftr10  6.77550850
# ftr11  0.31163221
# ftr12 -5.03901300
# ftr13 -2.71075225
# ftr14  0.95339019
# ftr15  1.16139594
# ftr16 -0.44447403
