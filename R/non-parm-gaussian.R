# Jason Rich
# On Parameteric Gaussian Kernel

getwd()
setwd('/Users/jasonrich/msim607-machine-learning/project1/project1-submit/')
################################################################################
#install packages (if missing)
################################################################################

pkgs <- installed.packages()[,1]
pkgs.need <- c('ggplot2','caTools','tidyverse','data.table', 'plyr')
pkgs.missing <- pkgs.need[!pkgs.need %in% pkgs]
if (length(pkgs.missing) > 0){
    install.packages(pkgs.missing, dep = TRUE)
}

library(ggplot2) # plotting and graphic library
library(caTools) # Toolkit for data preprocessing in R
library(tidyverse) # Loads tidyverse, financial pkgs, used to get data
library(data.table) # Used for naive bayes classifier
library(plyr)

################################################################################
# Task 4:
# Repeat Task 2. Utilize a nonparametric estimation technique to estimate the
# conditional distribution p(x|Ci), using a Gaussian kernel. Try 3 different
# kernel sizes h at your choices. Report classification accuracies on the testing
# data set.
################################################################################
View(trn)
no.prm.kern <- function(){
    df <- trn

    # show a couple of random images from df

    train <- df
    test <- tst


    model <- fit.classifier(train)
    # plot.model(model)
    # compute the posterior class distribution for each test point
     test.post <- adply(test, 1,function (x) classify(as.data.frame(as.double(x$ftr1)), model))

    # (optional)
    #train.post <- adply(train, 1, function (x) classify(as.double(x), model))

    # form the arg.max classes
    best <- apply(test.post, 1, function (x) names(x[1:2])[which.max(x)])
    test.post$best <- best
    # (optional)
    best <- apply(train.post, 1, function (x) names(x[1:2])[which.max(x)])
    train.post$best <- best

    # show a couple of these: test.post[342, c("class", "best")]

    # make the confusion table
    conf <- table(test.post[,c("rsp", "best")])
    conf <- aaply(conf, 1, function (x) x / sum(x))

    # compute the accuracy
    acc <- sum(diag(conf)) / sum(conf)

    plot.confusion(conf)
}



# fit a gaussian generative classifier

fit.classifier <- function(df){
    # fit the means and variances

    #df <- trn
    class.mean <-
        dlply(df, c("rsp"),
              function (x) sapply(x[,c("ftr1", "ftr2")], 2, FUN = mean))

    class.sd <-
        dlply(df, c("rsp"),
              function (x) sapply(x[,c("ftr1", "ftr2")], 2, FUN = sd))

    # fit the proportions

    class.prop <- table(df[,"rsp"])
    class.prop <- class.prop / sum(class.prop)

    list(class.mean=class.mean, class.sd=class.sd, class.prop=class.prop)
}


classify <- function(rgb, model){
    classes <- names(model$class.prop)
    post <-
        sapply(classes,
               function (class) {
                   log(model[["class.prop"]][class]) +
                       dnorm(rgb[0],mean=model[["class.mean"]][[class]][0],
                             sd=model[["class.sd"]][[class]][0],log=T) +
                       dnorm(rgb[1], mean=model[["class.mean"]][[class]][1],
                             sd=model[["class.sd"]][[class]][1], log=T)})
                       #dnorm(rgb[3],mean=model[["class.mean"]][[class]][3],
                    #         sd=model[["class.sd"]][[class]][3], log=T) })

    #post <- post - log.sum(post)
    #names(post) <- classes

    post
}


# given log(v), returns log(sum(v))

# log.sum <- function(v){
#
#     log.sum.pair <- function(x,y){
#         return(x+log(1 + exp(x)))
#         #else return(y+log(1 + exp(x-y)));
#     }
#     if (length(v) == 1) return(v)
#     r <- v[1];
#     for (i in 2:length(v))
#         r <- log.sum.pair(r, v[i])
#     return(r)
#}

# plot a confusion matrix

plot.confusion <- function(conf){
    melted <- melt(conf, c("rsp", "best"))
    p <- ggplot(data=melted, aes(x=best, y=class, size=value))
    p <- p + geom_point(shape=15)
    p <- p + scale_area(to=c(1,20))
    p <- p + theme_bw() + opts(axis.text.x = theme_text(angle=90))
    p
}

# plot model

plot.model <- function(model){
    melted <- melt(ldply(model[["class.mean"]], function (x) x))
    p <- ggplot(data=melted)
    p <- p + geom_point(aes(x=variable, y=value))
    p <- p + facet_grid(~ class)
    p <- p + theme_bw() + opts(axis.text.x = theme_text(angle=90))
    p
}


no.prm.kern()
