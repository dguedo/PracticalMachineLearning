# Practical Machine Learning: Prediction Assignment Writeup

## Overview
This is the course project for the Practical Machine Learning.

The goal of the project was to predict the manner in which a group of subjects performed exercises, these were denoted by the "classe" variable. 

The following sections describe my methods for determining the final model.

## Libraries

Load the necessary libraries and set the seed


```r
library(RCurl)
library(caret)
library(randomForest)

set.seed(3510)
```

## Securing the Data and Preliminary Analyses

**1. Load and assign the datasets**


```r
# set the working directory
setwd("~/GitHub/PracticalMachineLearning")

# Download the training and test sets
trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainingURL ,destfile="pml-training.csv",method="libcurl")
download.file(testingURL ,destfile="pml-testing.csv",method="libcurl")

# Read in and assign the datasets 
trainingData  = read.csv("pml-training.csv")
testingSet    = read.csv("pml-testing.csv")
```

**2. Remove un-needed columns **

After profiling the training dataset using str(), Summary(), and head().  It became obvious that the first seven columns contained no data that would be relevant in the prediction model.  In fact, their inclusion might cause unintended consequences. 


```r
columnsToRemove <- c(1,2,3,4,5,6,7)
names(trainingData[, columnsToRemove])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```


```r
# Remove the columns
trainingData <- trainingData[, -columnsToRemove]
testingSet <- testingSet[, -columnsToRemove]
```

## Partition the datasets

Create the training and test set partitions


```r
inTrain   <- createDataPartition(trainingData$classe, p = 0.70, list = FALSE)
training  <- trainingData[inTrain, ]
testing   <- trainingData[-inTrain, ]
```

## Cleaning the data

**1. Remove the near zero values**

I used nearZeroVar to remove near zero values.  In general terms, nearZeroVar detects variables that have only one unique value (i.e. are zero variance predictors) or predictors that are have very few unique values relative to the number of samples. 


```r
nearZero  <- nearZeroVar(training)
training  <- training[, -nearZero]
testing   <- testing[, -nearZero]
```

**2. Remove mostly NA varibales**

After looking at the data many of the columns contained high levels of NA's.  I removed any variable that value NA accounted for more than 90% of the observations.


```r
training  <- training[,colSums(is.na(training)) < nrow(training)*0.9]
testing   <- testing[,colSums(is.na(testing)) < nrow(testing)*0.9]
```

## Model Building

I applied two models to this classification problem, those being boosting and random forest.

**1. Boosted tree model**

By default, simple bootstrap resampling is used, instead I opted for K-fold cross-validation.


```r
# K-fold cross-validation (5 fold)
fitControl  <- trainControl(method = "cv", number = 5)
modelGbm    <- train(classe ~ .,data=training, method="gbm", trControl = fitControl, verbose = FALSE)
predictGbm  <- predict(modelGbm, testing)

confusionMatrix(predictGbm, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1650   41    0    2    0
##          B   18 1064   28    2   12
##          C    3   33  986   32   10
##          D    2    1   11  921    9
##          E    1    0    1    7 1051
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9638          
##                  95% CI : (0.9587, 0.9684)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9542          
##  Mcnemar's Test P-Value : 3.918e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9857   0.9342   0.9610   0.9554   0.9713
## Specificity            0.9898   0.9874   0.9839   0.9953   0.9981
## Pos Pred Value         0.9746   0.9466   0.9267   0.9756   0.9915
## Neg Pred Value         0.9943   0.9842   0.9917   0.9913   0.9936
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2804   0.1808   0.1675   0.1565   0.1786
## Detection Prevalence   0.2877   0.1910   0.1808   0.1604   0.1801
## Balanced Accuracy      0.9877   0.9608   0.9725   0.9754   0.9847
```

**2. Random Forrest**

```r
modelRf   <- randomForest(classe ~ ., data=training)
predictRF <- predict(modelRf, testing, type = "class")

confusionMatrix(predictRF, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    6    0    0    0
##          B    0 1132    7    0    0
##          C    0    1 1019    8    1
##          D    0    0    0  955    2
##          E    0    0    0    1 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9956          
##                  95% CI : (0.9935, 0.9971)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9944          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9939   0.9932   0.9907   0.9972
## Specificity            0.9986   0.9985   0.9979   0.9996   0.9998
## Pos Pred Value         0.9964   0.9939   0.9903   0.9979   0.9991
## Neg Pred Value         1.0000   0.9985   0.9986   0.9982   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1924   0.1732   0.1623   0.1833
## Detection Prevalence   0.2855   0.1935   0.1749   0.1626   0.1835
## Balanced Accuracy      0.9993   0.9962   0.9956   0.9951   0.9985
```

## Model selection

After comparing the accuracy and calculating the out of sample error for each model, I choose the random forest model.


```r
rfAcc   <- confusionMatrix(predictRF, testing$classe)$overall[1]
gbmAcc  <- confusionMatrix(predictGbm, testing$classe)$overall[1]

# The Out Sample Error (OOSE)
rferr   <- (1 - confusionMatrix(predictRF, testing$classe)$overall[1])
gbmerr  <- (1 - confusionMatrix(predictGbm, testing$classe)$overall[1])

# put into a table for easy reading
tbl = matrix( c(rfAcc, rferr, gbmAcc, gbmerr), ncol=2, byrow=TRUE)     
colnames(tbl) <- c("Accuracy","OOSE")
rownames(tbl) <- c("rf","gbm")
as.table(tbl)
```

```
##        Accuracy        OOSE
## rf  0.995581988 0.004418012
## gbm 0.963806287 0.036193713
```

## Submission: Apply the algorithm to the 20 test cases

having chosen the random forest model, for its higher accuracy and lower out-of-sample-error, I applied it to the 20 test cases.


```r
predictionFinal <- predict(modelRf, testingSet, type = "class")
predictionFinal
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
