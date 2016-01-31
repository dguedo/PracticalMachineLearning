# Course Project
# --------------------------

## loading libraries
library(RCurl)
library(caret)
library(randomForest)
# --
set.seed(3510)

## Securing the Data
setwd("~/GitHub/PracticalMachineLearning")
trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainingURL ,destfile="pml-training.csv",method="libcurl")
download.file(testingURL ,destfile="pml-testing.csv",method="libcurl")

trainingData  = read.csv("pml-training.csv")
testingSet    = read.csv("pml-testing.csv")

# Remove un-needed columns
columnsToRemove <- c(1,2,3,4,5,6)
trainingData <- trainingData[, -columnsToRemove]
testingSet <- testingSet[, -columnsToRemove]

## Creating the training set partitions
inTrain   <- createDataPartition(trainingData$classe, p = 0.70, list = FALSE)
training  <- trainingData[inTrain, ]
testing   <- trainingData[-inTrain, ]
# dim(training); dim(testing)

## Cleaning the data

# remove the near zero values
nearZero  <- nearZeroVar(training)
training  <- training[, -nearZero]
testing   <- testing[, -nearZero]
# removed 55 columns

# remove NA columns, where over 90% of the values are NA
training  <- training[,colSums(is.na(training)) < nrow(training)*0.9]
testing   <- testing[,colSums(is.na(testing)) < nrow(testing)*0.9]

## models
# K-fold cross-validation (10 fold)
fitControl <- trainControl(method = "cv", number = 5)

## 3. Boost
modelGbm    <- train(classe ~ .,data=training, method="gbm", trControl = fitControl, verbose = FALSE)
predictGbm  <- predict(modelGbm, testing)
# --
confusionMatrix(predictGbm, testing$classe)

## Random Forrest
#modelRf   <- train(classe ~ .,data=training, method="rf")
modelRf <- randomForest(classe ~ ., data=training)
predictRF <- predict(modelRf, testing, type = "class")
# --
confusionMatrix(predictRF, testing$classe)

#print(rf_model$finalModel)

## Results
confusionMatrix(predictGbm, testing$classe)$overall[1]
confusionMatrix(predictRF, testing$classe)$overall[1]



  
  confusionMatrix(predictGbm, testing$classe)$overall[1]

rfa <- confusionMatrix(predictRF, testing$classe)$overall[1]
rfe <- 1 - confusionMatrix(predictRF, testing$classe)$overall[1]

x = matrix( c(rfa, rfe, rfa, rfe), ncol=2, byrow=TRUE)     
colnames(x) <- c("Acc","err")
rownames(x) <- c("rf","gbm")
smoke <- as.table(x)
smoke

# 
# 1 - confusionMatrix(predictRF, testing$classe)$overall[1]
# 
# testing.count <- nrow(testing)
# out.of.sample.error <- sum(testing$classe != predictRF) / testing.count

## apply to test set 
# remove the near zero values
#testingSet <- testingSet[, -nearZero]
#testingSet <- testingSet[,colSums(is.na(testingSet)) < nrow(testingSet)*0.9]


predictionFinal <- predict(modelRf, testingSet, type = "class")
predictionFinal


## Conclusion
#1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
#B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 

