# Course Project
# --------------------------

library(RCurl)
trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainingURL ,destfile="pml-training.csv",method="libcurl")
download.file(testingURL ,destfile="pml-testing.csv",method="libcurl")

training  = read.csv("pml-training.csv")
testing   = read.csv("pml-testing.csv")

# --
set.seed(90210)

modelRf   <- train(classe ~ .,data=training, method="rf")
modelGbm  <- train(classe ~ .,data=training, method="gbm")
modelLda  <- train(classe ~ .,data=training, method="lda")

# --
modelRf
modelGbm
modelLda

---
pred1 <- predict(modelRf, testing)
pred2 <- predict(modelGbm, testing)
pred3 <- predict(modelLda, testing)

predDF <- data.frame(pred1,pred2, pred3, diagnosis=testing$diagnosis)

combModFit <- train(diagnosis ~.,method="rf",data=predDF)
combPred <- predict(combModFit, predDF)

confusionMatrix(pred1, testing$diagnosis)$overall[1]
confusionMatrix(pred2, testing$diagnosis)$overall[1]
confusionMatrix(pred3, testing$diagnosis)$overall[1]
confusionMatrix(combPred, testing$diagnosis)$overall[1]
