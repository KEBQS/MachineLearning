---
title: "Prediction Project Assignment"
course: "Practical Machine Learning" 
author: "KB"
date: "Saturday, May 28, 2016"
output: html_document
---

###Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
More information is available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

###Goal

The goal is to predict the manner in which they did the exercise.

###Exploratory Data Analysis

The two sets of data used in this assignments are :

- Training data available at <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv> (csv format)
- Test data available at <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv> (csv format)

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>. 

####Loading Data




```r
#URLs of datasets
URLTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
URLtest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

#Downloading sets of data
if (!file.exists("pml-training.csv")) {
  download.file(URLTrain, destfile = "pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
  download.file(URLtest, destfile = "pml-testing.csv")
}
TrainSet <- read.csv("pml-training.csv", sep = ",", na.strings = c("", "NA"))
TestSet <- read.csv("pml-testing.csv", sep = ",", na.strings = c("", "NA"))
```

Training dataset consists of 19622 rows and 160 columns
Testing dataset consists of 20 rows and 160 columns


```
## [1] 19622   160
```

```
## [1]  20 160
```

#### Data partitioning

Since the testing dataset will only be used at the end of the project, to answer questions about our modeling process, the training dataset is partitioned in two sets:
- a training set of 70 % of the data 
- a testing set of 30 % of the data
We will partition the data based on the "classe" variable. The classe variable consist of 5 values : A, B, C, D and E.


```r
#Partitioning training data
set.seed(3433)
inTrain  <- createDataPartition(TrainSet$classe, p=0.7, list=FALSE)
training <- TrainSet[inTrain, ]
testing  <- TrainSet[-inTrain, ]
dim(training)
```

```
## [1] 13737   160
```

```r
dim(testing)
```

```
## [1] 5885  160
```

####Cleaning Data

To build our model and predict classes, we need to remove data columns that contains mostly NAs and work with the ones with non-zero. After finding those columns, we narrow down the number of columns to 60.


```r
AllNA    <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[, AllNA==FALSE]
testing  <- testing[, AllNA==FALSE]
```

Since the first 7 columns contains the person's identification and timestamps, we proceed to remove the first 7 columns.


```r
training <- training[, -(1:7)]
testing  <- testing[, -(1:7)]

#Datasets must have same columns
newcolnames <- colnames(training)
testing <- testing[newcolnames]

dim(training)
```

```
## [1] 13737    53
```

Both datasets, training and testing must have the same data types. In order to do that, we coerce the data.


```r
for (i in 1:length(testing) ) {
    for(j in 1:length(training)) {
        if( length( grep(names(training[i]), names(testing)[j]) ) == 1)  {
            class(testing[j]) <- class(training[i])
        }      
    }      
}
```

####Column correlation to Classe

We look for columns that are highly correlated to "classe" variable, considering 90 % as high correlation rate.


```r
outcome = which(names(training) == "classe")
predictors = findCorrelation(abs(cor(training[,-outcome])),0.90)
colnames(training[predictors])
```

```
## [1] "accel_belt_z"     "roll_belt"        "accel_belt_y"    
## [4] "accel_belt_x"     "gyros_dumbbell_x" "gyros_dumbbell_z"
## [7] "gyros_arm_x"
```

```r
#training = training[,-predictors]
```

###Prediction Model building

I will build three different model algorithms and apply them to the training and testing datasets we built from the original training set. Also, we will do model comparison using the confusion matrix function, which will give us information about how well the model's did on new data sets.

Cross validation will be performed for each model (3 models).


```r
tC <- trainControl(method='cv', number = 3)
```

####Model 1: Generalized Boosted Model (gbm package)


```r
ModelGBM <- train(classe ~ ., data=training, trControl=tC, method='gbm')
predictGBM <- predict(ModelGBM, newdata=testing)
cmGBM <- confusionMatrix(predictGBM, testing$classe)
```

####Model 2: Decision trees (rpart package)

```r
ModelDT <- train(classe ~ ., data=training, method='rpart', trControl=tC)
#fancyRpartPlot(ModelDT)
predictDT <- predict(ModelDT, newdata=testing)
cmDT <- confusionMatrix(predictDT, testing$classe)
```
 
####Model 3: Random forest decision trees (rf)


```r
ModelRF <- train(classe ~ ., data=training, trControl=tC, method='rf', ntree=100)
predictRF <- predict(ModelRF, newdata=testing)
cmRF <- confusionMatrix(predictRF, testing$classe)
```


```r
ModelRF2 <- rpart(classe ~ ., data=training, method="class")
fancyRpartPlot(ModelRF2)
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png)

####Cross Validation


```r
AccuracyValues <- data.frame( Model = c('GBM', 'DT', 'RF'), Accuracy = rbind(cmGBM$overall[1], cmDT$overall[1], cmRF$overall[1]))
print(AccuracyValues)
```

```
##   Model  Accuracy
## 1   GBM 0.9643161
## 2    DT 0.4898895
## 3    RF 0.9928632
```

Based on the accuracy results, we identify Decision Tree Model as the least accurate, and Random Forest Model slightly outperforms GBM Model.


```r
RFcm = randomForest(classe ~ ., data = training, ntree=500, importance = TRUE)
#varImp(RFcm, scale = FALSE)
head(importance(RFcm))
```

```
##                         A        B        C        D        E
## roll_belt        38.59486 46.70536 46.43983 46.90896 42.39047
## pitch_belt       29.99803 49.26334 37.84752 34.12349 32.26912
## yaw_belt         43.32777 44.22819 41.17348 45.13669 32.69602
## total_accel_belt 14.55203 16.99150 14.20766 14.22087 15.48877
## gyros_belt_x     18.75663 18.25704 18.81002 14.94668 16.06333
## gyros_belt_y     12.04113 18.38569 16.64938 15.11657 19.64608
##                  MeanDecreaseAccuracy MeanDecreaseGini
## roll_belt                    55.09163        877.19711
## pitch_belt                   47.62741        490.42463
## yaw_belt                     59.82283        609.42008
## total_accel_belt             17.88405        143.71658
## gyros_belt_x                 27.20455         66.30591
## gyros_belt_y                 22.77263         79.84133
```

```r
varImpPlot(RFcm,type=2)
```

![plot of chunk unnamed-chunk-15](figure/unnamed-chunk-15-1.png)

We can identify our best predictors as : roll_belt, yaw_belt, pitch_forearm, magnet_dumbbell_x, pitch_belt.

###Prediction

Now that we found that Random Forest is the model we will use to predict a classe for each of the 20 observations contained in the testing dataset from file ('pml-testing.csv'), we do the prediction with the testing data.


```r
predictTesting <- predict(ModelRF, newdata=TestSet)
QuizResults <- data.frame(problem_id=TestSet$problem_id, predicted=predictTesting)
print(QuizResults)
```

```
##    problem_id predicted
## 1           1         B
## 2           2         A
## 3           3         B
## 4           4         A
## 5           5         A
## 6           6         E
## 7           7         D
## 8           8         B
## 9           9         A
## 10         10         A
## 11         11         B
## 12         12         C
## 13         13         B
## 14         14         A
## 15         15         E
## 16         16         E
## 17         17         A
## 18         18         B
## 19         19         B
## 20         20         B
```



###Conclusion

Based on the data was available at the time, Random Forest Model with cross validation was implemented due to its 99.2 % accuracy, to predict the 20 sample observations given as testing.
One thing to note, is lots of missing data in both datasets. We proceeded to remove some of those columns with NAs.
