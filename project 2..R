library(caret)
library(e1071)
library(ggvis)
library(class)
library(cluster)
library(RWeka)

#Loading IRIS and Lense Data set
datairis <- iris
lens<-read.table('lenses.data',sep = '')
value=c("nr","age","spec-prescription","astigmatic","tear","class")
colnames(lens) <- value
lens$nr<-NULL


#K fold validation iris
k = 5 #Folds

# sample from 1 to k, nrow times (the number of observations in the data)
datairis$id <- sample(1:k, nrow(datairis), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(datairis, id %in% list[-i])
  testset <- subset(datairis, id %in% list[i])
  
  # run a random forest model
  #mymodel <- randomForest(trainingset$Species ~ ., data = trainingset, ntree = 100)
  start.time <- Sys.time()
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="linear",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="polynomial",cost=1,scale=FALSE)
  mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  end.time <- Sys.time()
  avg_time <-(end.time - start.time)/2
  
  # remove response column 1, Sepal.Length
  temp <- as.data.frame(predict(mymodel, testset[,-5]))
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,5]))
  progress.bar$step()
  result <- cbind(prediction,testsetCopy)
  names(result) <- c("Predicted", "Actual")
  cat(paste0("iteration: ", i))
  print(head(result))
  count<-0
  for (i in 1:nrow(result)){
    if(result$Predicted[i]!=result$Actual[i]){count<-count+1}
      
  }
  print(count/nrow(result))
  print(avg_time)
}


cat(paste0("this ","iteration ",i,"has ",(count/nrow(testsetCopy)),"error"))

### lens SVM
lens$age<-as.factor(lens$age)
lens$`spec-prescription`<-as.factor(lens$`spec-prescription`)
lens$astigmatic<-as.factor(lens$astigmatic)
lens$tear<-as.factor(lens$tear)
lens$class<-as.factor(lens$class)
#K fold validation iris
k = 5 #Folds

# sample from 1 to k, nrow times (the number of observations in the data)
lens$id <- sample(1:k, nrow(lens), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(lens, id %in% list[-i])
  testset <- subset(lens, id %in% list[i])
  
  # run a random forest model
  #mymodel <- randomForest(trainingset$Species ~ ., data = trainingset, ntree = 100)
  start.time <- Sys.time()
  mymodel <- svm(class~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="polynomial",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  end.time <- Sys.time()
  avg_time <-(end.time - start.time)/2
  
  # remove response column 1, Sepal.Length
  temp <- as.data.frame(predict(mymodel, testset[,-5]))
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,5]))
  progress.bar$step()
  result <- cbind(prediction,testsetCopy)
  names(result) <- c("Predicted", "Actual")
  cat(paste0("iteration: ", i))
  print(head(result))
  count<-0
  for (i in 1:nrow(result)){
    if(result$Predicted[i]!=result$Actual[i]){count<-count+1}
    
  }
  print(count/nrow(result))
  print(avg_time)
}

#### Question 2## how many pca did you pick and why
datairis$Species<-as.numeric(datairis$Species)
datairis$id<-NULL
iris.pca <- prcomp(datairis[ ,-5],center = TRUE,scale. = TRUE) 
summary(iris.pca)

irisnew<-as.data.frame(iris.pca$x)
irisnew$s<-iris$Species
irisnew<-irisnew[ ,1:2]
irisnew$s<-iris$Species
###Run SVM on reduced data
k = 5 
#Folds

# sample from 1 to k, nrow times (the number of observations in the data)
irisnew$id <- sample(1:k, nrow(irisnew), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(irisnew, id %in% list[-i])
  testset <- subset(irisnew, id %in% list[i])
  
  #mymodel <- randomForest(trainingset$Species ~ ., data = trainingset, ntree = 100)
  start.time <- Sys.time()
  #mymodel <- svm(class~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="polynomial",cost=1,scale=FALSE)
  mymodel <- svm(trainingset$s ~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  end.time <- Sys.time()
  avg_time <-(end.time - start.time)/2
  
  # remove response column 1, Sepal.Length
  temp <- as.data.frame(predict(mymodel, testset[,-5]))
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,3]))
  progress.bar$step()
  result <- cbind(prediction,testsetCopy)
  names(result) <- c("Predicted", "Actual")
  cat(paste0("iteration: ", i))
  print(head(result))
  count<-0
  for (i in 1:nrow(result)){
    if(result$Predicted[i]!=result$Actual[i]){count<-count+1}
    
  }
  print(count/nrow(result))
  print(avg_time)
}



## Lens PCA
lens_id<-lens[ ,-6]
lens_id$age<-as.numeric(lens_id$age)
lens_id$`spec-prescription`<-as.numeric(lens_id$`spec-prescription`)
lens_id$astigmatic<-as.numeric(lens_id$astigmatic)
lens_id$tear<-as.numeric(lens_id$tear)
lens.pca <- prcomp(lens_id[ ,-5],center = TRUE,scale. = TRUE) 
summary(lens.pca)
plot(lens.pca)

#g <- ggbiplot(lens.pca, obs.scale = 1, var.scale = 1, groups = as.factor(lens_df[, 5]), ellipse = TRUE, circle = TRUE)
#g <- g + scale_color_discrete(name = '')
#g <- g + theme(legend.direction = 'horizontal', legend.position = 'top')
### i think i need to pick all the principle components

##### Random Forest Lens
lens_r<-lens
lens_r$id=NULL
colnames(lens_r)[2] <- "spec"
lens_r$sp<-lens_r$`spec-prescription`
k = 5 #Folds

# sample from 1 to k, nrow times (the number of observations in the data)
lens_r$id <- sample(1:k, nrow(lens_r), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(lens_r, id %in% list[-i])
  testset <- subset(lens_r, id %in% list[i])
  
  # run a random forest model
  mymodel <- randomForest(class ~ ., data = trainingset, ntree = 200)
  start.time <- Sys.time()
  #mymodel <- svm(class~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="polynomial",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  end.time <- Sys.time()
  avg_time <-(end.time - start.time)/2
  
  # remove response column 1, Sepal.Length
  temp <- as.data.frame(predict(mymodel, testset[,-5]))
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,5]))
  progress.bar$step()
  result <- cbind(prediction,testsetCopy)
  names(result) <- c("Predicted", "Actual")
  cat(paste0("iteration: ", i))
  print(head(result))
  count<-0
  for (i in 1:nrow(result)){
    if(result$Predicted[i]!=result$Actual[i]){count<-count+1}
    
  }
  print(count/nrow(result))
  print(avg_time)
}

## IRIS random forest
k = 5 #Folds
datairis<-iris
# sample from 1 to k, nrow times (the number of observations in the data)
datairis$id <- sample(1:k, nrow(datairis), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(datairis, id %in% list[-i])
  testset <- subset(datairis, id %in% list[i])
  
  # run a random forest model
  mymodel <- randomForest(trainingset$Species ~ ., data = trainingset, ntree = 200)
  start.time <- Sys.time()
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="linear",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="polynomial",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  end.time <- Sys.time()
  avg_time <-(end.time - start.time)/2
  
  # remove response column 1, Sepal.Length
  temp <- as.data.frame(predict(mymodel, testset[,-5]))
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,5]))
  progress.bar$step()
  result <- cbind(prediction,testsetCopy)
  names(result) <- c("Predicted", "Actual")
  cat(paste0("iteration: ", i))
  print(head(result))
  count<-0
  for (i in 1:nrow(result)){
    if(result$Predicted[i]!=result$Actual[i]){count<-count+1}
    
  }
  print(count/nrow(result))
  print(avg_time)
}
### J48 Iris###
k = 5 #Folds
datairis<-iris
# sample from 1 to k, nrow times (the number of observations in the data)
datairis$id <- sample(1:k, nrow(datairis), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(datairis, id %in% list[-i])
  testset <- subset(datairis, id %in% list[i])
  
  # run a random forest model
  mymodel <- J48(trainingset$Species ~ ., data = trainingset)
  start.time <- Sys.time()
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="linear",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="polynomial",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  end.time <- Sys.time()
  avg_time <-(end.time - start.time)/2
  
  # remove response column 1, Sepal.Length
  temp <- as.data.frame(predict(mymodel, testset[,-5]))
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,5]))
  progress.bar$step()
  result <- cbind(prediction,testsetCopy)
  names(result) <- c("Predicted", "Actual")
  cat(paste0("iteration: ", i))
  print(head(result))
  count<-0
  for (i in 1:nrow(result)){
    if(result$Predicted[i]!=result$Actual[i]){count<-count+1}
    
  }
  print(count/nrow(result))
  print(avg_time)
}

###Lens j48

lens_r<-lens
lens_r$id=NULL
colnames(lens_r)[2] <- "spec"
k = 5 #Folds
# sample from 1 to k, nrow times (the number of observations in the data)
lens_r$id <- sample(1:k, nrow(lens_r), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(lens_r, id %in% list[-i])
  testset <- subset(lens_r, id %in% list[i])
  
  # run a random forest model
  mymodel <- J48(class ~ ., data = trainingset)
  start.time <- Sys.time()
  #mymodel <- svm(class~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="polynomial",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  end.time <- Sys.time()
  avg_time <-(end.time - start.time)/2
  
  # remove response column 1, Sepal.Length
  temp <- as.data.frame(predict(mymodel, testset[,-5]))
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,5]))
  progress.bar$step()
  result <- cbind(prediction,testsetCopy)
  names(result) <- c("Predicted", "Actual")
  cat(paste0("iteration: ", i))
  print(head(result))
  count<-0
  for (i in 1:nrow(result)){
    if(result$Predicted[i]!=result$Actual[i]){count<-count+1}
    
  }
  print(count/nrow(result))
  print(avg_time)
}

### K means clustering
# iris
newiris <- iris
newiris$Species <- NULL
(kc <- kmeans(newiris, 3))
newiris$sp<-factor(kc$cluster)
as.factor(newiris$sp)
class(newiris$sp)
k = 5 #Folds

# sample from 1 to k, nrow times (the number of observations in the data)
newiris$id <- sample(1:k, nrow(newiris), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(newiris, id %in% list[-i])
  testset <- subset(newiris, id %in% list[i])
  
  # run a random forest model
  #mymodel <- randomForest(trainingset$Species ~ ., data = trainingset, ntree = 100)
  start.time <- Sys.time()
  #mymodel <- svm(trainingset$sp ~ ., data = trainingset,kernel="linear",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="polynomial",cost=1,scale=FALSE)
  mymodel <- J48(trainingset$sp ~ ., data = trainingset)
  end.time <- Sys.time()
  avg_time <-(end.time - start.time)/2
  
  # remove response column 1, Sepal.Length
  temp <- as.data.frame(predict(mymodel, testset[,-5]))
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,5]))
  progress.bar$step()
  result <- cbind(prediction,testsetCopy)
  names(result) <- c("Predicted", "Actual")
  cat(paste0("iteration: ", i))
  print(head(result))
  count<-0
  for (i in 1:nrow(result)){
    if(result$Predicted[i]!=result$Actual[i]){count<-count+1}
    
  }
  print(count/nrow(result))
  print(avg_time)
}

##K means lens
newlens<-lens
newlens$id<-NULL
newlens$class<-NULL
(kc <- kmeans(newlens, 3))
newlens$class<-factor(kc$cluster)
class(newlens$class)

k = 5 #Folds

# sample from 1 to k, nrow times (the number of observations in the data)
newlens$id <- sample(1:k, nrow(newlens), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)

for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(newlens, id %in% list[-i])
  testset <- subset(newlens, id %in% list[i])
  
  # run a random forest model
  #mymodel <- randomForest(trainingset$Species ~ ., data = trainingset, ntree = 100)
  start.time <- Sys.time()
  #mymodel <- svm(trainingset$class ~ ., data = trainingset,kernel="linear",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$Species ~ ., data = trainingset,kernel="polynomial",cost=1,scale=FALSE)
  mymodel <- J48(trainingset$class ~ ., data = trainingset)
  end.time <- Sys.time()
  avg_time <-(end.time - start.time)/2
  
  # remove response column 1, Sepal.Length
  temp <- as.data.frame(predict(mymodel, testset[,-5]))
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,5]))
  progress.bar$step()
  result <- cbind(prediction,testsetCopy)
  names(result) <- c("Predicted", "Actual")
  cat(paste0("iteration: ", i))
  print(head(result))
  count<-0
  for (i in 1:nrow(result)){
    if(result$Predicted[i]!=result$Actual[i]){count<-count+1}
    
  }
  print(count/nrow(result))
  print(avg_time)
}


result$Predicted
replace(result$Predicted, result$Predicted > 2 & x < 5, -1)
x2 <- pi * 100^(-1:3)
round(result$Predicted, 0)
