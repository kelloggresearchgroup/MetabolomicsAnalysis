library(ggplot2)
library(cowplot)
library(randomForest)
library(caret) 
library(data.table) # used for reading and manipulation of data
library(dplyr)      # used for data manipulation and joining
library(glmnet)     # used for regression    
library(xgboost)    # used for building XGBoost model
library(e1071)      # used for skewness
library(cowplot)    # used for combining multiple plots
library(mixOmics)
library(MetabolAnalyze)
library(ggrepel)
library(mdatools)
library(reshape2)
theme_set(theme_classic())

##set up data sets
# load data 
data <-  read.csv("GH_data.csv", row.names = 1, header = TRUE)
#transform data
data <- sqrt(data)
##replace NA values with small number
data[is.na(data)] = 0.0001
##replace 0 with small value
data <- replace(data, data<0,0.0001)

#organize data by species
OBdata <- data[,1:10]
OGdata <- data[,11:18]
OTdata <- data[,19:31]
tOB <- t(OBdata)
tOG <- t(OGdata)
tOT <- t(OTdata)
tNOTot <- rbind(tOB, tOG)
tNOTob <- rbind(tOG, tOT)
tNOTog <- rbind(tOB, tOT)

#LASSO MODEL#
##set up is OT (1) is not OT (2)
X <- rbind(tOT, tNOTot) ## all data 
Y <- c( 'OT', 'OT', 'OT', 'OT', 'OT', 'OT', 'OT','OT','OT','OT','OT','OT','OT',
        'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT','nOT', 'nOT',
        'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT')
Y <- ifelse(Y == "OT", 1, 0)
Xm <- as.matrix(X)
Ym <- as.matrix(Y)

#set up data frames for storing model information 
#make empty matrix to store variables used in each model
var <- as.data.frame(matrix(ncol = 1123, nrow = 0))
colnames(var) <- c(colnames(X))
#make empty data frame to store validation results from each run (test set)
resultsOT <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsOT) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                         'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store the training set for each run 
training.runsOT <- data.frame(matrix(ncol = 20,nrow = 0))
colnames(training.runsOT) <- c(paste0(1:20)
#make empty data frame to store the lambda min for each run
lambda.min.OT <- data.frame(matrix(ncol = 1, nrow = 0))
#make empty data frame to store coefficients and variable selection) - note, variable numbers are position 
#in the list, not actual variable number, so will have to refer back to the starting list to find the number
coef.OT <- data.frame(matrix(ncol = 1123, nrow = 0))
variables.OT <- data.frame(matrix(ncol = 1123, nrow = 0))
#make empty data frame to store test set predictions
test.storeOT <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(test.storeOT) <- c('OT', 'OT', 'OT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT')

#load dried data
dried <-  read.csv("Consumer_data.csv", row.names = 1, header = TRUE)
tdried <- t(dried)
#transform data
TSdried <- sqrt(tdried)
##replace NA values with small number
TSdried[is.na(TSdried)] = 0.0001
##replace 0 with small value
TSdried <- replace(TSdried, TSdried<0,0.0001)
y.dried <- c('OT', 'OT', 'nOT', 'OT', 'nOT', 'OT', 'OT','nOT', 'nOT', 'OT', 'nOT', 'OT', 'nOT', 'OT', 'OT', 'OT', 'OT')

#make empty dataframe for consumer set predictions
pred.storedriedOT <- data.frame(matrix(ncol = 17, nrow = 0))
resultsdriedOT <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsdriedOT) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                              'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')

##set up validation set for OT is not OT
VS <-  read.csv("VS_data.csv", row.names = 1, header = TRUE)
tVS <- t(VS)
#transform data
tsVS <- sqrt(tVS)
##replace NA values with small number
tsVS[is.na(tsVS)] = 0.0001
##replace 0 with small value
tsVS <- replace(tsVS, tsVS<0,0.0001)
y.VS <- c('nOT', 'OT', 'OT', 'nOT','nOT','nOT', 'OT', 'OT','nOT')
#make empty dataframe for VS predictions
pred.storedVSOT <- data.frame(matrix(ncol = 9, nrow = 0))
resultsVSOT <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsVSOT) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')

#LASSO model with 100 repeats
for(i in 1:100) {
  set.seed(NULL)
  ##split into test and train
  data_set_size_OT <- floor(nrow(tOT)*0.8)
  index_OT <- sample(1:nrow(tOT), size = data_set_size_OT)
  data_set_size_notOT <- floor(nrow(tOT)*0.8)
  index_notOT <- sample(1:nrow(tNOTot), size = data_set_size_notOT)
  train_OT <- tOT[index_OT,]
  test_OT <- tOT[-index_OT,]
  train_nOT <- tNOTot[index_OT,]
  test_nOT <- tNOTot[-index_OT,]
  X.trainOT <- as.data.frame(rbind(train_OT, train_nOT))
  X.trainOT <- as.matrix(X.trainOT)
  X.testOT <- as.data.frame(rbind(test_OT, test_nOT))
  X.testOT <- as.matrix(X.testOT)
  Y.trainOT <- c('OT', 'OT', 'OT', 'OT', 'OT','OT','OT','OT','OT','OT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT','nOT','nOT','nOT','nOT',)
  Y.trainOTn <- ifelse(Y.trainOT == "OT", 1, 0)
  Y.testOT <- c('OT', 'OT', 'OT','nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT','nOT','nOT')
  Y.testOTn <- ifelse(Y.testOT == "OT", 1, 0)
  ##autoscale data in training set
  trainmeans <- as.data.frame(colMeans(X.trainOT)) #calculate means
  trainSD <- as.data.frame(apply(X.trainOT,2, sd)) #calculate SDs
  train_meansub <- X.trainOT[] - trainmeans[col(X.trainOT[])] #subtract means from each cell
  train_divdSD <- train_meansub[] / trainSD[col(train_meansub[])] #divide each cell by the SD
  X.trainOT <- train_divdSD
  #autoscale test data using training data means and SDs
  test_meansub <-  X.testOT[] - trainmeans[col(X.testOT[])]
  test_divdSD <- test_meansub[] / trainSD[col(test_meansub[])]
  X.testOT <- test_divdSD
  #store training set information
  training.transposeOT <- t(X.trainOT)
  training.row.namesOT <- rownames(X.trainOT)
  training.runsOT <- rbind(training.runsOT, training.row.namesOT)
  ##replace NA values with small number
  X.trainOT[is.na(X.trainOT)] = 0.001
  X.testOT[is.na(X.testOT)] = 0.0001
  ##replace 0 with small value
  X.trainOT <- replace(X.trainOT, X.trainOT<0,0.001)
  X.testOT <- replace(X.testOT, X.testOT<0,0.0001)
  # Find the best lambda using cross-validation
  ttrain <- t(X.trainOT)
  cv.lassoOT <- cv.glmnet(X.trainOT, Y.trainOTn, alpha = 1, nfolds = 10, family = "binomial")
  #store lambda.min
  lambda.min.OT <- rbind(lambda.min.OT, cv.lassoOT$lambda.min)
  # Fit the final model on the training data
  modelOT <- glmnet(X.trainOT, Y.trainOTn, alpha = 1, family = "binomial",
                    lambda = cv.lassoOT$lambda.min)
  # Store variables with non-zero coefficients
  coefOT <- summary(coef(modelOT))
  coefOTx <- t(coefOT$x)
  coefOTi <- as.data.frame(t(coefOT$i))
  coffsel <- which(colnames(X) %in% coefOTi[1,])
  vari <- data.frame(matrix(ncol = 1123, nrow = 0))
  colnames(vari) <- c(colnames(X))
  vari[nrow(vari) +1,] <- c( ifelse(colnames(vari) %in% coffsel, 1, 0))
  var <- rbind(var, vari)
  #predict test set
  probabilitiesOT <- modelOT %>% predict(newx = X.testOT)
  predicted.classesOT <- ifelse(probabilitiesOT > 0.5, "OT", "nOT")
  # Model accuracy
  mean(predicted.classesOT == Y.testOT)
  predicted.classesOT <- as.data.frame(predicted.classesOT)
  #make confusion matrix for test set predictions to calculate validation scores
  expectedOT <- as.factor(Y.testOT)
  predictedOT <- as.factor(predicted.classesOT$s0)
  pred.matrixOT <- confusionMatrix(data=predictedOT, reference=expectedOT, positive = "OT") #make confusion matrix
  pred.outcomeOT <- as.matrix(pred.matrixOT, what = "classes") #print the validation results
  pred.rowOT <- t(pred.outcomeOT) ##make results into a tables
  #store test set validation scores and prediction results
  test.storeOT <- rbind(test.storeOT, t(predicted.classesOT$s0))
  resultsOT <- rbind(resultsOT, pred.rowOT)
  #autoscale consumer set data using training data means and SDs
  dried_meansub <-  TSdried[] - trainmeans[col(TSdried[])]
  dried_divdSD <- dried_meansub[] / trainSD[col(dried_meansub[])]
  TSPdried <- dried_divdSD
  TSPdried[is.infinite(TSPdried)] <- NA
  ##replace NA values with small number
  TSPdried[is.na(TSPdried)] = 0.0001
  ##replace 0 with small value
  TSPdried <- replace(TSPdried, TSPdried<0,0.0001)
  #predict consumer products 
  driedOT <- modelOT %>% predict(newx = TSPdried)
  predicted.classesdriedOT <- ifelse(driedOT > 0.5, "OT", "nOT")
  # Model accuracy
  predicted.classesdriedOT <- as.data.frame(predicted.classesdriedOT)
  #make confusion matrix to calculate consumer product validation scores
  expecteddriedOT <- as.factor(y.dried)
  predicteddriedOT <- as.factor(predicted.classesdriedOT$s0)
  pred.matrixdriedOT <- confusionMatrix(data=predicteddriedOT, reference=expecteddriedOT, positive = "OT") #make confusion matrix
  pred.outcomedriedOT <- as.matrix(pred.matrixdriedOT, what = "classes") #print the validation results
  pred.rowdriedOT <- t(pred.outcomedriedOT) ##make results into a tables
  #store consumer product predictions and validation scores
  pred.storedriedOT <- rbind(pred.storedriedOT, t(predicted.classesdriedOT$s0))
  resultsdriedOT <- rbind(resultsdriedOT, pred.rowdriedOT)
  #autoscale VS data using training data means and SDs
  VS_meansub <-  tsVS[] - trainmeans[col(tsVS[])]
  VS_divdSD <- VS_meansub[] / trainSD[col(VS_meansub[])]
  TSPVS <- VS_divdSD
  TSPVS[is.infinite(TSPVS)] <- NA
  ##replace NA values with small number
  TSPVS[is.na(TSPVS)] = 0.0001
  ##replace 0 with small value
  TSPVS <- replace(TSPVS, TSPVS<0,0.0001)
  #predict VS
  VSOT <- modelOT %>% predict(newx = TSPVS)
  predicted.classesVSOT <- ifelse(VSOT > 0.5, "OT", "nOT")
  # Model accuracy
  predicted.classesVSOT <- as.data.frame(predicted.classesVSOT)
  #make confusion matrix to calculate VS validation scores and predictions
  expectedVSOT <- as.factor(y.VS)
  predictedVSOT <- as.factor(predicted.classesVSOT$s0)
  pred.matrixVSOT <- confusionMatrix(data=predictedVSOT, reference=expectedVSOT, positive = "OT") #make confusion matrix
  pred.outcomeVSOT <- as.matrix(pred.matrixVSOT, what = "classes") #print the validation results
  pred.rowVSOT <- t(pred.outcomeVSOT) ##make results into a tables
  #store VS predictions and validation scores
  pred.storedVSOT <- rbind(pred.storedVSOT, t(predicted.classesVSOT$s0))
  resultsVSOT <- rbind(resultsVSOT, pred.rowVSOT)
  
}

##set up is OB is not OB
X <- rbind(tOB, tNOTob) ## all data 
Y <- c('OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'nOB', 'nOB','nOB','nOB','nOB','nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB','nOB')
Y <- ifelse(Y == "OB", 1, 0)
Xm <- as.matrix(X)
Ym <- as.matrix(Y)

#make empty matrix to store variables used in model
varOB <- as.data.frame(matrix(ncol = 1123, nrow = 0))
colnames(varOB) <- c(colnames(X))
#make empty data frame to store validation results from each run (test set)
resultsOB <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsOB) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                         'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store the training set for each run 
training.runsOB <- data.frame(matrix(ncol = 16, nrow = 0))
colnames(training.runsOB) <- c(paste0(1:16))
#make empty data frame to store the lambda min 
lambda.min.OB <- data.frame(matrix(ncol = 1, nrow = 0))
#make empty data frame to store test set predictions
pred.storeOB <- data.frame(matrix(ncol = 15, nrow = 0))
colnames(pred.storeOB) <- c('OB', 'OB', 'nOB', 'nOB', 'nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB')
#load consumer product data
dried <-  read.csv("Consumer_data.csv", row.names = 1, header = TRUE)
tdried <- t(dried)
#transform data
TSdried <- sqrt(tdried)
##replace NA values with small number
TSdried[is.na(TSdried)] = 0.0001
##replace 0 with small value
TSdriedOB <- replace(TSdried, TSdried<0,0.0001)
y.predict <- c('nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'OB', 'OB', 'nOB', 'OB', 'nOB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB')
#make empty dataframe for predictions for the consumer products
pred.storedriedOB <- data.frame(matrix(ncol = 17, nrow = 0))
resultsdriedOB <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsdriedOB) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                              'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
##set up validation set 
VS <-  read.csv("VS_data.csv", row.names = 1, header = TRUE)
tVS <- t(VS)
#transform data
tsVS <- sqrt(tVS)
##replace NA values with small number
tsVS[is.na(tsVS)] = 0.0001
##replace 0 with small value
tsVS <- replace(tsVS, tsVS<0,0.0001)
y.VS <- c('nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB','nOB', 'OB')
#make empty dataframe for VS predictions
pred.storedVSOB <- data.frame(matrix(ncol = 7, nrow = 0))
resultsVSOB <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsVSOB) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#build and run model 100 times for is/isnot OB
for(i in 1:100) {
  set.seed(NULL)
  ##split into test and train
  data_set_size_OB <- floor(nrow(tOB)*0.80)
  index_OB <- sample(1:nrow(tOB), size = data_set_size_OB)
  data_set_size_notOB <- floor(nrow(tOB)*0.80)
  index_notOB <- sample(1:nrow(tNOTob), size = data_set_size_notOB)
  train_OB <- tOB[index_OB,]
  test_OB <- tOB[-index_OB,]
  train_nOB <- tNOTob[index_OB,]
  test_nOB <- tNOTob[-index_OB,]
  
  X.trainOB <- as.data.frame(rbind(train_OB, train_nOB))
  X.trainOB <- as.matrix(X.trainOB)
  X.testOB <- as.data.frame(rbind(test_OB, test_nOB))
  X.testOB <- as.matrix(X.testOB)
  Y.trainOB <- c('OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB')
  Y.trainOBn <- ifelse(Y.trainOB == "OB", 1, 0)
  Y.testOB <- c('OB', 'OB', 'nOB', 'nOB','nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB')
  Y.testOBn <- ifelse(Y.testOB == "OB", 1, 0)
  ##autoscale data in training set
  trainmeansOB <- as.data.frame(colMeans(X.trainOB)) #calculate means
  trainSDOB <- as.data.frame(apply(X.trainOB,2, sd)) #calculate SDs
  train_meansubOB <- X.trainOB[] - trainmeansOB[col(X.trainOB[])] #subtract means from each cell
  train_divdSDOB <- train_meansubOB[] / trainSDOB[col(train_meansubOB[])] #divide each cell by the SD
  X.trainOB <- train_divdSDOB
  #autoscale test data using training data means and SDs
  test_meansubOB <-  X.testOB[] - trainmeansOB[col(X.testOB[])]
  test_divdSDOB <- test_meansubOB[] / trainSDOB[col(test_meansubOB[])]
  X.testOB <- test_divdSDOB
  ##replace NA values with small number
  X.trainOB[is.na(X.trainOB)] = 0.001
  X.testOB[is.na(X.testOB)] = 0.0001
  ##replace 0 with small value
  X.trainOB <- replace(X.trainOB, X.trainOB<0,0.001)
  X.testOB <- replace(X.testOB, X.testOB<0,0.0001)
  #store training set information
  training.transposeOB <- t(X.trainOB)
  training.row.namesOB <- rownames(X.trainOB)
  training.runsOB <- rbind(training.runsOB, training.row.namesOB)
  # Find the best lambda using cross-validation
  cv.lassoOB <- cv.glmnet(X.trainOB, Y.trainOB, alpha = 1, nfolds = 20, family = "binomial")
  #store lambda.min
  lambda.min.OB <- rbind(lambda.min.OB, cv.lassoOB$lambda.min)
  # Fit the final model on the training data
  modelOB <- glmnet(X.trainOB, Y.trainOB, alpha = 1, family = "binomial",
                    lambda = cv.lassoOB$lambda.min)
  # Store variables with non-zero coefficients
  coefOB <- summary(coef(modelOB))
  coefOBx <- t(coefOB$x)
  coefOBi <- as.data.frame(t(coefOB$i))
  coffselOB <- which(colnames(X) %in% coefOBi[1,])
  variOB <- data.frame(matrix(ncol = 1123, nrow = 0))
  colnames(variOB) <- c(colnames(X))
  variOB[nrow(variOB) +1,] <- c( ifelse(colnames(variOB) %in% coffselOB, 1, 0))
  varOB <- rbind(varOB, variOB)
  #predict test set
  probabilitiesOB <- modelOB %>% predict(newx = X.testOB)
  predicted.classesOB <- ifelse(probabilitiesOB > 0.5, "OB", "nOB")
  # Model accuracy
  mean(predicted.classesOB == Y.testOB)
  predicted.classesOB <- as.data.frame(predicted.classesOB)
  #make confusion matrix to calculate test set validation scores
  expectedOB <- as.factor(Y.testOB)
  predictedOB <- as.factor(predicted.classesOB$s0)
  pred.matrixOB <- confusionMatrix(data=predictedOB, reference=expectedOB, positive = "OB") #make confusion matrix
  pred.outcomeOB <- as.matrix(pred.matrixOB, what = "classes") #print the validation results
  pred.rowOB <- t(pred.outcomeOB) ##make results into a tables
  #store test set predictions and validation scores
  pred.storeOB <- rbind(pred.storeOB, t(predicted.classesOB$s0))
  resultsOB <- rbind(resultsOB, pred.rowOB)
  #autoscale consumer product set data using training data means and SDs
  dried_meansubOB <-  TSdriedOB[] - trainmeansOB[col(TSdriedOB[])]
  dried_divdSDOB <- dried_meansubOB[] / trainSDOB[col(dried_meansubOB[])]
  TSPdried <- dried_divdSDOB
  #predict consumer product set
  driedOB <- modelOB %>% predict(newx = TSPdried)
  predicted.classesdriedOB <- ifelse(driedOB > 0.5, "OB", "nOB")
  # Model accuracy
  predicted.classesdriedOB <- as.data.frame(predicted.classesdriedOB)
  #make confusion matrix to calculate consumer product validation scores
  expecteddriedOB <- as.factor(y.predict)
  predicteddriedOB <- as.factor(predicted.classesdriedOB$s0)
  pred.matrixdriedOB <- confusionMatrix(data=predicteddriedOB, reference=expecteddriedOB, positive = "OB") #make confusion matrix
  pred.outcomedriedOB <- as.matrix(pred.matrixdriedOB, what = "classes") #print the validation results
  pred.rowdriedOB <- t(pred.outcomedriedOB) ##make results into a tables
  #store consumer product predictions and validation scores
  pred.storedriedOB <- rbind(pred.storedriedOB, t(predicted.classesdriedOB$s0))
  resultsdriedOB <- rbind(resultsdriedOB, pred.rowdriedOB)
  #autoscale VS data using training data means and SDs
  VS_meansub <-  tsVS[] - trainmeansOB[col(tsVS[])]
  VS_divdSD <- VS_meansub[] / trainSDOB[col(VS_meansub[])]
  TSPVS <- VS_divdSD
  TSPVS[is.infinite(TSPVS)] <- NA
  ##replace NA values with small number
  TSPVS[is.na(TSPVS)] = 0.0001
  ##replace 0 with small value
  TSPVS <- replace(TSPVS, TSPVS<0,0.0001)
  #predict VS
  VSOB <- modelOB %>% predict(newx = TSPVS)
  predicted.classesVSOB <- ifelse(VSOB > 0.5, "OB", "nOB")
  # Model accuracy
  predicted.classesVSOB <- as.data.frame(predicted.classesVSOB)
  #make confusion matrix to calcuate VS validation scores
  expectedVSOB <- as.factor(y.VS)
  predictedVSOB <- as.factor(predicted.classesVSOB$s0)
  pred.matrixVSOB <- confusionMatrix(data=predictedVSOB, reference=expectedVSOB, positive = "OB") #make confusion matrix
  pred.outcomeVSOB <- as.matrix(pred.matrixVSOB, what = "classes") #print the validation results
  pred.rowVSOB <- t(pred.outcomeVSOB) ##make results into a tables
  pred.storedVSOB <- rbind(pred.storedVSOB, t(predicted.classesVSOB$s0))
  resultsVSOB <- rbind(resultsVSOB, pred.rowVSOB)
  
}

#set up is OG (1) is not OG (2)
X <- rbind(tOG, tNOTog) ## all data 
YOG <- c('OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'nOG', 'nOG', 'nOG', 
         'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG',
         'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
Y <- ifelse(YOG == "OG", 1, 2)
Xm <- as.matrix(X)
Ym <- as.matrix(Y)

#make empty matrix to store variable
varOG <- as.data.frame(matrix(ncol = 1123, nrow = 0))
colnames(varOG) <- c(colnames(X))
#make empty data frame to store validation results from each run (test set)
resultsOG <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsOG) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                         'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store the training set for each run 
training.runsOG <- data.frame(matrix(ncol = 12, nrow = 0))
colnames(training.runsOG) <- c(paste0(1:12))
#make empty data frame to store the lambda min 
lambda.min.OG <- data.frame(matrix(ncol = 1, nrow = 0))
#make empty data frame to store test set predictions
pred.storeOG <- data.frame(matrix(ncol = 19, nrow = 0))
colnames(pred.storeOG)  <- c('OG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 
                             'nOG', 'nOG','nOG','nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
#load consumer product data
dried <-  read.csv("Consumer_data.csv", row.names = 1, header = TRUE)
tdried <- t(dried)
#transform data
TSdried <- sqrt(tdried)
##replace NA values with small number
TSdried[is.na(TSdried)] = 0.0001
##replace 0 with small value
TSdriedOG <- replace(TSdried, TSdried<0,0.0001)
y.predict <- c('nOG', 'nOG', 'OG', 'nOG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'OG', 'nOG', 'nOG', 'OG')
#make empty dataframe for consumer product predictions
pred.storedriedOG <- data.frame(matrix(ncol = 17, nrow = 0))
resultsdriedOG <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsdriedOG) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                              'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
##set up validation set
VS <-  read.csv("VS_data.csv", row.names = 1, header = TRUE)
tVS <- t(VS)
#transform data
tsVS <- sqrt(tVS)
##replace NA values with small number
tsVS[is.na(tsVS)] = 0.0001
##replace 0 with small value
tsVS <- replace(tsVS, tsVS<0,0.0001)
y.VS <- c('OG', 'nOG', 'nOG', 'OG', 'nOG','nOG','nOG', 'OG', 'nOG')
#make empty dataframe for VS predictions
pred.storedVSOG <- data.frame(matrix(ncol = 7, nrow = 0))
resultsVSOG <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsVSOG) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')

for(i in 1:100) {
  set.seed(NULL)
  ##split into test and train
  data_set_size_OG <- floor(nrow(tOG)*0.8)
  index_OG <- sample(1:nrow(tOG), size = data_set_size_OG)
  data_set_size_notOG <- floor(nrow(tOG)*0.8)
  index_notOG <- sample(1:nrow(tNOTog), size = data_set_size_notOG)
  train_OG <- tOG[index_OG,]
  test_OG <- tOG[-index_OG,]
  train_nOG <- tNOTog[index_OG,]
  test_nOG <- tNOTog[-index_OG,]
  X.trainOG <- as.data.frame(rbind(train_OG, train_nOG))
  X.trainOG <- as.matrix(X.trainOG)
  X.testOG <- as.data.frame(rbind(test_OG, test_nOG))
  X.testOG <- as.matrix(X.testOG)
  Y.trainOG <- c('OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
  Y.trainOGn <- ifelse(Y.trainOG == "OG", 1, 0)
  Y.testOG <- c('OG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG','nOG','nOG', 'nOG', 'nOG', 'nOG')
  Y.testOGn <- ifelse(Y.testOG == "OG", 1, 0)
  ##autoscale training data
  trainmeansOG <- as.data.frame(colMeans(X.trainOG)) #calculate means
  trainSDOG <- as.data.frame(apply(X.trainOG,2, sd)) #calculate SDs
  train_meansubOG <- X.trainOG[] - trainmeansOG[col(X.trainOG[])] #subtract means from each cell
  train_divdSDOG <- train_meansubOG[] / trainSDOG[col(train_meansubOG[])] #divide each cell by the SD
  X.trainOG <- train_divdSDOG
  #autoscale test data using training data means and SDs
  test_meansubOG <-  X.testOG[] - trainmeansOG[col(X.testOG[])]
  test_divdSDOG <- test_meansubOG[] / trainSDOG[col(test_meansubOG[])]
  X.testOG <- test_divdSDOG
  ##replace NA values with small number
  X.trainOG[is.na(X.trainOG)] = 0.001
  X.testOG[is.na(X.testOG)] = 0.0001
  ##replace 0 with small value
  X.trainOG <- replace(X.trainOG, X.trainOG<0,0.001)
  X.testOG <- replace(X.testOG, X.testOG<0,0.0001)
  #store training set information
  training.transposeOG <- t(X.trainOG)
  training.row.namesOG <- rownames(X.trainOG)
  training.runsOG <- rbind(training.runsOG, training.row.namesOG)
  # Find the best lambda using cross-validation
  cv.lassoOG <- cv.glmnet(X.trainOG, Y.trainOG, alpha = 1, nfolds = 10, family = "binomial")
  # Fit the final model on the training data
  modelOG <- glmnet(X.trainOG, Y.trainOG, alpha = 1, family = "binomial",
                    lambda = cv.lassoOG$lambda.min)
  #store lambda.min
  lambda.min.OG <- rbind(lambda.min.OG, cv.lassoOG$lambda.min)
  # Store variables with non-zero coefficients
  coefOG <- summary(coef(modelOG))
  coefOGx <- t(coefOG$x)
  coefOGi <- as.data.frame(t(coefOG$i))
  coffselOG <- which(colnames(X) %in% coefOGi[1,])
  variOG <- data.frame(matrix(ncol = 1123, nrow = 0))
  colnames(variOG) <- c(colnames(X))
  variOG[nrow(variOG) +1,] <- c( ifelse(colnames(variOG) %in% coffselOG, 1, 0))
  varOG <- rbind(varOG, variOG)
  #predict test set
  probabilitiesOG <- modelOG %>% predict(newx = X.testOG)
  predicted.classesOG <- ifelse(probabilitiesOG > 0.5, "OG", "nOG")
  # Model accuracy
  mean(predicted.classesOG == Y.testOG)
  predicted.classesOG <- as.data.frame(predicted.classesOG)
  #make confusion matrix for calculating test set validation scores
  expectedOG <- as.factor(Y.testOG)
  predictedOG <- as.factor(predicted.classesOG$s0)
  pred.matrixOG <- confusionMatrix(data=predictedOG, reference=expectedOG, positive= "OG") #make confusion matrix
  pred.outcomeOG <- as.matrix(pred.matrixOG, what = "classes") #print the validation results
  pred.rowOG <- t(pred.outcomeOG) ##make results into a tables
  #store test set predcitions and validation scores
  pred.storeOG <- rbind(pred.storeOG, t(predicted.classesOG$s0))
  resultsOG <- rbind(resultsOG, pred.rowOG)
  #autoscale consumer product set based on training set
  dried_meansubOG <-  TSdriedOG[] - trainmeansOG[col(TSdriedOG[])]
  dried_divdSDOG <- dried_meansubOG[] / trainSDOG[col(dried_meansubOG[])]
  TSPdried <- dried_divdSDOG
  TSPdried[is.infinite(TSPdried)] <- NA
  ##replace NA values with small number
  TSPdried[is.na(TSPdried)] = 0.0001
  ##replace 0 with small value
  TSPdried <- replace(TSPdried, TSPdried<0,0.0001)
  driedOG <- modelOG %>% predict(newx = TSPdried)
  predicted.classesdriedOG <- ifelse(driedOG > 0.5, "OG", "nOG")
  # Model accuracy
  predicted.classesdriedOG <- as.data.frame(predicted.classesdriedOG)
  #make confusion matrix to calculate consumer product validation scores
  expecteddriedOG <- as.factor(y.predict)
  predicteddriedOG <- as.factor(predicted.classesdriedOG$s0)
  pred.matrixdriedOG <- confusionMatrix(data=predicteddriedOG, reference=expecteddriedOG, positive= "OG") #make confusion matrix
  pred.outcomedriedOG <- as.matrix(pred.matrixdriedOG, what = "classes") #print the validation results
  pred.rowdriedOG <- t(pred.outcomedriedOG) ##make results into a tables
  #store consumer product validation scores and predicitons
  pred.storedriedOG <- rbind(pred.storedriedOG, t(predicted.classesdriedOG$s0))
  resultsdriedOG <- rbind(resultsdriedOG, pred.rowdriedOG)
  #autoscale VS data using training data means and SDs
  VS_meansub <-  tsVS[] - trainmeansOG[col(tsVS[])]
  VS_divdSD <- VS_meansub[] / trainSDOG[col(VS_meansub[])]
  TSPVS <- VS_divdSD
  TSPVS[is.infinite(TSPVS)] <- NA
  ##replace NA values with small number
  TSPVS[is.na(TSPVS)] = 0.0001
  ##replace 0 with small value
  TSPVS <- replace(TSPVS, TSPVS<0,0.0001)
  #predict VS
  VSOG <- modelOG %>% predict(newx = TSPVS)
  predicted.classesVSOG <- ifelse(VSOG > 0.5, "OG", "nOG")
  # Model accuracy
  predicted.classesVSOG <- as.data.frame(predicted.classesVSOG)
  #make confusion matrix to caluclate VS validation scores
  expectedVSOG <- as.factor(y.VS)
  predictedVSOG <- as.factor(predicted.classesVSOG$s0)
  pred.matrixVSOG <- confusionMatrix(data=predictedVSOG, reference=expectedVSOG, positive= "OG") #make confusion matrix
  pred.outcomeVSOG <- as.matrix(pred.matrixVSOG, what = "classes") #print the validation results
  pred.rowVSOG <- t(pred.outcomeVSOG) ##make results into a tables
  #store VS predictions and VS scores
  pred.storedVSOG <- rbind(pred.storedVSOG, t(predicted.classesVSOG$s0))
  resultsVSOG <- rbind(resultsVSOG, pred.rowVSOG)
}

##PLS-DA MODEL##

#first, set up a grid of values to be assessed for each component
list.keepX <- c(5:10, seq(20, 100, 10))
#Set up is OT is not OT
X <- rbind(tOT, tNOTot)
Y <- c('OT', 'OT', 'OT', 'OT', 'OT', 'OT', 'OT', 'OT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT',
       'nOT', 'nOT','nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT','nOT','nOT')
#make empty data frame to store validation results from each run (test set)
results <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(results) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                       'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store the training set for each run 
training.runs <- data.frame(matrix(ncol = 20, nrow = 0))
colnames(training.runs) <- c(paste0(1:20))
#make empty data frame to store the predictions for each run 
predictionstorage <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(predictionstorage)  <- c('OT', 'OT','OT' 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT','nOT','nOT')
#make empty data frame to store ncomps selected for each run 
ncomps.runs <- data.frame(matrix(ncol = 1, nrow = 0))
colnames(ncomps.runs) <- c('ncomps')
#make empty data frame to store keepX selected for each component for each run 
keepx.runs <- data.frame(matrix(ncol = 2, nrow = 0))
#load consumer product data
dried <-  read.csv("Consumer_data.csv", row.names = 1, header = TRUE)
tdried <- t(dried)
#transform data
TSdried <- sqrt(tdried)
##replace NA values with small number
TSdried[is.na(TSdried)] = 0.0001
##replace 0 with small value
TSdried <- replace(TSdried, TSdried<0,0.0001)
y.predict <- c('OT', 'OT', 'nOT', 'OT', 'nOT', 'OT', 'OT', 'nOT', 'nOT', 'OT', 'nOT', 'OT', 'nOT', 'OT', 'nOT', 'OT','OT')
#make empty dataframe for predictions
driedpredOT <- data.frame(matrix(ncol = 17, nrow = 0))
resultsdriedOT <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsdriedOT) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                              'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
##set up validation set for OT is not OT
VS <-  read.csv("VS_data.csv", row.names = 1, header = TRUE)
tVS <- t(VS)
#transform data
tsVS <- sqrt(tVS)
##replace NA values with small number
tsVS[is.na(tsVS)] = 0.0001
##replace 0 with small value
TSVS <- replace(tsVS, tsVS<0,0.0001)
y.VS <- c('nOT', 'OT', 'OT', 'nOT','nOT','nOT', 'OT', 'OT', 'nOT')
#make empty data frame to store the validation set predictions for each run 
VSpredOT <- data.frame(matrix(ncol = 9, nrow = 0))
colnames(VSpredOT)  <- c('nOT', 'OT', 'OT', 'nOT','nOT','nOT', 'OT', 'OT', 'nOT')
#make empty data frame to store consumer product val scores
resultsVSOT <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsVSOT) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#set up data frame to store the selected features for each run (top 20)
OT.features.PLS <- data.frame(matrix(ncol = 20, nrow = 0))

for(i in 1:100) {
  set.seed(NULL)
  #set up test and training sets
  data_set_size_OT <- floor(nrow(tOT)*0.80)
  index_OT <- sample(1:nrow(tOT), size = data_set_size_OT)
  data_set_size_notOT <- floor(nrow(tOT)*0.80)
  index_notOT <- sample(1:nrow(tNOTot), size = data_set_size_notOT)
  train_OT <- tOT[index_OT,]
  test_OT <- tOT[-index_OT,]
  train_nOT <- tNOTot[index_OT,]
  test_nOT <- tNOTot[-index_OT,]
  X.train <- as.data.frame(rbind(train_OT, train_nOT))
  X.test <- as.data.frame(rbind(test_OT, test_nOT))
  Y.train <- c('OT', 'OT', 'OT', 'OT', 'OT', 'OT', 'OT','OT','OT','OT',
               'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT')
  Y.test <- c('OT', 'OT', 'OT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT','nOT','nOT')
  training.transpose <- t(X.train)
  training.row.names <- rownames(X.train)
  #store training runs
  training.runs <- rbind(training.runs, training.row.names)
  ##autoscale data in training set
  trainmeansOT <- as.data.frame(colMeans(X.train)) #calculate means
  trainSDOT <- as.data.frame(apply(X.train,2, sd)) #calculate SDs
  train_meansubOT <- X.train[] - trainmeansOT[col(X.train[])] #subtract means from each cell
  train_divdSDOT <- train_meansubOT[] / trainSDOT[col(train_meansubOT[])] #divide each cell by the SD
  X.train <- train_divdSDOT
  #autoscale test data using training data means and SDs
  test_meansubOT <-  X.test[] - trainmeansOT[col(X.test[])]
  test_divdSDOT <- test_meansubOT[] / trainSDOT[col(test_meansubOT[])]
  X.test <- test_divdSDOT
  #autoscale consumer product data using training data means and SDs
  dried_meansubOT <-  TSdried[] - trainmeansOT[col(TSdried[])]
  dried_divdSDOT <- dried_meansubOT[] / trainSDOT[col(dried_meansubOT[])]
  TSPdried <- dried_divdSDOT
  #autoscale VS data using training data means and SDs
  VS_meansubOT <-  TSVS[] - trainmeansOT[col(TSVS[])]
  VS_divdSDOT <- VS_meansubOT[] / trainSDOT[col(VS_meansubOT[])]
  TSPVS <- VS_divdSDOT
  ##replace NA values with small number
  X.train[is.na(X.train)] = 0.001
  X.test[is.na(X.test)] = 0.0001
  TSPdried[is.na(TSPdried)] = 0.0001
  TSPVS[is.na(TSPVS)] = 0.0001
  ##replace 0 with small value
  X.train <- replace(X.train, X.train<0,0.001)
  X.test <- replace(X.test, X.test<0,0.0001)
  TSPdried <- replace(TSPdried, TSPdried<0,0.0001)
  TSPVS <- replace(TSPVS, TSPVS<0,0.0001)
  #remove features with zero variance 
  X.trainzvOT <- as.data.frame(nearZeroVar(X.train))
  X.testzvOT <- as.data.frame(nearZeroVar(X.test))
  tX.trainzvOT <- t(X.trainzvOT)
  colnames(tX.trainzvOT) <- row.names(X.trainzvOT)
  tX.testzvOT <- t(X.testzvOT)
  colnames(tX.testzvOT) <- row.names(X.testzvOT)
  #added this function so the subset only occurs if there are variables with zero variance
  is_empty <- function(x) {
    if (length(x) == 0 & !is.null(x)) {
      TRUE
    } else {
      FALSE
    }}
  train_empty <- is_empty(tX.trainzvOT)
  test_empty <- is_empty(tX.testzvOT)
  if(train_empty == "FALSE") {
    ixtrain <- which(colnames(X.train) %in% colnames(tX.trainzvOT))
    X.train <- subset(X.train, select = -ixtrain)
    X.test <- subset(X.test, select = -ixtrain)
    TSPdried <- subset(TSPdried, select = -ixtrain)
    TSPVS <- subset(TSVS, select = - ixtrain)
  }
  if(test_empty == "FALSE") {
    ixtest <- which(colnames(X.test) %in% colnames(tX.testzvOT))
    X.train <- subset(X.train, select = -ixtest)
    X.test <- subset(X.test, select = -ixtest)
    TSPdried <- subset(TSPdried, select = -ixtest)
    TSPVS <- subset(TSPVS, select = - ixtest)
  }
  #select optimal number of components and variables
  tune.splsda.L <- tune.splsda(X.train, Y.train, ncomp = 5,
                               validation = 'Mfold', folds = 5,
                               dist = 'max.dist', progressBar = FALSE,
                               measure = 'BER', test.keepX = list.keepX, nrepeat = 30,
                               cpus = 2)
  ncomp <- tune.splsda.L$choice.ncomp$ncomp 
  select.keepX <- tune.splsda.L$choice.keepX[1:ncomp]
 #store ncomps and selected variables
  ncomps.runs <- rbind(ncomps.runs, ncomp)
  keepx.runs <- rbind(keepx.runs, select.keepX)
  #build model with training set
  train.OT <- mixOmics::splsda(X.train, Y.train, ncomp = ncomp, keepX = select.keepX) ##make model with training set based on optimal comps and vars from model 
  #predict test set
  test.predict <- predict(train.OT, X.test, method = "max.dist") ##predict test set
  Prediction <- as.data.frame(test.predict[["MajorityVote"]][["max.dist"]])  #extract prediction 
  test.pred <- as.data.frame(cbind(Y = as.character(Y.test), Prediction$comp1)) #table with true and predicted values
  predictionasrow <- t(test.pred$V2)
  #store test set predictions
  predictionstorage <- rbind(predictionstorage, predictionasrow)
  #create confusion matrix to calculate validation scores for the test set 
  expected <- factor(test.pred$Y)  #define expected values
  predicted <- factor(Prediction$comp1) #define predicted values
  pred.matrix <- confusionMatrix(data=predicted, reference=expected, positive = "OT") #make confusion matrix
  pred.outcome <- as.matrix(pred.matrix, what = "classes") #print the validation results
  pred.row <- t(pred.outcome) ##make results into a tables
  results <- rbind(results, pred.row)
  #extract the specific features selected on the first two components
  comp = 1 # select which component you are inspecting
  features.to.view <- 10 # how many features do you want to look at
  loadingsX1 = abs(train.OT$loadings$X[, comp]) # extract the absolute loading values
  feat1 <- as.data.frame(t(sort(loadingsX1, decreasing = TRUE)[1:features.to.view]))
  feat1 <- melt(as.matrix(feat1))[-1]
  feat1 <- t(feat1$Var2)
  features <- as.data.frame(train.OT$loadings$X)
  #this function only runs if model used more than one component
  if("comp2" %in% colnames(features))
  {
    comp2 = 2 # select which component you are inspecting
    features.to.view <- 10 # how many features do you want to look at
    loadingsX2 = abs(train.OT$loadings$X[, comp2]) # extract the absolute loading values
    feat2 <- as.data.frame(t(sort(loadingsX2, decreasing = TRUE)[1:features.to.view]))
    feat2 <- melt(as.matrix(feat2))[-1]
    feat2 <- t(feat2$Var2);
  }
  features.PLSOT <- cbind(feat1, feat2)
  #store features
  OT.features.PLS <- rbind(OT.features.PLS, features.PLSOT)
  #predict consumer products based on train OT
  dried.predict <- predict(train.OT, TSPdried, method = "max.dist") ##predict test set
  dried.Prediction <- as.data.frame(dried.predict[["MajorityVote"]][["max.dist"]])  #extract prediction 
  dried.pred <- as.data.frame(cbind(Y = as.character(y.predict), dried.Prediction$comp1)) #table with true and predicted values
  tdried.pred <- as.data.frame(t(dried.Prediction$comp1))
  driedpredOT <- rbind(driedpredOT, tdried.pred)
  #make confusion matrix to calculate consumer product validation scores
  expecteddriedOT <- as.factor(y.predict)
  predicteddriedOT <- as.factor(dried.Prediction$comp1)
  pred.matrixdriedOT <- confusionMatrix(data=predicteddriedOT, reference=expecteddriedOT, positive= "OT") #make confusion matrix
  pred.outcomedriedOT <- as.matrix(pred.matrixdriedOT, what = "classes") #print the validation results
  pred.rowdriedOT <- t(pred.outcomedriedOT) ##make results into a tables
  resultsdriedOT <- rbind(resultsdriedOT, pred.rowdriedOT)
  #predict VS based on train OT
  VS.predict <- predict(train.OT, TSPVS, method = "max.dist") ##predict test set
  VS.Prediction <- as.data.frame(VS.predict[["MajorityVote"]][["max.dist"]])  #extract prediction 
  VS.pred <- as.data.frame(cbind(Y = as.character(y.VS), VS.Prediction$comp1)) #table with true and predicted values
  tVS.pred <- as.data.frame(t(VS.Prediction$comp1))
  VSpredOT <- rbind(VSpredOT, tVS.pred)
  #make confusion matrix to calcuate VS validation scores
  expectedVSOT <- as.factor(y.VS)
  predictedVSOT <- as.factor(VS.Prediction$comp1)
  pred.matrixVSOT <- confusionMatrix(data=predictedVSOT, reference=expectedVSOT, positive = "OT") #make confusion matrix
  pred.outcomeVSOT <- as.matrix(pred.matrixVSOT, what = "classes") #print the validation results
  pred.rowVSOT <- t(pred.outcomeVSOT) ##make results into a tables
  resultsVSOT <- rbind(resultsVSOT, pred.rowVSOT)
}

#set up is OB is not OB
XOB <- rbind(tOB, tNOTob)
YOB <- c('OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB','nOB', 'nOB')
#make empty data frame to store validation results from each run (test set)
resultsOB <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsOB) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                         'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store the training set for each run 
training.runsOB <- data.frame(matrix(ncol = 16, nrow = 0))
colnames(training.runsOB) <- c(paste0(1:16))
#make empty data frame to store the test set predictions for each run 
predictionstorageOB <- data.frame(matrix(ncol = 15, nrow = 0))
colnames(predictionstorageOB)  <- c('OB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB')
#make empty data frame to store ncomps for each run 
ncomps.runsOB <- data.frame(matrix(ncol = 2, nrow = 0))
colnames(ncomps.runsOB) <- c('ncomps')
#make empty data frame to store keepX for each run 
keepx.runsOB <- data.frame(matrix(ncol = 2, nrow = 0))
#load consumer product data
dried <-  read.csv("Consumer_data.csv", row.names = 1, header = TRUE)
tdried <- t(dried)
#transform data
TSdried <- sqrt(tdried)
##replace NA values with small number
TSdried[is.na(TSdried)] = 0.0001
##replace 0 with small value
TSdriedOB <- replace(TSdried, TSdried<0,0.0001)
y.predictOB <- c('nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'OB', 'OB', 'nOB', 'OB', 'nOB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB')
#make empty dataframe for consumer product predictions
driedpredOB <- data.frame(matrix(ncol = 17, nrow = 0))
resultsdriedOB <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsdriedOB) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                              'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
##set up validation set
VS <-  read.csv("VS_data.csv", row.names = 1, header = TRUE)
tVS <- t(VS)
#transform data
tsVS <- sqrt(tVS)
##replace NA values with small number
tsVS[is.na(tsVS)] = 0.0001
##replace 0 with small value
TSVS <- replace(tsVS, tsVS<0,0.0001)
y.VS <- c('nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB','OB')
#make empty data frame to store the predictions for each run 
VSpredOB <- data.frame(matrix(ncol = 9, nrow = 0))
colnames(VSpredOB)  <- c('nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB','nOB', 'OB')
#make empty data frame to store VS val scores
resultsVSOB <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsVSOB) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make dataframe to store selected variables
OB.features.PLS <- data.frame(matrix(ncol = 20, nrow = 0))

for (i in 1:100){
  set.seed(NULL)
#set up test and training sets
  data_set_size_OB <- floor(nrow(tOB)*0.80)
  index_OB <- sample(1:nrow(tOB), size = data_set_size_OB)
  data_set_size_notOB <- floor(nrow(tOB)*0.80)
  index_notOB <- sample(1:nrow(tNOTob), size = data_set_size_notOB)
  train_OB <- tOB[index_OB,]
  test_OB <- tOB[-index_OB,]
  train_nOB <- tNOTob[index_OB,]
  test_nOB <- tNOTob[-index_OB,]
  X.trainOB <- as.data.frame(rbind(train_OB, train_nOB))
  X.testOB <- as.data.frame(rbind(test_OB, test_nOB))
  Y.trainOB <- c('OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB')
  Y.testOB <- c('OB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB')
  training.transposeOB <- t(X.trainOB)
  training.row.namesOB <- rownames(X.trainOB)
  training.runsOB <- rbind(training.runs, training.row.namesOB)
  ##autoscale data for trainin set
  trainmeansOB <- as.data.frame(colMeans(X.trainOB)) #calculate means
  trainSDOB <- as.data.frame(apply(X.trainOB,2, sd)) #calculate SDs
  train_meansubOB <- X.trainOB[] - trainmeansOB[col(X.trainOB[])] #subtract means from each cell
  train_divdSDOB <- train_meansubOB[] / trainSDOB[col(train_meansubOB[])] #divide each cell by the SD
  X.trainOB <- train_divdSDOB
  #autoscale test data using training data means and SDs
  test_meansubOB <-  X.testOB[] - trainmeansOB[col(X.testOB[])]
  test_divdSDOB <- test_meansubOB[] / trainSDOB[col(test_meansubOB[])]
  X.testOB <- test_divdSDOB
  ##replace NA values with small number
  X.trainOB[is.na(X.trainOB)] = 0.001
  X.testOB[is.na(X.testOB)] = 0.0001
  ##replace 0 with small value
  X.trainOB <- replace(X.trainOB, X.trainOB<0,0.001)
  X.testOB <- replace(X.testOB, X.testOB<0,0.0001)
  #remove features with zero variance 
  X.trainzvOB <- as.data.frame(nearZeroVar(X.trainOB))
  X.testzvOB <- as.data.frame(nearZeroVar(X.testOB))
  tX.trainzvOB <- t(X.trainzvOB)
  tX.testzvOB <- t(X.testzvOB)
  ixtrain <- which(colnames(X.trainOB) %in% colnames(tX.trainzvOB))
  ixtest <- which(colnames(X.testOB) %in% colnames(tX.testzvOB))
  X.trainOB <- subset(X.trainOB, select = -ixtrain)
  X.trainOB <-subset(X.trainOB, select = -ixtest)
  X.testOB <- subset(X.testOB, select = -ixtrain)
  X.testOB <- subset(X.testOB, select = -ixtest)
  #select optimal number of components and variables
  tune.splsda.LOB <- tune.splsda(X.trainOB, Y.trainOB, ncomp = 5,
                                 validation = 'Mfold', folds = 5,
                                 dist = 'max.dist', progressBar = FALSE,
                                 measure = 'BER', test.keepX = list.keepX,
                                 nrepeat = 30, cpus = 2)
  ncompOB <- tune.splsda.LOB$choice.ncomp$ncomp 
  select.keepXOB <- tune.splsda.LOB$choice.keepX[1:ncompOB]
  #store ncomp and selected variables
  ncomps.runsOB <- rbind(ncomps.runsOB, ncompOB)
  keepx.runsOB <- rbind(keepx.runsOB, select.keepXOB)
  #build model with training set
  train.OB <- mixOmics::splsda(X.trainOB, Y.trainOB, ncomp = ncompOB, keepX = select.keepXOB) ##make model with training set based on optimal comps and vars from model 
  test.predictOB <- predict(train.OB, X.testOB, method = "max.dist") ##predict test set
  PredictionOB <- as.data.frame(test.predictOB[["MajorityVote"]][["max.dist"]])  #extract prediction 
  test.predOB <- as.data.frame(cbind(Y = as.character(Y.testOB), PredictionOB$comp1)) #table with true and predicted values
  predictionasrowOB <- t(test.predOB$V2)
  #store test set predictions
  predictionstorageOB <- rbind(predictionstorageOB, predictionasrowOB)
  #make confusion matrix to calculate test set validation scores
  expectedOB <- factor(test.predOB$Y)  #define expected values
  predictedOB <- factor(PredictionOB$comp1) #define predicted values
  pred.matrixOB <- confusionMatrix(data=predictedOB, reference=expectedOB, positive = "OB") #make confusion matrix
  pred.outcomeOB <- as.matrix(pred.matrixOB, what = "classes") #print the validation results
  pred.rowOB <- t(pred.outcomeOB) ##make results into a tables
  resultsOB <- rbind(resultsOB, pred.rowOB)
  #extract selected variabels for first two components
  comp = 1 # select which component you are inspecting
  features.to.view <- 10 # how many features do you want to look at
  loadingsX1 = abs(train.OB$loadings$X[, comp]) # extract the absolute loading values
  feat1 <- as.data.frame(t(sort(loadingsX1, decreasing = TRUE)[1:features.to.view]))
  feat1 <- melt(as.matrix(feat1))[-1]
  feat1 <- t(feat1$Var2)
  features <- as.data.frame(train.OB$loadings$X)
  if("comp2" %in% colnames(features))
  {
    comp2 = 2 # select which component you are inspecting
    features.to.view <- 10 # how many features do you want to look at
    loadingsX2 = abs(train.OB$loadings$X[, comp2]) # extract the absolute loading values
    feat2 <- as.data.frame(t(sort(loadingsX1, decreasing = TRUE)[1:features.to.view]))
    feat2 <- melt(as.matrix(feat2))[-1]
    feat2 <- t(feat2$Var2);
  }
  features.PLSOB <- cbind(feat1, feat2)
  #store features
  OB.features.PLS <- rbind(OB.features.PLS, features.PLSOB)
  #autoscale consumer product data using training data means and SDs
  dried_meansubOB <-  TSdriedOB[] - trainmeansOB[col(TSdriedOB[])]
  dried_divdSDOB <- dried_meansubOB[] / trainSDOB[col(dried_meansubOB[])]
  TSPdried <- dried_divdSDOB
  rmvdvars <- which(colnames(TSPdried) %in% colnames(X.testOB))
  TSPdried <- subset(TSPdried, select = rmvdvars)
  #predict consumer products
  dried.predictOB <- predict(train.OB, TSPdried, method = "max.dist") ##predict test set
  dried.PredictionOB <- as.data.frame(dried.predictOB[["MajorityVote"]][["max.dist"]])  #extract prediction 
  dried.predOB <- as.data.frame(cbind(Y = as.character(y.predictOB), dried.PredictionOB$comp1)) #table with true and predicted values
  #make confusion matrix to calculate consumer product validation scores
  expecteddriedOB <- as.factor(y.predictOB)
  predicteddriedOB <- as.factor(dried.PredictionOB$comp1)
  pred.matrixdriedOB <- confusionMatrix(data=predicteddriedOB, reference=expecteddriedOB, positive = "OB") #make confusion matrix
  pred.outcomedriedOB <- as.matrix(pred.matrixdriedOB, what = "classes") #print the validation results
  pred.rowdriedOB <- t(pred.outcomedriedOB) ##make results into a tables
  #store consumer product predictions and validation scores
  resultsdriedOB <- rbind(resultsdriedOB, pred.rowdriedOB)
  driedpredOB <- rbind(driedpredOB, predicteddriedOB)
  #autoscale VS data using training data means and SDs
  VS_meansubOB <-  TSVS[] - trainmeansOB[col(TSVS[])]
  VS_divdSDOB <- VS_meansubOB[] / trainSDOB[col(VS_meansubOB[])]
  TSPVS <- VS_divdSDOB
  rmvdvarsVS <- which(colnames(TSPVS) %in% colnames(X.testOB))
  TSPVS <- subset(TSPVS, select = rmvdvarsVS)
  #predict VS
  VS.predictOB <- predict(train.OB, TSPVS, method = "max.dist") ##predict test set
  VS.PredictionOB <- as.data.frame(VS.predictOB[["MajorityVote"]][["max.dist"]])  #extract prediction 
  VS.predOB <- as.data.frame(cbind(Y = as.character(y.VS), VS.PredictionOB$comp1)) #table with true and predicted values
  #make confusion matrix to calculate validation set validation scores
  expectedVSOB <- as.factor(y.VS)
  predictedVSOB <- as.factor(VS.PredictionOB$comp1)
  pred.matrixVSOB <- confusionMatrix(data=predictedVSOB, reference=expectedVSOB, positive = "OB") #make confusion matrix
  pred.outcomeVSOB <- as.matrix(pred.matrixVSOB, what = "classes") #print the validation results
  pred.rowVSOB <- t(pred.outcomeVSOB) ##make results into a tables
  #store VS predictions and validation scores
  resultsVSOB <- rbind(resultsVSOB, pred.rowVSOB)
  VSpredOB <- rbind(VSpredOB, predictedVSOB)
  
}

#set up is OG is not OG
XOG <- rbind(tOG, tNOTog)
YOG <- c('OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'nOG', 'nOG', 'nOG', 
         'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG',
         'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
##make empty data frame to store validation results from each run (test set)
resultsOG <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsOG) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                         'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
##make empty data frame to store the training set for each run 
training.runsOG <- data.frame(matrix(ncol = 12, nrow = 0))
colnames(training.runsOG) <- c(paste0(1:12))
##make empty data frame to store the predictions for each run 
predictionstorageOG <- data.frame(matrix(ncol = 19, nrow = 0))
colnames(predictionstorageOG)  <- c('OG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 
                                    'nOG', 'nOG','nOG','nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
##make empty data frame to store ncomps for each run 
ncomps.runsOG <- data.frame(matrix(ncol = 2, nrow = 0))
colnames(ncomps.runsOG) <- c('ncomps')
##make empty data frame to store keepX for each run 
keepx.runsOG <- data.frame(matrix(ncol = 2, nrow = 0))
##load consumer product data
dried <-  read.csv("Consumer_data.csv", row.names = 1, header = TRUE)
tdried <- t(dried)
##transform data
TSdried <- sqrt(tdried)
##replace NA values with small number
TSdried[is.na(TSdried)] = 0.0001
##replace 0 with small value
TSdriedOG <- replace(TSdried, TSdried<0,0.0001)
y.predictOG <- c('nOG', 'nOG', 'OG', 'nOG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'OG', 'nOG','nOG', 'OG')
##make dataframe to store consumer product set validation scores
resultsdriedOG <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsdriedOG) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                              'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
##make empty dataframe for consumer product predictions
driedpredOG <- data.frame(matrix(ncol = 17, nrow = 0))
##set up validation set 
VS <-  read.csv("VS_data.csv", row.names = 1, header = TRUE)
tVS <- t(VS)
##transform data
tsVS <- sqrt(tVS)
##replace NA values with small number
tsVS[is.na(tsVS)] = 0.0001
##replace 0 with small value
TSVS <- replace(tsVS, tsVS<0,0.0001)
y.VS <- c('OG', 'nOG', 'nOG', 'OG', 'nOG', 'nOG','nOG','OG', 'nOG')
##make empty data frame to store the VS predictions for each run 
VSpredOG <- data.frame(matrix(ncol = 9, nrow = 0))
colnames(VSpredOG)  <- c('OG', 'nOG', 'nOG', 'OG', 'nOG','nOG','nOG', 'OG', 'nOG')
##make empty data frame to store VS val scores
resultsVSOG <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsVSOG) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
##data frame for storing variables
OG.features.PLS <- data.frame(matrix(ncol = 20, nrow = 0))

for (i in 1:100){
  set.seed(NULL)
 #set up test and training set
  data_set_size_OG <- floor(nrow(tOG)*0.80)
  index_OG <- sample(1:nrow(tOG), size = data_set_size_OG)
  data_set_size_notOG <- floor(nrow(tOG)*0.80)
  index_notOG <- sample(1:nrow(tNOTog), size = data_set_size_notOG)
  train_OG <- tOG[index_OG,]
  test_OG <- tOG[-index_OG,]
  train_nOG <- tNOTog[index_OG,]
  test_nOG <- tNOTog[-index_OG,]
  X.trainOG <- as.data.frame(rbind(train_OG, train_nOG))
  X.testOG <- as.data.frame(rbind(test_OG, test_nOG))
  Y.trainOG <- c('OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
  Y.testOG <- c('OG', 'OG', 'nOG', 'nOG','nOG','nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
  training.transposeOG <- t(X.trainOG)
  training.row.namesOG <- rownames(X.trainOG)
  training.runsOG <- rbind(training.runsOG, training.row.namesOG)
  ##autoscale data in training set
  trainmeansOG <- as.data.frame(colMeans(X.trainOG)) #calculate means
  trainSDOG <- as.data.frame(apply(X.trainOG,2, sd)) #calculate SDs
  train_meansubOG <- X.trainOG[] - trainmeansOG[col(X.trainOG[])] #subtract means from each cell
  train_divdSDOG <- train_meansubOG[] / trainSDOG[col(train_meansubOG[])] #divide each cell by the SD
  X.trainOG <- train_divdSDOG
  #autoscale test data using training data means and SDs
  test_meansubOG <-  X.testOG[] - trainmeansOG[col(X.testOG[])]
  test_divdSDOG <- test_meansubOG[] / trainSDOG[col(test_meansubOG[])]
  X.testOG <- test_divdSDOG
  ##replace NA values with small number
  X.trainOG[is.na(X.trainOG)] = 0.001
  X.testOG[is.na(X.testOG)] = 0.0001
  ##replace 0 with small value
  X.trainOG <- replace(X.trainOG, X.trainOG<0,0.001)
  X.testOG <- replace(X.testOG, X.testOG<0,0.0001)
  #remove features with zero variance 
  X.trainzvOG <- as.data.frame(nearZeroVar(X.trainOG))
  tX.trainzvOG <- t(X.trainzvOG)
  X.testzvOG <- as.data.frame(nearZeroVar(X.testOG))
  tX.testzvOG <- t(X.testzvOG)
  ixtrain <- which(colnames(X.trainOG) %in% colnames(tX.trainzvOG))
  ixtest <- which(colnames(X.testOG) %in% colnames(tX.testzvOG))
  X.trainOG <- subset(X.trainOG, select = -ixtrain)
  X.testOG <- subset(X.testOG, select = -ixtrain)
  X.trainOG <- subset(X.trainOG, select = -ixtest)
  X.testOG <- subset(X.testOG, select = -ixtest)
  #tselect optimal components and variables
  tune.splsda.LOG <- tune.splsda(X.trainOG, Y.trainOG, ncomp = 5,
                                 validation = 'Mfold', folds = 5,
                                 dist = 'max.dist', progressBar = FALSE,
                                 measure = 'BER', test.keepX = list.keepX,
                                 nrepeat = 30, cpus = 2)
  ncompOG <- tune.splsda.LOG$choice.ncomp$ncomp 
  select.keepXOG <- tune.splsda.LOG$choice.keepX[1:ncompOG]
  #store ncomps and selected variables
  ncomps.runsOG <- rbind(ncomps.runsOG, ncompOG)
  keepx.runsOG <- rbind(keepx.runsOG, select.keepXOG)
  #build model with training set
  train.OG <- mixOmics::splsda(X.trainOG, Y.trainOG, ncomp = ncompOG, keepX = select.keepXOG) ##make model with training set based on optimal comps and vars from model 
  test.predictOG <- predict(train.OG, X.testOG, method = "max.dist") ##predict test set
  PredictionOG <- as.data.frame(test.predictOG[["MajorityVote"]][["max.dist"]])  #extract prediction 
  test.predOG <- as.data.frame(cbind(Y = as.character(Y.testOG), PredictionOG$comp1)) #table with true and predicted values
  predictionasrowOG <- t(test.predOG$V2)
  #store test predicitons
  predictionstorageOG <- rbind(predictionstorageOG, predictionasrowOG)
  #make confusion matrix to calculate test set validation scores
  expectedOG <- factor(test.predOG$Y)  #define expected values
  predictedOG <- factor(PredictionOG$comp1) #define predicted values
  pred.matrixOG <- confusionMatrix(data=predictedOG, reference=expectedOG, positive = "OG") #make confusion matrix
  pred.outcomeOG <- as.matrix(pred.matrixOG, what = "classes") #print the validation results
  pred.rowOG <- t(pred.outcomeOG) ##make results into a tables
  resultsOG <- rbind(resultsOG, pred.rowOG)
  #extract variables selcted on first two components
  comp = 1 # select which component you are inspecting
  features.to.view <- 10 # how many features do you want to look at
  loadingsX1 = abs(train.OG$loadings$X[, comp]) # extract the absolute loading values
  feat1 <- as.data.frame(t(sort(loadingsX1, decreasing = TRUE)[1:features.to.view]))
  feat1 <- melt(as.matrix(feat1))[-1]
  feat1 <- t(feat1$Var2)
  feature <- as.data.frame(train.OG$loadings$X)
  if("comp2" %in% colnames(feature))
  {
    comp2 = 2 # select which component you are inspecting
    features.to.view <- 1 # how many features do you want to look at
    loadingsX2 = abs(train.OG$loadings$X[, comp2]) # extract the absolute loading values
    feat2 <- as.data.frame(t(sort(loadingsX1, decreasing = TRUE)[1:features.to.view]))
    feat2 <- melt(as.matrix(feat2))[-1]
    feat2 <- t(feat2$Var2);
  }
  features.PLSOG <- cbind(feat1, feat2)
  OG.features.PLS <- rbind(OG.features.PLS, features.PLSOG)
  #autoscale consumer product data using training data means and SDs
  dried_meansubOG <-  TSdriedOG[] - trainmeansOG[col(TSdriedOG[])]
  dried_divdSDOG <- dried_meansubOG[] / trainSDOG[col(dried_meansubOG[])]
  TSPdried <- dried_divdSDOG
  rmvdvars <- which(colnames(TSPdried) %in% colnames(X.testOG))
  TSPdried <- subset(TSPdried, select = rmvdvars)
  #predict consumer products
  dried.predictOG <- predict(train.OG, TSPdried, method = "max.dist") ##predict test set
  dried.PredictionOG <- as.data.frame(dried.predictOG[["MajorityVote"]][["max.dist"]])  #extract prediction 
  dried.predOG <- as.data.frame(cbind(Y = as.character(y.predictOG), dried.PredictionOG$comp1)) #table with true and predicted values
  #make confusion matrix to calculate consumer product validation scores
  expecteddriedOG <- as.factor(y.predictOG)
  predicteddriedOG <- as.factor(dried.PredictionOG$comp1)
  pred.matrixdriedOG <- confusionMatrix(data=predicteddriedOG, reference=expecteddriedOG, positive = "OG") #make confusion matrix
  pred.outcomedriedOG <- as.matrix(pred.matrixdriedOG, what = "classes") #print the validation results
  pred.rowdriedOG <- t(pred.outcomedriedOG) ##make results into a tables
  driedpredOG <- rbind(driedpredOG, dried.PredictionOG$comp1)
  resultsdriedOG <- rbind(resultsdriedOG, pred.rowdriedOG)
  #autoscale VS data using training data means and SDs
  VS_meansubOG <-  TSVS[] - trainmeansOG[col(TSVS[])]
  VS_divdSDOG <- VS_meansubOG[] / trainSDOG[col(VS_meansubOG[])]
  TSPVS <- VS_divdSDOG
  rmvdvarsVS <- which(colnames(TSPVS) %in% colnames(X.testOG))
  TSPVS <- subset(TSPVS, select = rmvdvarsVS)
  #predict VS
  VS.predictOG <- predict(train.OG, TSPVS, method = "max.dist") ##predict test set
  VS.PredictionOG <- as.data.frame(VS.predictOG[["MajorityVote"]][["max.dist"]])  #extract prediction 
  VS.predOG <- as.data.frame(cbind(Y = as.character(y.VS), VS.PredictionOG$comp1)) #table with true and predicted values
  #make confusion matrix to calculate consumer product validation scores
  expectedVSOG <- as.factor(y.VS)
  predictedVSOG <- as.factor(VS.PredictionOG$comp1)
  pred.matrixVSOG <- confusionMatrix(data=predictedVSOG, reference=expectedVSOG, positive = "OG") #make confusion matrix
  pred.outcomeVSOG <- as.matrix(pred.matrixVSOG, what = "classes") #print the validation results
  pred.rowVSOG <- t(pred.outcomeVSOG) ##make results into a tables
  #store VS predictions and Validation scores
  VSpredOG <- rbind(VSpredOG, VS.PredictionOG$comp1)
  resultsVSOG <- rbind(resultsVSOG, pred.rowVSOG)
}


##RANDOM FOREST MODEL##

#set up is OT is not OT
##set up is OT (1) is not OT (2)
Xot <- rbind(tOT, tNOTot) ## all data 
Yot <- c( 'OT', 'OT', 'OT', 'OT', 'OT', 'OT', 'OT', 'OT',, 'OT', 'OT', 'OT', 'OT', 'OT',
          'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT','nOT', 'nOT',
          'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT')
Yot <- as.factor(Yot)
Yot <- ifelse(Yot == "OT", 1, 2)
##replace NA values with small number
Xot[is.na(Xot)] = 0.001
##replace 0 with small value
Xot <- replace(Xot, Xot<0,0.001)
#Make empty data frame to store OOB error for each run
OOBerror <- data.frame(matrix(ncol = 1, nrow = 0))
#make empty data frame to store validation results from each run (test set)
testresults <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(testresults) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store bootstrap predictions
BSpredictionstorage <- data.frame(matrix(ncol = 12, nrow = 0))
colnames(BSpredictionstorage) <- c('OT', 'OT', 'OT', 'OT', 'OT', 'OT',, 'OT', 'OT', 'OT', 'OT',
                                   'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT')
#make empty data frame to strore bootstrap validation scores
BSresults <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(BSresults) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                         'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store the training set for each run 
training.runs <- data.frame(matrix(ncol = 20, nrow = 0))
colnames(training.runs) <- c(paste0(1:20))
#make empty data frame to store the test predictions for each run 
testpredictionstorage <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(testpredictionstorage)  <- c('OT', 'OT', 'OT' ,'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 
                                      'nOT', 'nOT', 'nOT')
#make empty data frame to store the consumer product predictions for each run 
driedpredictionstorage <- data.frame(matrix(ncol = 17, nrow = 0))
colnames(driedpredictionstorage)  <- c('OT', 'OT', 'nOT', 'OT', 'OT', 'OT', 'OT', 'nOT', 'nOT', 'OT', 'nOT', 'OT', 'nOT', 'OT', 'nOT', 'OT','OT')
#make empty data frame to store consumer product val scores
resultsdriedOT <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsdriedOT) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                              'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty matrix to store variables selected in each run
OTvarimportance <- as.data.frame(matrix(ncol = 1123, nrow = 0))
colnames(OTvarimportance) <- c(colnames(Xot))
#load consumer product data
dried <-  read.csv("Consumer_data.csv", row.names = 1, header = TRUE)
tdried <- t(dried)
#transform data
TSdried <- sqrt(tdried)
##replace NA values with small number
TSdried[is.na(TSdried)] = 0.0001
##replace 0 with small value
TSdried <- replace(TSdried, TSdried<0,0.0001)
driedclass <- c('OT', 'OT', 'nOT', 'OT', 'nOT', 'OT', 'OT', 'nOT', 'nOT', 'OT', 'nOT', 'OT', 'nOT', 'OT', 'nOT', 'OT','OT')
driedclass <- as.factor(ifelse(driedclass == "OT", 1, 2))
##set up validation set
VS <-  read.csv("VS_data.csv", row.names = 1, header = TRUE)
tVS <- t(VS)
#transform data
tsVS <- sqrt(tVS)
##replace NA values with small number
tsVS[is.na(tsVS)] = 0.0001
##replace 0 with small value
TSVS <- replace(tsVS, tsVS<0,0.0001)
y.VS <- c('nOT', 'OT', 'OT', 'nOT', 'nOT','nOT', 'OT', 'OT', 'nOT')
VSclass <- y.VS
VSclass <- as.factor(ifelse(VSclass == "OT",1,2))
#make empty data frame to store the VS predictions for each run 
VSpredictionstorage <- data.frame(matrix(ncol = 9, nrow = 0))
colnames(VSpredictionstorage)  <- c('nOT', 'OT', 'OT', 'nOT', 'nOT','nOT','OT', 'OT', 'nOT')
#make empty data frame to store VS val scores
resultsVSOT <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsVSOT) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')

for(i in 1:100) {
  set.seed(NULL)
  #make test and training sets
  data_set_size_OT <- floor(nrow(tOT)*0.8)
  index_OT <- sample(1:nrow(tOT), size = data_set_size_OT)
  data_set_size_notOT <- floor(nrow(tOT)*0.8)
  index_notOT <- sample(1:nrow(tNOTot), size = data_set_size_notOT)
  train_OT <- tOT[index_OT,]
  test_OT <- tOT[-index_OT,]
  train_nOT <- tNOTot[index_OT,]
  test_nOT <- tNOTot[-index_OT,]
  X.train <- as.data.frame(rbind(train_OT, train_nOT))
  X.test <- as.data.frame(rbind(test_OT, test_nOT))
  #store training set for reference
  training.transpose <- t(X.train)
  training.row.names <- rownames(X.train)
  training.runs <- rbind(training.runs, training.row.names)
  #set up Y variables 
  Y.train <- c('OT', 'OT', 'OT', 'OT', 'OT','OT', ,'OT','OT','OT','OT','nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT')
  Y.test <- c('OT', 'OT','OT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT', 'nOT','nOT')
  Y.train <-  ifelse(Y.train == "OT", 1, 2)
  Y.train <- as.factor(Y.train)
  Y.test <-  ifelse(Y.test == "OT", 1, 2)
  Y.test <- as.factor(Y.test)
  ##autoscale training data
  trainmeansOT <- as.data.frame(colMeans(X.train)) #calculate means
  trainSDOT <- as.data.frame(apply(X.train,2, sd)) #calculate SDs
  train_meansubOT <- X.train[] - trainmeansOT[col(X.train[])] #subtract means from each cell
  train_divdSDOT <- train_meansubOT[] / trainSDOT[col(train_meansubOT[])] #divide each cell by the SD
  X.train <- train_divdSDOT
  #autoscale test data using training data means and SDs
  test_meansubOT <-  X.test[] - trainmeansOT[col(X.test[])]
  test_divdSDOT <- test_meansubOT[] / trainSDOT[col(test_meansubOT[])]
  X.test <- test_divdSDOT
  #autoscale consumer product data using training data means and SDs
  dried_meansub <-  TSdried[] - trainmeansOT[col(TSdried[])]
  dried_divdSD <- dried_meansub[] / trainSDOT[col(dried_meansub[])]
  TSPdried <- dried_divdSD
  #autoscale VS data using training data means and SDs
  VS_meansub <-  TSVS[] - trainmeansOT[col(TSVS[])]
  VS_divdSD <- VS_meansub[] / trainSDOT[col(VS_meansub[])]
  TSPVS <- VS_divdSD
  #remove infinate values
  X.train[sapply(X.train, is.infinite)] <- NA
  X.test[sapply(X.test, is.infinite)] <- NA
  TSPdried[sapply(TSPdried, is.infinite)] <- NA
  TSPVS[sapply(TSPVS, is.infinite)] <- NA
  ##replace NA values with small number
  X.train[is.na(X.train)] = 0.001
  X.test[is.na(X.test)] = 0.001
  TSPdried[is.na(TSPdried)] = 0.001
  TSPVS[is.na(TSPVS)] = 0.001
  ##replace 0 with small value
  X.train <- replace(X.train, X.train<0,0.001)
  X.test <- replace(X.test, X.test<0,0.001)
  TSPdried <- replace(TSPdried, TSPdried<0,0.001)
  TSPVS <- replace(TSPVS, TSPVS<0,0.001)
  #rename test and training set
  X.trainRF <- X.train
  Y.trainRF <- Y.train
  X.testRF <- X.test
  #build model with training set
  classifier_RF = randomForest(x = X.trainRF,
                               y = Y.trainRF,
                               ntree = 500)
  # store at OOB error
  error <- mean(classifier_RF$err.rate[,1])
  OOBerror <- rbind(OOBerror, error)
  #make matrix of expected and predicted values from training bootstrap repeats
  expectedbs <- Y.trainRF
  predictedbs <-  classifier_RF[["predicted"]]
  bspred.matrixOT <- confusionMatrix(data=predictedbs, reference=expectedbs) #make confusion matrix
  bspred.outcomeOT <- as.matrix(bspred.matrixOT, what = "classes") #print the validation results
  bspred.rowOT <- t(bspred.outcomeOT) ##make results into a tables
  #store bootstrap predictions and validation scores
  BSpredictionstorage <- rbind(BSpredictionstorage, predictedbs)
  BSresults <- rbind(BSresults, bspred.rowOT)
  #store variable importance
  varimportanceOT <- as.data.frame(t(importance(classifier_RF)))
  OTvarimportance <- rbind(OTvarimportance, varimportanceOT)
  # Predicting the Test set
  y_pred <- predict(classifier_RF, X.testRF)
  expectedtest <- Y.test
  predictedtest <- y_pred
  testpred.matrixOT <- confusionMatrix(data=predictedtest, reference=expectedtest, positive = "1") #make confusion matrix
  testpred.outcomeOT <- as.matrix(testpred.matrixOT, what = "classes") #print the validation results
  testpred.rowOT <- t(testpred.outcomeOT) ##make results into a tables
  #store test set predictions and validation scores
  testpredictionstorage <- rbind(testpredictionstorage, predictedtest)
  testresults <- rbind(testresults, testpred.rowOT)
  #predicting consumer products
  dried_pred = predict(classifier_RF, newdata = TSPdried)
  expecteddried <- driedclass
  predicteddried <- dried_pred
  driedpred.matrixOT <- confusionMatrix(data=predicteddried, reference=expecteddried, positive = "1") #make confusion matrix
  driedpred.outcomeOT <- as.matrix(driedpred.matrixOT, what = "classes") #print the validation results
  driedpred.rowOT <- t(driedpred.outcomeOT) ##make results into a tables
  #store consumer predictions and validation scores
  driedpredictionstorage <- rbind(driedpredictionstorage, predicteddried)
  resultsdriedOT <- rbind(resultsdriedOT, driedpred.rowOT)
  #predicting VS products
  VS_pred = predict(classifier_RF, newdata = TSPVS)
  expectedVS <- VSclass
  predictedVS <- VS_pred
  VSpred.matrixOT <- confusionMatrix(data=predictedVS, reference=expectedVS, positive = "1") #make confusion matrix
  VSpred.outcomeOT <- as.matrix(VSpred.matrixOT, what = "classes") #print the validation results
  VSpred.rowOT <- t(VSpred.outcomeOT) ##make results into a tables
  #store VS predictions and validation scores
  VSpredictionstorage <- rbind(VSpredictionstorage, predictedVS)
  resultsVSOT <- rbind(resultsVSOT, VSpred.rowOT)
  
}

#set up is OB is not OB
##set up is OB (1) is not OB (2)
Xob <- rbind(tOB, tNOTob) ## all data 
Yob <- c('OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB','nOB', 'nOB')
Yob <- as.factor(Yob)
Yob <- ifelse(Yob == "OB", 1, 2)
##replace NA values with small number
Xob[is.na(Xob)] = 0.001
##replace 0 with small value
Xob <- replace(Xob, Xob<0,0.001)
#Make empty data frame to store OOB error
OOBerrorOB <- data.frame(matrix(ncol = 1, nrow = 0))
#make empty data frame to store validation results from each run (test set)
testresultsOB <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(testresultsOB) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                             'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store bootstrap predictions
BSpredictionstorageOB <- data.frame(matrix(ncol = 16, nrow = 0))
colnames(BSpredictionstorageOB) <- c('OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB')
#make empty data frame to strore bootstrap validation scores
BSresultsOB <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(BSresultsOB) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store the training set for each run 
training.runsOB <- data.frame(matrix(ncol = 16, nrow = 0))
colnames(training.runsOB) <- c(paste0(1:16))
#make empty data frame to store the test predictions for each run 
testpredictionstorageOB <- data.frame(matrix(ncol = 15, nrow = 0))
colnames(testpredictionstorageOB)  <- c('OB', 'OB', 'nOB', 'nOB','nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB')
#make empty data frame to store the consumer product predictions for each run 
driedpredictionstorageOB <- data.frame(matrix(ncol = 17, nrow = 0))
colnames(driedpredictionstorageOB)  <- c('nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'OB', 'OB', 'nOB', 'OB', 'nOB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB')
#make empty data frame to store dried val scores
resultsdriedOB <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsdriedOB) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                              'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty matrix to store selected variables
OBvarimportance <- as.data.frame(matrix(ncol = 1123, nrow = 0))
colnames(OBvarimportance) <- c(colnames(Xob))
#load consumer product data
dried <-  read.csv("Consumer_data.csv", row.names = 1, header = TRUE)
tdried <- t(dried)
#transform data
TSdried <- sqrt(tdried)
##replace NA values with small number
TSdried[is.na(TSdried)] = 0.0001
##replace 0 with small value
TSdried <- as.data.frame(replace(TSdried, TSdried<0,0.0001))
driedclass <- c('nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'OB', 'OB', 'nOB', 'OB', 'nOB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB')
driedclass <- as.factor(ifelse(driedclass == "OB", 1, 2))
##set up validation set 
VS <-  read.csv("VS_data.csv", row.names = 1, header = TRUE)
tVS <- t(VS)
#transform data
tsVS <- sqrt(tVS)
##replace NA values with small number
tsVS[is.na(tsVS)] = 0.0001
##replace 0 with small value
TSVS <- replace(tsVS, tsVS<0,0.0001)
y.VS <- c('nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','OB')
VSclass <- y.VS
VSclass <- as.factor(ifelse(VSclass == "OB",1,2))
#make empty data frame to store the predictions for each run 
VSpredictionstorageOB <- data.frame(matrix(ncol = 9, nrow = 0))
colnames(VSpredictionstorage)  <- c('nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB', 'nOB', 'OB')
#make empty data frame to store validation set val scores
resultsVSOB <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsVSOT) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
for(i in 1:100) {
  set.seed(NULL)
  data_set_size_OB <- floor(nrow(tOB)*0.80)
  index_OB <- sample(1:nrow(tOB), size = data_set_size_OB)
  data_set_size_notOB <- floor(nrow(tOB)*0.80)
  index_notOB <- sample(1:nrow(tNOTob), size = data_set_size_notOB)
  train_OB <- tOB[index_OB,]
  test_OB <- tOB[-index_OB,]
  train_nOB <- tNOTob[index_OB,]
  test_nOB <- tNOTob[-index_OB,]
  X.trainOB <- as.data.frame(rbind(train_OB, train_nOB))
  X.testOB <- as.data.frame(rbind(test_OB, test_nOB))
  Y.trainOB <- c('OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'OB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB', 'nOB')
  Y.testOB <- c('OB', 'OB', 'nOB', 'nOB', 'nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB','nOB', 'nOB', 'nOB', 'nOB', 'nOB')
  training.transposeOB <- t(X.trainOB)
  training.row.namesOB <- rownames(X.trainOB)
  training.runsOB <- rbind(training.runsOB, training.row.namesOB)
  Y.trainOB <-  ifelse(Y.trainOB == "OB", 1, 2)
  Y.trainOB <- as.factor(Y.trainOB)
  Y.testOB <-  ifelse(Y.testOB == "OB", 1, 2)
  Y.testOB <- as.factor(Y.testOB)
  ##autoscale data in training set
  trainmeansOB <- as.data.frame(colMeans(X.trainOB)) #calculate means
  trainSDOB <- as.data.frame(apply(X.trainOB,2, sd)) #calculate SDs
  train_meansubOB <- X.trainOB[] - trainmeansOB[col(X.trainOB[])] #subtract means from each cell
  train_divdSDOB <- train_meansubOB[] / trainSDOB[col(train_meansubOB[])] #divide each cell by the SD
  X.trainOB <- train_divdSDOB
  #autoscale test data using training data means and SDs
  test_meansubOB <-  X.testOB[] - trainmeansOB[col(X.testOB[])]
  test_divdSDOB <- test_meansubOB[] / trainSDOB[col(test_meansubOB[])]
  X.testOB <- test_divdSDOB
  #autoscale consumer prodcut data using training data means and SDs
  dried_meansubOB <-  TSdried[] - trainmeansOB[col(TSdried[])]
  dried_divdSDOB <- dried_meansubOB[] / trainSDOB[col(dried_meansubOB[])]
  TSPdried <- dried_divdSDOB
  #autoscale VS data using training data means and SDs
  VS_meansubOB <-  TSVS[] - trainmeansOB[col(TSVS[])]
  VS_divdSDOB <- VS_meansubOB[] / trainSDOB[col(VS_meansubOB[])]
  TSPVS <- VS_divdSDOB
  #remove infinate values
  X.trainOB[sapply(X.trainOB, is.infinite)] <- NA
  X.testOB[sapply(X.testOB, is.infinite)] <- NA
  TSPdried[sapply(TSPdried, is.infinite)] <- NA
  TSPVS[sapply(TSPVS, is.infinite)] <- NA
  ##replace NA values with small number
  X.trainOB[is.na(X.trainOB)] = 0.001
  X.testOB[is.na(X.testOB)] = 0.001
  TSPdried[is.na(TSPdried)] = 0.001
  TSPVS[is.na(TSPVS)] = 0.001
  #build model
  OBclassifierRF = randomForest(x = X.trainOB,
                                y = Y.trainOB,
                                ntree = 500)
  # store at OOB error
  errorOB <- mean(OBclassifierRF$err.rate[,1])
  OOBerrorOB <- rbind(OOBerrorOB, errorOB)
  #bootstrap matrix
  OBexpectedbs <- Y.trainOB
  OBpredictedbs <-  OBclassifierRF[["predicted"]]
  bspred.matrixOB <- confusionMatrix(data=OBpredictedbs, reference=OBexpectedbs, positive = "1") #make confusion matrix
  bspred.outcomeOB <- as.matrix(bspred.matrixOB, what = "classes") #print the validation results
  bspred.rowOB <- t(bspred.outcomeOB) ##make results into a tables
  #store bootstrap predictions and results
  BSpredictionstorageOB <- rbind(BSpredictionstorageOB, OBpredictedbs)
  BSresultsOB <- rbind(BSresultsOB, bspred.rowOB)
  #store selected variables
  varimportanceOB <- as.data.frame(t(importance(OBclassifierRF)))
  OBvarimportance <- rbind(OBvarimportance, varimportanceOB)
  # Predicting the Test set results
  y_predOB <- predict(OBclassifierRF, newdata = X.testOB)
  expectedtestOB <- Y.testOB
  predictedtestOB <- y_predOB
  testpred.matrixOB <- confusionMatrix(data=predictedtestOB, reference=expectedtestOB, positive = "1") #make confusion matrix
  testpred.outcomeOB <- as.matrix(testpred.matrixOB, what = "classes") #print the validation results
  testpred.rowOB <- t(testpred.outcomeOB) ##make results into a tables
  #store test set results and validation scores
  testpredictionstorageOB <- rbind(testpredictionstorageOB, predictedtestOB)
  testresultsOB <- rbind(testresultsOB, testpred.rowOB)
  #Predict consumer prodcut set
  dried_pred = predict(OBclassifierRF, newdata = TSPdried)
  expecteddried <- driedclass
  predicteddried <- dried_pred
  driedpred.matrixOB <- confusionMatrix(data=predicteddried, reference=expecteddried, positive = "1") #make confusion matrix
  driedpred.outcomeOB <- as.matrix(driedpred.matrixOB, what = "classes") #print the validation results
  driedpred.rowOB <- t(driedpred.outcomeOB) ##make results into a tables
  #store consumer prodcut validation scores and predicitons
  driedpredictionstorageOB <- rbind(driedpredictionstorageOB, predicteddried)
  resultsdriedOB <- rbind(resultsdriedOB, driedpred.rowOB)
  #Predict VS set
  VS_pred = predict(OBclassifierRF, newdata = TSPVS)
  expectedVS <- VSclass
  predictedVS <- VS_pred
  VSpred.matrixOB <- confusionMatrix(data=predictedVS, reference=expectedVS) #make confusion matrix
  VSpred.outcomeOB <- as.matrix(VSpred.matrixOB, what = "classes") #print the validation results
  VSpred.rowOB <- t(VSpred.outcomeOB) ##make results into a tables
  #store validation set predictions and validation scores
  VSpredictionstorageOB <- rbind(VSpredictionstorageOB, predictedVS)
  resultsVSOB <- rbind(resultsVSOB, VSpred.rowOB)
}

#set up is OG is not OG
Xog <- rbind(tOG, tNOTog) ## all data 
Yog <- c('OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'nOG', 'nOG', 'nOG', 
         'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG',
         'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
Yog <- as.factor(Yog)
Yog <- ifelse(Yob == "OG", 1, 2)
##replace NA values with small number
Xog[is.na(Xog)] = 0.001
##replace 0 with small value
Xog <- replace(Xog, Xog<0,0.001)
#Make empty data frame to store OOB error
OOBerrorOG <- data.frame(matrix(ncol = 1, nrow = 0))
#make empty data frame to store validation results from each run (test set)
testresultsOG <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(testresultsOG) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                             'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store bootstrap predictions
BSpredictionstorageOG <- data.frame(matrix(ncol = 12, nrow = 0))
colnames(BSpredictionstorageOG) <- c('OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
#make empty data frame to store bootstrap validation scores
BSresultsOG <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(BSresultsOG) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty data frame to store the training set for each run 
training.runsOG <- data.frame(matrix(ncol = 12, nrow = 0))
colnames(training.runsOG) <- c(paste0(1:12))
#make empty data frame to store the test predictions for each run 
testpredictionstorageOG <- data.frame(matrix(ncol = 19, nrow = 0))
colnames(testpredictionstorageOG)  <- c('OG', 'OG', 'OG','OG', 'nOG', 'nOG', 'nOG', 'nOG', 
                                        'nOG', 'nOG','nOG','nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
#make empty data frame to store the consumer product predictions for each run 
driedpredictionstorageOG <- data.frame(matrix(ncol = 17, nrow = 0))
colnames(driedpredictionstorageOG)  <- c('nOG', 'nOG', 'OG', 'nOG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'OG', 'nOG','nOT', 'OG')
resultsdriedOG <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsdriedOG) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                              'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
#make empty matrix to store selected variables
OGvarimportance <- as.data.frame(matrix(ncol = 1123, nrow = 0))
colnames(OGvarimportance) <- c(colnames(Xog))
#load consumer product data
dried <-  read.csv("Consumer_data.csv", row.names = 1, header = TRUE)
tdried <- t(dried)
#transform data
TSdried <- sqrt(tdried)
##replace NA values with small number
TSdried[is.na(TSdried)] = 0.0001
##replace 0 with small value
TSdried <- as.data.frame(replace(TSdried, TSdried<0,0.0001))
driedclass <- c('nOG', 'nOG', 'OG', 'nOG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'OG', 'nOG','nOT', 'OG')
driedclass <- as.factor(ifelse(driedclass == "OG", 1, 2))
##set up validation set 
VS <-  read.csv("VS_data.csv", row.names = 1, header = TRUE)
tVS <- t(VS)
#transform data
tsVS <- sqrt(tVS)
##replace NA values with small number
tsVS[is.na(tsVS)] = 0.0001
##replace 0 with small value
TSVS <- replace(tsVS, tsVS<0,0.0001)
y.VS <- c('OG', 'nOG', 'nOG', 'OG', 'nOG','nOG','nOG', 'OG', 'nOG')
VSclass <- y.VS
VSclass <- as.factor(ifelse(VSclass == "OG",1,2))
#make empty data frame to store VS the predictions for each run 
VSpredictionstorageOG <- data.frame(matrix(ncol = 9, nrow = 0))
colnames(VSpredictionstorageOG)  <- c('OG', 'nOG', 'nOG', 'OG', 'nOG', 'nOG','nOG','OG', 'nOB')
#make empty data frame to store VS val scores
resultsVSOG <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(resultsVSOG) <- c('Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Precision', 
                           'Recall', 'F1', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy')
for(i in 1:100) {
  set.seed(NULL)
  data_set_size_OG <- floor(nrow(tOG)*0.80)
  index_OG <- sample(1:nrow(tOG), size = data_set_size_OG)
  data_set_size_notOG <- floor(nrow(tOG)*0.80)
  index_notOG <- sample(1:nrow(tNOTog), size = data_set_size_notOG)
  train_OG <- tOG[index_OG,]
  test_OG <- tOG[-index_OG,]
  train_nOG <- tNOTog[index_OG,]
  test_nOG <- tNOTog[-index_OG,]
  X.trainOG <- as.data.frame(rbind(train_OG, train_nOG))
  X.testOG <- as.data.frame(rbind(test_OG, test_nOG))
  Y.trainOG <- c('OG', 'OG', 'OG', 'OG', 'OG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG')
  Y.testOG <- c('OG', 'OG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG', 'nOG','nOG','nOG', 'nOG', 'nOG', 'nOG')
  training.transposeOG <- t(X.trainOG)
  training.row.namesOG <- rownames(X.trainOG)
  training.runsOG <- rbind(training.runsOG, training.row.namesOG)
  Y.trainOG <-  ifelse(Y.trainOG == "OG", 1, 2)
  Y.trainOG <- as.factor(Y.trainOG)
  Y.testOG <-  ifelse(Y.testOG == "OG", 1, 2)
  Y.testOG <- as.factor(Y.testOG)
  ##autoscale data for training set
  trainmeansOG <- as.data.frame(colMeans(X.trainOG)) #calculate means
  trainSDOG <- as.data.frame(apply(X.trainOG,2, sd)) #calculate SDs
  train_meansubOG <- X.trainOG[] - trainmeansOG[col(X.trainOG[])] #subtract means from each cell
  train_divdSDOG <- train_meansubOG[] / trainSDOG[col(train_meansubOG[])] #divide each cell by the SD
  X.trainOG <- train_divdSDOG
  #autoscale test data using training data means and SDs
  test_meansubOG <-  X.testOG[] - trainmeansOG[col(X.testOG[])]
  test_divdSDOG <- test_meansubOG[] / trainSDOG[col(test_meansubOG[])]
  X.testOG <- test_divdSDOG
  #autoscale consumer product data using training data means and SDs
  dried_meansubOG <-  TSdried[] - trainmeansOG[col(TSdried[])]
  dried_divdSDOG <- dried_meansubOG[]/ trainSDOG[col(dried_meansubOG[])]
  TSPdried <- dried_divdSDOG
  #autoscale VS data using training data means and SDs
  VS_meansubOG <-  TSVS[] - trainmeansOG[col(TSVS[])]
  VS_divdSDOG <- VS_meansubOG[]/ trainSDOG[col(VS_meansubOG[])]
  TSPVS <- VS_divdSDOG
  #remove infinate values
  X.trainOG[sapply(X.trainOG, is.infinite)] <- NA
  X.testOG[sapply(X.testOG, is.infinite)] <- NA
  TSPdried[sapply(TSPdried, is.infinite)] <- NA
  TSPVS[sapply(TSPVS, is.infinite)] <- NA
  ##replace NA values with small number
  X.trainOG[is.na(X.trainOG)] = 0.001
  X.testOG[is.na(X.testOG)] = 0.001
  TSPdried[is.na(TSPdried)] = 0.001
  TSPVS[is.na(TSPVS)] = 0.001
  #build model
  OGclassifierRF = randomForest(x = X.trainOG,
                                y = Y.trainOG,
                                ntree = 500)
  # store at OOB error
  errorOG <- mean(OGclassifierRF$err.rate[,1])
  OOBerrorOG <- rbind(OOBerrorOG, errorOG)
  #bootstrap matrix
  OGexpectedbs <- Y.trainOG
  OGpredictedbs <-  OGclassifierRF[["predicted"]]
  bspred.matrixOG <- confusionMatrix(data=OGpredictedbs, reference=OGexpectedbs, positive = "1") #make confusion matrix
  bspred.outcomeOG <- as.matrix(bspred.matrixOG, what = "classes") #print the validation results
  bspred.rowOG <- t(bspred.outcomeOG) ##make results into a tables
  #store bootstrap predictions and validation scores
  BSpredictionstorageOG <- rbind(BSpredictionstorageOG, OGpredictedbs)
  BSresultsOG <- rbind(BSresultsOG, bspred.rowOG)
  #store selected variables
  varimportanceOG <- as.data.frame(t(importance(OGclassifierRF)))
  OGvarimportance <- rbind(OGvarimportance, varimportanceOG)
  # Predicting the Test set results
  y_predOG <- predict(OGclassifierRF, newdata = X.testOG)
  expectedtestOG <- Y.testOG
  predictedtestOG <- y_predOG
  testpred.matrixOG <- confusionMatrix(data=predictedtestOG, reference=expectedtestOG, positive = "1") #make confusion matrix
  testpred.outcomeOG <- as.matrix(testpred.matrixOG, what = "classes") #print the validation results
  testpred.rowOG <- t(testpred.outcomeOG) ##make results into a tables
  #store test set predictions and validation scores
  testpredictionstorageOG <- rbind(testpredictionstorageOG, predictedtestOG)
  testresultsOG <- rbind(testresultsOG, testpred.rowOG)
  #Predict consumer product set
  dried_pred = predict(OGclassifierRF, newdata = TSPdried)
  expecteddried <- driedclass
  predicteddried <- dried_pred
  driedpred.matrixOG <- confusionMatrix(data=predicteddried, reference=expecteddried, positive = "1") #make confusion matrix
  driedpred.outcomeOG <- as.matrix(driedpred.matrixOG, what = "classes") #print the validation results
  driedpred.rowOG <- t(driedpred.outcomeOG) ##make results into a tables
  #store consumer product validation scores and predictions
  driedpredictionstorageOG <- rbind(driedpredictionstorageOG, predicteddried)
  resultsdriedOG <- rbind(resultsdriedOG, driedpred.rowOG)
  #Predict VS set
  VS_pred = predict(OGclassifierRF, newdata = TSPVS)
  expectedVS <- VSclass
  predictedVS <- VS_pred
  VSpred.matrixOG <- confusionMatrix(data=predictedVS, reference=expectedVS) #make confusion matrix
  VSpred.outcomeOG <- as.matrix(VSpred.matrixOG, what = "classes") #print the validation results
  VSpred.rowOG <- t(VSpred.outcomeOG) ##make results into a tables
  #store validation set predictions and valdiation scores
  VSpredictionstorageOG <- rbind(VSpredictionstorageOG, predictedVS)
  resultsVSOG <- rbind(resultsVSOG, VSpred.rowOG)
}