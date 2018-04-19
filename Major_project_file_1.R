rm(list = ls())
setwd("D:/R_DataSet")
getwd()

#_________  LOAD DATA FROM HARD-DISK  _______________#

marketing_train = read.csv("marketing_tr.csv",na.strings = "")
colnames(marketing_train)

#_________  DATA LOADED SUCESSFULLY  _______________#

#_________  NOW EXPLORE THE DATA     _______________#

marketing_train$schooling[marketing_train$schooling %in% "illiterate"] = "unknown"
marketing_train$schooling[marketing_train$schooling %in% c("basic.4y","basic.6y","basic.9y","high.school","professional.course")] = "high.school"

marketing_train$default[marketing_train$default %in% "yes"] = "unknown"
marketing_train$default = as.factor(as.character(marketing_train$default))

marketing_train$marital[marketing_train$marital %in% "unknown"] = "married"
marketing_train$marital = as.factor(as.character(marketing_train$marital))

marketing_train$month[marketing_train$month %in% c("sep","oct","mar","dec")] = "dec"
marketing_train$month[marketing_train$month %in% c("aug","jul","jun","may","nov")] = "jun"
marketing_train$month = as.factor(as.character(marketing_train$month))

marketing_train$loan[marketing_train$loan %in% "unknown"] = "no"
marketing_train$loan = as.factor(as.character(marketing_train$loan))

marketing_train$schooling = as.factor(as.character(marketing_train$schooling))

marketing_train$profession[marketing_train$profession %in% c("management","unknown","unemployed","admin.")] = "admin."
marketing_train$profession[marketing_train$profession %in% c("blue-collar","housemaid","services","self-employed","entrepreneur","technician")] = "blue-collar"
marketing_train$profession = as.factor(as.character(marketing_train$profession))


#_________  MISSING VALUE ANALYSIS  ________________#

missing_val = data.frame(apply(marketing_train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(marketing_train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
sum(is.na(marketing_train))
write.csv(missing_val, "Major_project_missing_report.csv", row.names = F)

#__________     IMPUTE THE MISSING VALUES USING KNN METHOD      ________________#

library("DMwR")
marketing_train = knnImputation(marketing_train, k = 3)


#__________     CONVERT CATEGORICAL VALUES IN UNIQUE LEVELS     ________________#

for(i in 1:ncol(marketing_train)){
  
  if(class(marketing_train[,i]) == 'factor'){
    
    marketing_train[,i] = factor(marketing_train[,i], labels=(1:length(levels(factor(marketing_train[,i])))))
    
  }
}

#__________  OUTLIER ANALYSIS  ______________#

numeric_index = sapply(marketing_train,is.numeric) #selecting only numeric

numeric_data = marketing_train[,numeric_index]

cnames = colnames(numeric_data)
cnames

#__________ CORRELATION PLOT FOR FEATURES   ____________#

library("corrgram")
corrgram(marketing_train[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


# Chi-squared Test of Independence

factor_index = sapply(marketing_train,is.factor)
factor_data = marketing_train[,factor_index]

for (i in 1:10)
{
  print(names(factor_data)[i])
  print=(chisq.test(table(factor_data$responded,factor_data[,i])))
}

write.csv(marketing_train,"marketing_train_clean.csv",row.names = T)

# Dimension Reduction
# _____   CHI SQUARE TEST CONCLUDES WHICH ARE THE VARIABLES IMPORTANT TO DETERMINE TARGET VARIABLE  ___________#

colnames(marketing_train)
marketing_train = subset(marketing_train,select=-c(pdays,emp.var.rate,day_of_week,loan,housing))


#________     SELECT NUMERIC VARIABLES AND PERFORM NORMALISATION OPERATION OVER THEM   _____________#

cnames = c("custAge","campaign","previous","cons.price.idx","cons.conf.idx","euribor3m","nr.employed",
           "pmonths","pastEmail")

for(i in cnames){
  print(i)
  marketing_train[,i] = (marketing_train[,i] - min(marketing_train[,i]))/
    (max(marketing_train[,i] - min(marketing_train[,i])))
}
write.csv(marketing_train,"Cleaned_project_data.csv", row.names = T)

#____    MODEL DEVELOPMENT PHASE   _____________#

library("DataCombine")
rmExcept("marketing_train")

#_____    Divide data into train and test using stratified sampling method  ______#

marketing_train = read.csv("Cleaned_project_data.csv")[-1]
set.seed(1234)
library("caret")
train.index = createDataPartition(marketing_train$responded, p = .80, list = FALSE)
train = marketing_train[ train.index,]
test  = marketing_train[-train.index,]

##Decision tree for classification
#Develop Model on training data
#C50_model = C5.0(responded ~., train, trials = 100, rules = TRUE)

#__________       Summary of DT model
#summary(C50_model)

#__________      write rules into disk
#write(capture.output(summary(C50_model)), "c50Rules.txt")

#__________      Lets predict for test cases
#C50_Predictions = predict(C50_model, test[,-17], type = "class")

#__________      Evaluate the performance of classification model

#ConfMatrix_C50 = table(test$responded, C50_Predictions)
#confusionMatrix(ConfMatrix_C50)
#write(capture.output(confusionMatrix(ConfMatrix_C50)),"Predicted_test_confusion_matrix.txt")

#__________ APPLY RANDOM FOREST MODEL ___________________________

library("randomForest")
library("inTrees")
#RF_model = randomForest(responded ~ ., train, importance = TRUE, ntree = 500)
#treeList = RF2List(RF_model)

#__________ EXTRACT RULES __________

#exec = extractRules(treeList,train[,-17])

#exec[1:2,]
#
#_____ Evaluate some rules  ______________#

#ruleMetric[1:2,]

#Predict test data using random forest model 

#RF_predictions = predict(RF_model,test[,-17])

#sum(RF_predictions %in% 2)

#store_data_index = RF_predictions %in% 2  

#store_data = marketing_train[store_data_index,]
#dim(store_data)

#Evaluate the performance of classification model  #Accuracy = 90.55%
#ConfMatrix_RF = table(test$responded,RF_predictions)
#confusionMatrix(ConfMatrix_RF)
#ConfMatrix_RF

#False negative rate # 64%
#FNR = FN/FN+TP


#Logistic Regression Model

#library("dplyr")

#marketing_train$responded = marketing_train$responded -1

#logit_model = glm(responded ~ ., data = train, family = "binomial")

#summary(logit_model)


#___ Predict using logistic regression trained model _____#

#logit_predictions = predict(logit_model, newdata = test,type = "response")

#Convert into probabilities

#logit_predictions = ifelse(logit_predictions > 0.5,1,0)


#______  Evaluation of logistic regression model __#

#ConfMatrix_RF =  table(test$responded, logit_predictions)

#FALSE NEGATIVE RATE 
#Accuracy 90.6%
#FNR = FN/FN+TP #76  #71

#__________KNN IMPLEMENTATION MODEL

library("class")

#_______   PREDICT TEST DATA  _________________

KNN_prediction = knn(train[,1:16], test[,1:16] , train$responded, k=5)

KNN_prediction
KNN_prediction = as.data.frame(KNN_prediction)

market = as.data.frame(test)

my_data_frame = cbind(market,KNN_prediction)

data_index = which(KNN_prediction %in% 2)

data = subset(my_data_frame , KNN_prediction %in% 2)

#THIS IS THE FINAL DATA SET OBTAINED
#PER 1482 RECORDS THERE IS A CLASSIFICATION OF 75 CLIENTS WHO ARE MOST LIKELY TO ACCEPT THE POLICY


#Confusion Matrix

Conf_matrix = table(KNN_prediction,test$responded)

#Accuracy
sum(diag(Conf_matrix))/nrow(test)  # 89.67
Conf_matrix

#FNR FN/FN+TP #48%
36/(36+39)

#__________________   NAIVE BASE MODEL     ___________________

#library(e1071)
#NB_model = naiveBayes(responded ~. , data =train)

#_______________ PREDICT ON THE TEST CASE  ___________________

#NB_predictions = predict(NB_model, test[,1:16], type = 'class')

# CHECK OUT THE CONFUSION MATRIX ___________________

#conf_matrix = table(observed = test[,17] , predicted =  NB_predictions)
#confusionMatrix(conf_matrix)
#ACCURACY 86.1
#FNR 54.1 



