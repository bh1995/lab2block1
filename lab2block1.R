## https://github.com/bh1995/lab2block1.git ##

setwd("C:/Users/Bjorn/Documents/LIU/machine_learning/labs")
library(readxl)
library(tree)
library(e1071)
library(SDMTools)
library(ggrepel)


## Assignment 2. Analysis of credit scoring ##

credit_data = read_excel("creditscoring.xls")
credit_data$good_bad = as.factor(credit_data$good_bad)

## 1
# Split data into training/validation/test as 50/25/25.

n=dim(credit_data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
id2=sample((1:1000)[-id],floor(n*0.25))
train=credit_data[id,]
validation=credit_data[id2,]
test=credit_data[-c(id,id2),]

## 2
# Produce the models

deviance_model = tree(good_bad ~., data=train, split="deviance")
gini_model = tree(good_bad ~., data=train, split="gini")

# Predictions/ errors for deviance tree
# Test Error
deviance_fitted.results_test = predict(deviance_model, newdata=test, type="class")
deviance_misClasificError_test = mean(deviance_fitted.results_test != test$good_bad)
deviance_confmat_test = table("Y"=test$good_bad,"Y hat"=deviance_fitted.results_test)
deviance_confmat_test
print(paste("Deviance Tree Error Rate with Test Data:", 1-sum(diag(deviance_confmat_test)) / sum(deviance_confmat_test)))
# Train Error
deviance_fitted.results_train = predict(deviance_model, newdata=train, type="class")
deviance_misClasificError_train = mean(deviance_fitted.results_train != train$good_bad)
deviance_confmat_train = table("Y"=train$good_bad,"Y hat"=deviance_fitted.results_train)
deviance_confmat_train
print(paste("Deviance Tree Error Rate with Train Data:", 1-sum(diag(deviance_confmat_train)) / sum(deviance_confmat_train)))

# Predictions/ errors for gini tree
# Test Error
gini_fitted.results_test = predict(gini_model, newdata=test, type="class")
gini_misClasificError_test = mean(gini_fitted.results_test != test$good_bad)
gini_confmat_test = table("Y"=test$good_bad,"Y hat"=gini_fitted.results_test)
gini_confmat_test
print(paste("Gini Tree Error Rate with Test Data:", 1-sum(diag(gini_confmat_test)) / sum(gini_confmat_test)))

# Train Error
gini_fitted.results_train = predict(gini_model, newdata=train, type="class")
gini_misClasificError_train = mean(gini_fitted.results_train != train$good_bad)
gini_confmat_train = table("Y"=train$good_bad,"Y hat"=gini_fitted.results_train)
gini_confmat_train
print(paste("Gini Tree Error Rate with Train Data:", 1-sum(diag(gini_confmat_train)) / sum(gini_confmat_train)))

# After running the deviance and gini models a few times it seems that both have similar
# results, but the deviance index may be a slightly better predicter for the data.

## 3

prune_train = prune.tree(deviance_model, method = "deviance")
prune_validation = prune.tree(deviance_model, newdata = validation, method = "deviance")

prune_dataframe = data.frame(Length = prune_train$size, Train = prune_train$dev, Validation = prune_validation$dev)
colnames(prune_dataframe) = c("Nr Leaves", "Training Deviation", "Validation deviation")

plot(prune_dataframe$`Nr Leaves`, prune_dataframe$`Training Deviation`, col="brown", ylim = c(250,900))
points(prune_dataframe$`Nr Leaves`, prune_dataframe$`Validation deviation`, col="blue")
legend("topleft", c("Train data","Validation data"),pch=c(1,1),col = c("brown","blue"))
# Lowest deviance seems to be when nr leaves = 4 for the validation data and about 12 
# for the training data.

# Plot the variables used in model
best_tree = prune.tree(deviance_model, best = 4, method="deviance")
plot(best_tree)
text(best_tree, pretty = 0)
# The deviance model with four leaves is using variables: duration, history, savings. 


## 4
naivebayes_model = naiveBayes(good_bad ~., data=train)

# Predictions/ errors for Naive Bayes model
# Test Error
naivebayes_fitted.results_test = predict(naivebayes_model, newdata=test, type="class")
naivebayes_confmat_test = table("Y"=test$good_bad,"Y hat"=naivebayes_fitted.results_test)
naivebayes_confmat_test
print(paste("naivebayes Error Rate with Test Data:", 1-sum(diag(naivebayes_confmat_test)) / sum(naivebayes_confmat_test)))

# Train Error
naivebayes_fitted.results_train = predict(naivebayes_model, newdata=train, type="class")
naivebayes_confmat_train = table("Y"=train$good_bad,"Y hat"=naivebayes_fitted.results_train)
naivebayes_confmat_test
print(paste("naivebayes Error Rate with Train Data:", 1-sum(diag(naivebayes_confmat_train)) / sum(naivebayes_confmat_train)))
# It looks like the Naive Bayes model performs similarly well compared to the deviance
# tree and gini index models, but has a bit more error than both. 

## 5

# TPR and FPR for optimal tree
Pi = seq(0.05, 0.95, 0.05)

best_tree_fit = predict(best_tree, newdata = test)
tree_good = best_tree_fit[,2]
true_assign = ifelse(test$good_bad == "good", 1, 0)

tree_TPR_FPR = matrix(nrow = 2, ncol = length(Pi))
rownames(tree_TPR_FPR) = c("TPR", "FPR")



for (i in 1:length(Pi)){
  tree_assign = ifelse(tree_good > Pi[i], 1, 0)
  tree_confmat = confusion.matrix(tree_assign, true_assign)

  tpr1 = tree_confmat[2,2]/(tree_confmat[2,1] + tree_confmat[2,2])
  fpr1 = tree_confmat[1,2]/(tree_confmat[1,1] + tree_confmat[1,2])
  
  tree_TPR_FPR[,i] <- c(tpr1,fpr1)
}

# TPR and FPR for naive bayes
bayes_model = naiveBayes(good_bad ~ ., data = train)
bayes_fit = predict(bayes_model, newdata = test, type = "raw")
bayes_good = bayes_fit[,2]

bayes_TPR_FPR = matrix(nrow = 2, ncol = length(Pi))
rownames(bayes_TPR_FPR) = c("TPR", "FPR")


for (i in 1:length(Pi)) {
  bayes_assign = ifelse(bayes_good > Pi[i], 1, 0)
  bayes_confmat = confusion.matrix(bayes_assign, true_assign)

  tpr2 = bayes_confmat[2,2]/(bayes_confmat[2,1] + bayes_confmat[2,2])
  fpr2 = bayes_confmat[1,2]/(bayes_confmat[1,1] + bayes_confmat[1,2])

  bayes_TPR_FPR[,i] = c(tpr2,fpr2)
}

# ROC optimal Tree Naive Bayes plot

ggplot() + 
  geom_line(aes(x = tree_TPR_FPR[2,], y = tree_TPR_FPR[1,], col = "Optimal Tree")) + 
  geom_line(aes(x = bayes_TPR_FPR[2,], y = bayes_TPR_FPR[1,], col = "Naive Bayes")) + 
  xlab("False-Positive Rate") + 
  ylab("True-Positive Rate") +
  ggtitle("ROC")
# According to the ROC plot the Naive Bayes performs better due to it’s larger AUC, 
# this could be attributed to all thresholds being used.
# The percentage of FPR to TPR in Naive Bayes is smaller than its percentage in the 
# optimal tree.

## 6

loss_mat = matrix(c(0,10,1,0), nrow = 2)

loss_function = function(data,loss_mat){
  prob = ifelse(data$good_bad == "good",1,0)
  
  naivebayes_model = naiveBayes(good_bad ~.,data=train)
  naivebayes_fit = predict(naivebayes_model, newdata = data, type = "raw")
  
  #To penalize the FPR, the probability of the predicted as good need to be 
  #10 times the probability of the predicted as bad to be classified as good
  naivebayes_fit = ifelse(loss_mat[1,2] * naivebayes_fit[,2] > loss_mat[2,1] * naivebayes_fit[,1],1,0)
  
  conf_mat = table("Y" = prob, "Y hat" = naivebayes_fit)
  miss_rate = 1-sum(diag(conf_mat))/sum(conf_mat)
  # rownames(conf_mat) <- c("Bad", "Good")
  # colnames(conf_mat) <- c("Bad", "Good")
  
  result = list("Confusion Matrix" = conf_mat, "Error Rate" = miss_rate)
  return(result)
}
print("Training data:")
loss_function(train,loss_mat)
print("Testing data:")
loss_function(test,loss_mat)
# When the loss matrix was applied the FPR decreased, but the error rate has gotten worse.




