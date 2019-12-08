## https://github.com/bh1995/lab2block1.git ##

setwd("C:/Users/Bjorn/Documents/LIU/machine_learning/labs")
library(readxl)
library(tree)
library(e1071)
library(SDMTools)
library(ggrepel)
library(ggplot2)
library(boot)
library(fastICA)


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
# According to the ROC plot the Naive Bayes performs better due to its larger AUC, 
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

## Assignment 3. Uncertainty estimation ##

## 1
state_data = read.csv2("State.csv")
state_data = state_data[order(state_data$MET),]

ggplot(data = as.data.frame(state_data), aes(y = state_data[,1], x = state_data[,3]))+
  xlab("MET") + ylab("EX")+
  geom_point(color = "blue") 
# The data points looks spead out over everywhere and no apparent distribution looks to fit the data.

## 2
set.seed(12345)
tree_model = tree(EX ~ MET, data = state_data, control = tree.control(nobs = nrow(state_data), minsize = 8))
best_fit_tree1 = cv.tree(tree_model)
prune_best_fit_tree1 = prune.tree(tree_model, best = 3)
summary(prune_best_fit_tree1)

plot(prune_best_fit_tree1)
text(prune_best_fit_tree1, pretty=1, cex = 0.8, xpd = TRUE)

tree_fitted_results = predict(prune_best_fit_tree1, newdata = state_data)

ggplot(data = as.data.frame(state_data), 
       aes(y = state_data[,1], x = state_data[,3])) +
  xlab("MET") + 
  ylab("EX") +
  geom_point(col = "red") +
  geom_point(x = state_data$MET, y = tree_fitted_results, col = "blue")

hist(residuals(prune_best_fit_tree1),
     main = "Residual Histogram",
     xlab = "Residuals")
# The residual histogram is skewed to the left and there seems to be some high variance between all the residuals. 
# The tree model fits the data poorly and there are many outliers outside the three terminal nodes.

## 3
tree_fun = function(data, ind){
  set.seed(12345)
  sample = state_data[ind,]
  tree_model = tree(EX ~ MET, data = sample, control = tree.control(nobs = nrow(sample), minsize = 8)) 
  
  pruned_tree = prune.tree(tree_model, best = 3) 
  
  fitted_results = predict(pruned_tree, newdata = state_data)
  return(fitted_results)
}

res = boot(state_data, tree_fun, R=1000)

conf = envelope(res, level=0.95) 

ggplot(data = as.data.frame(state_data), 
       aes(y = state_data[,1], x = state_data[,3])) +
  xlab("MET") + 
  ylab("EX") +
  geom_point(col = "red") +
  geom_line(aes(x = state_data$MET, y = tree_fitted_results), col = "blue") +
  geom_line(aes(x = state_data$MET, y = conf$point[1,]), col = "orange") +
  geom_line(aes(x = state_data$MET, y = conf$point[2,]), col = "orange")
# As is seen in the plot above, the confidence interval is not smooth because there is large variance between points. 
# The histogram earlier showed us that the residuals are large and varied. With the confidence interval being so large
# we can conclude that the results are not extremly accurate and that the model does not fit the data very well. 

## 4

mle = prune_best_fit_tree1

rng = function(data, mle){ 
  data1 = data.frame(EX = data$EX, MET = data$MET) 
  n = length(data1$EX)
  pred = predict(mle, newdata = state_data)
  residual = data1$EX - pred
  data1$EX = rnorm(n, pred, sd(residual))
  return(data1)
}

f1 = function(data){
  res = tree(EX ~ MET, data = data, control = tree.control(nobs=nrow(state_data),minsize = 8))
  opt_res = prune.tree(res, best = 3)
  return(predict(opt_res, newdata = data))
}

f2 = function(data){
  res = tree(EX ~ MET, data = data, control = tree.control(nobs=nrow(state_data),minsize = 8))
  opt_res = prune.tree(res, best = 3)
  n = length(state_data$EX)
  opt_pred = predict(opt_res, newdata = state_data)
  pred = rnorm(n,opt_pred, sd(residuals(mle)))
  return(pred)
}
set.seed(12345)
par_boot_conf = boot(state_data, statistic = f1, R = 1000, mle = mle, ran.gen = rng, sim = "parametric") 
conf_interval = envelope(par_boot_conf, level=0.95)  

set.seed(12345)
par_boot_pred = boot(state_data, statistic = f2, R = 1000, mle = mle, ran.gen = rng, sim = "parametric") 
pred_interval = envelope(par_boot_pred, level = 0.95)  


ggplot(data = as.data.frame(state_data), 
       aes(y = state_data[,1], x = state_data[,3])) +
  xlab("MET") + 
  ylab("EX") +
  geom_point(col = "red") +
  geom_line(aes(x = state_data$MET, y = tree_fitted_results), col = "blue") +
  geom_line(aes(x = state_data$MET, y = conf_interval$point[1,]), col = "orange") +
  geom_line(aes(x = state_data$MET, y = conf_interval$point[2,]), col = "orange") +
  geom_line(aes(x = state_data$MET, y = pred_interval$point[1,]), col = "black") +
  geom_line(aes(x = state_data$MET, y = pred_interval$point[2,]), col = "black")

# The 95% confidence interval fits the points better now with only two points falling outside the interval. 
# Parametric bootstrap makes the model a bit better. 

## 5
# There is high variance between the residuals when the histogram above is considered. Because of this the parametric
# bootstrap seems to be more appropriate as it gives less variance in the confidence interval. 

## Assignment 4. Principal components ##

## 1
data = read.csv2("NIRspectra.csv", header = TRUE)

data$Viscosity = c()
prc = prcomp(data) 
summary(prc)
lambda = prc$sdev^2

var = sprintf("%2.3f", lambda/sum(lambda)*100)

screeplot(prc, main = "Principal Components")

ggplot() +
  geom_point(aes(prc$x[,1], prc$x[,2])) +
  xlab("x1") + ylab("x2")

# As can be seen from the summary and plot above, more than 99.9% of the variance is captured by the first two 
# components. There are some outliers. 

## 2
plot(prc$rotation[,1], 
     main="PC1",
     xlab = "Features",
     ylab = "Scores")
plot(prc$rotation[,2], 
     main="PC2",
     xlab = "Features",
     ylab = "Scores")
# Plot shows PC2 is explained by a few features.

## 3
data_mat = as.matrix(data)
set.seed(12345)
ICA = fastICA(data_mat, n.comp = 2, fun = "logcosh", alpha = 1, row.norm = FALSE, maxit = 200, tol = 0.0001, verbose = TRUE) 

posterior = ICA$K %*% ICA$W

plot(posterior[,1], 
     main="PC1",
     xlab = "Features", 
     ylab = "Scores")
plot(posterior[,2], 
     main="PC2",
     xlab = "Features",
     ylab = "Scores")

ggplot() +
  geom_point(aes(ICA$S[,1],ICA$S[,2])) +
  labs(x = "W1", y = "W2")

# The previos trace plots are similar, the main differance is that in the second plots, PC2 is 
# described by many more features than the first plot.
# W is the un-mixing matrix that maximizes the non-gaussianity of the components so we can extract 
# the independent components.


