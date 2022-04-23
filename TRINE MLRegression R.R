library(car)
library(mltools)
library(data.table)
library(pscl)
library(caret)
library(MASS)
library(forecast)
library(summarytools)
library(lmtest)


#descriptive statistics Data exploration & preparation 
data = read.csv("~/50_Startups.csv")
dfSummary(data)

data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)],as.factor)
dfSummary(data)

#model fitting 
model <- lm(Profit ~., data = data)
dfSummary(data)

#Assumption checking
par(mfrow = c(2, 2))
plot(model)

# distribution of studentized residuals
sresid <- studres(model) 
shapiro.test(sresid)

ncvTest(model)
#Null Hypothesis (H0): Homoscedasticity is not present
#p < .05 reject,.

bptest(model)
#Null Hypothesis (H0): Homoscedasticity is present
#If the p-value <.05 reject

durbinWatsonTest(model)
#Null Hypothesis (H0): autocorrelation is not present
#If the p-value <.05 reject

#test for multicolinearity 
data2 = subset(data, select = -c(State, Profit))
corMatrix <- round(cor(data2), 2)
corMatrix
findCorrelation(corMatrix, cutoff = 0.7, names = TRUE)

#feature selection
caret::varImp(model)
#Refined model fit
model <- lm(Profit ~R.D.Spend , data = data)
#model evaluation
summary(model)

#Assumption checking
par(mfrow = c(2, 2))
plot(model)
plot(model, 4)

#Adjust for over fitting
#Use 70% of dataset as training set and remaining 30% as testing set
set.seed(1)
sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.7,0.3))
train <- data[sample, ]
test <- data[!sample, ]  

#Refined model fit
model <- lm(Profit ~R.D.Spend , data = train)
#model evaluation
summary(model)

#test model on unseen data
pdata <- predict(model, newdata = test,)

#model evaluation
#mape <10 great ... 10-20 good ... 20-50 ok ...<50 bad
accuracy(pdata, test$Profit) 
res =as.data.frame(test$Profit - pdata)
table = cbind(test$Profit,pdata,res)
table

