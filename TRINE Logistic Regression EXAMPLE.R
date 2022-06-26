install.packages("car")
install.packages("mltools")
install.packages("data.table")
install.packages("pscl",
install.packages("caret")
install.packages("summarytools")

library(car)
library(mltools)
library(data.table)
library(pscl)
library(caret)
library(summarytools)

#descriptive statistics Data exploration & preparation 
data = read.csv("~/bank.csv")
data=na.omit(data)
dfSummary(data)

data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)],as.factor)
data = one_hot(as.data.table(data))
data=na.omit(data)
dfSummary(data)

#Adjust for overfitting
#Use 70% of dataset as training set and remaining 30% as testing set
set.seed(1)
sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.7,0.3))
train <- data[sample, ]
test <- data[!sample, ]  

train = subset(train, select = -c(y_no) )
test = subset(test, select = -c(y_no))
train=na.omit(train)


#model fitting 
glm.fit <- glm(y_yes~., family = binomial, data = train)
summary(glm.fit)

#Assumption checking
corMatrix <- round(cor(data), 2)
findCorrelation(corMatrix, cutoff = 0.7, names = TRUE)
plot(glm.fit,4)
par(mfrow = c(2, 2))
plot(glm.fit)

#Adjust for overfitting
#Use 70% of dataset as training set and remaining 30% as testing set
set.seed(1)
sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.7,0.3))
train <- data[sample, ]
test <- data[!sample, ]  

train = subset(train, select = -c(y_no) )
test = subset(test, select = -c(y_no))
train=na.omit(train)


#model evaluation
pscl::pR2(glm.fit)["McFadden"]


#test model on unseen data
pdata <- predict(glm.fit, newdata = test, type = "response")
x= as.factor(test$y_yes)
y=as.factor(as.numeric(pdata>0.5))
confusionMatrix(data =x , reference = y)

#feature selection
caret::varImp(glm.fit)
