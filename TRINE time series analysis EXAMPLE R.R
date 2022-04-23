library(forecast)
library(ggplot2)
library(prophet)

#Data exploration & preparation 

data = read.csv("~/Daily_Demand_Forecasting_Orders.csv")
data = na.omit(data)
h2 <- 6L
train <- head(data, round(length(data) - h2))
test <- tail(data, h2)
ts.plot(train$y)

#Assumption checking
#feature selection
#descriptive statistics

acf(train)
pacf(train)

#model fitting 
fit <- auto.arima(train$y, trace=TRUE)

#test model on unseen data, (Forecast)
plot(forecast(fit,h=6))

#model evaluation
fc = as.data.frame(forecast(fit,h=6))
fc1=as.ts(fc$`Point Forecast`)
res =as.data.frame(test$y - fc1)
table = cbind(fc$`Point Forecast`,test$y,res,fc$`Lo 95`,fc$`Hi 95`)
table 
#mape
accuracy(fc1, test$y)

#times series decomposistion 
m <- prophet(data,weekly.seasonality = FALSE)
future <- make_future_dataframe(m, periods = 6)
forecast <- predict(m, future)
prophet_plot_components(m, forecast)