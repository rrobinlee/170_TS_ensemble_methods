# Remove key later
library(Quandl)
library(tidyverse)
library(dygraphs)
library(tsfeatures)
Quandl.api_key("")
library(TSstudio)

## As a package, TSstudio has its own functions. 

unemployment_rate = Quandl(code="FRED/UNRATENSA",
                           type="ts",  
                           collapse="monthly", 
                           order="asc", 
                           start_date="1980-01-01",
                           end_date="2019-01-01",
                           meta=TRUE)
class(unemployment_rate)

ts_info(unemployment_rate) # rapper for frequency(), start() and  end() combined
ts_plot(unemployment_rate, 
        title="US Monthly Unemployment Rate", 
        Ytitle="Time",
        Xtitle="Rate") 
# The series components 
ts_decompose(unemployment_rate) #wrapper for decompose 
tsfeatures::tsfeatures(unemployment_rate)
#Seasonal analysis

unemployment_rate_detrend=unemployment_rate-decompose(unemployment_rate)$trend

# the following is an enhanced wrapper for seasonal boxplot.
ts_seasonal(unemployment_rate_detrend,type="box") 

ts_decompose(AirPassengers, type = "multiplicative")

#The following is a wrapper for correlation analysis with acf and ccf 
# notice that the wrapper omits the lag 0 in the acf
# which the acf of a ts() object does not do. 
ts_cor(unemployment_rate)
## the following is a wrapper for lag plots 
ts_lags(unemployment_rate, lags=c(12,24,36))

##Select training data 
# See below a wrapper to create a training window of the data 
# and at the same time rename the variables to be ready for prophet. 
df=ts_to_prophet(window(unemployment_rate,start=c(1980,1)))
names(df)=c("date","y")
head(df)
ts_plot(df, 
        title="US Monthly Unemployment Rate", 
        Ytitle="Time",
        Xtitle="Rate")

## Feature engineering 
# create new features that will be used as inputs in the model 

#  install.packages("dplyr")
library(lubridate)

df <- df %>% mutate(month = factor(lubridate::month(date, label = TRUE), ordered = FALSE),
                    lag12 = lag(y, n = 12)) %>%
  filter(!is.na(lag12))
df$trend <- 1:nrow(df)
df$trend_sqr <- df$trend ^ 2

str(df)

## training, testing and model evaluation

h <- 12
train_df <- df[1:(nrow(df) - h), ]
test_df <- df[(nrow(df) - h + 1):nrow(df), ]

## inputs for forecast 

forecast_df <- data.frame(date = seq.Date(from = max(df$date) + lubridate::month(1),
                                          length.out = h, by = "month"),
                          trend = seq(from = max(df$trend) + 1, length.out = h, by = 1))
forecast_df$trend_sqr <- forecast_df$trend ^ 2

# to avoid conflict with the h2o `month` function use the "lubridate::month" 
# to explicly call the month from the lubridate function 
forecast_df$month <- factor(lubridate::month(forecast_df$date, 
                                             label = TRUE), ordered= FALSE) 
forecast_df$lag12 <- tail(df$y, 12)



####################################
# Benchmark regression model
######################################

lr <- lm(y ~ month + lag12 + trend + trend_sqr, data = train_df)
summary(lr)
test_df$yhat <- predict(lr, newdata = test_df)
mape_lr <- mean(abs(test_df$y - test_df$yhat) / test_df$y)
mape_lr
rmse_lr <- sqrt(mean((test_df$y - test_df$yhat)^2))
rmse_lr
rmse_lr_st <- sqrt(mean((test_df$y[1:2] - test_df$yhat[1:2])^2))
rmse_lr_st





####################################
# Supervised Machine Learning 
######################################


#### Prepare for ML 
# install.packages("h2o")
library(h2o)

h2o.init(max_mem_size = "16G")
train_h <- as.h2o(train_df)
test_h <- as.h2o(test_df)
forecast_h <- as.h2o(forecast_df)

forecast_h


####################################
# Random Forest
######################################


###### Simple RF model using 500 trees and 5 folder CV 

### Training process. Will add stop criterion. Stopping metric 
## is RMSE, tolerance is 0.0001, stopping rounds set 
# to 10 (Krispin, 2019)

x=c("month", "lag12", "trend", "trend_sqr")
y="y"

rf_md = h2o.randomForest(training_frame=train_h,
                         nfolds=5,
                         x=x, 
                         y=y, 
                         ntrees=500,
                         stopping_rounds=10,
                         stopping_metric="RMSE", 
                         score_each_iteration=TRUE,
                         stopping_tolerance=0.0001,
                         seed=1234)

## see the contribution of the mode inputs

h2o.varimp_plot(rf_md)

## See model output 

rf_md@model$model_summary

rf_md

## learning process of the model as a function of the number of trees

library(plotly)
tree_score =rf_md@model$scoring_history$training_rmse

plot_ly(x=seq_along(tree_score), y=tree_score,
        type="scatter", mode="line") %>%
  layout(title="The trained Model Score History", 
         yaxis=list(title="RMSE"), 
         xaxis=list(title="Num. of Trees"))

## Model performance in test set. Forecasting performance 

test_h$pred_rf = h2o.predict(rf_md, test_h)
test_h$pred_rf  

## transfer h2o data frame to a data.frame object 

test_1= as.data.frame(test_h)

## Calculate the MAPE score of the RF model on the test partition

mape_rf = mean(abs(test_1$y -test_1$pred_rf)/test_1$y)
mape_rf  
rmse_rf <- sqrt(mean((test_1$y - test_1$pred_rf)^2))
rmse_rf
rmse_rf_st <- sqrt(mean((test_1$y[1:2] - test_1$pred_rf[1:2])^2))
rmse_rf_st

## Visualizing model performance 

plot_ly(data=test_1) %>%
  add_lines(x=~date, y=~y, name="Actual") %>%
  add_lines(x=~date, y=~yhat, name="Linear Regression", line=
              list(dash="dot")) %>%
  add_lines(x=~date, y=~pred_rf, name="RF", line=
              list(dash="dash")) %>% 
  layout(title="US Unemployment-Actual vs. Prediction (Random Forest)", 
         yaxis= list(title="Rate"), 
         xaxis=list(title="Month"))


####################################
# Gradient boosting 
######################################


### Train the GB model with the same input used in RF 

gbm_md =h2o.gbm( 
  training_frame = train_h,
  nfold=5, 
  x=x,
  y=y, 
  max_depth=20, 
  distribution="gaussian",
  ntrees=500, 
  learn_rate=0.1,
  score_each_iteration=TRUE 
)

## See model output 

gbm_md@model$model_summary

gbm_md

## How important are the model variables in the training (fitting)
h2o.varimp_plot(gbm_md)

## learning process of the model as a function of the number of trees

library(plotly)
tree_score =gbm_md@model$scoring_history$training_rmse

plot_ly(x=seq_along(tree_score), y=tree_score,
        type="scatter", mode="line") %>%
  layout(title="The trained Model Score History", 
         yaxis=list(title="RMSE"), 
         xaxis=list(title="Num. of Trees"))

# test the model's performance on the testing set 

test_h$pred_gbm = h2o.predict(gbm_md, test_h)
test_1= as.data.frame(test_h)  

## calculate mape in the test set (of the forecast)
mape_gbm = mean(abs(test_1$y -test_1$pred_gbm)/test_1$y)
mape_gbm 
rmse_gbm <- sqrt(mean((test_1$y - test_1$pred_gbm)^2))
rmse_gbm
rmse_gbm_st <- sqrt(mean((test_1$y[1:2] - test_1$pred_gbm[1:2])^2))
rmse_gbm_st

## Visualizing model performance in the test set 

plot_ly(data=test_1) %>%
  add_lines(x=~date, y=~y, name="Actual") %>%
  add_lines(x=~date, y=~yhat, name="Linear Regression", line=
              list(dash="dot")) %>%
  add_lines(x=~date, y=~pred_gbm, name="GBM", line=
              list(dash="dash")) %>% 
  add_lines(x=~date, y=~pred_rf, name="RF", line=
              list(dash="dash")) %>% 
  layout(title="US Unemployment-Actual vs. Prediction (GBM)", 
         yaxis= list(title="Rate"), 
         xaxis=list(title="Month"))




####################################
# Prophet
######################################
#install.packages("prophet")
library(prophet)
colnames(train_df)[1] <- "ds"
m <- prophet(train_df, daily.seasonality = F, weekly.seasonality = F)
future <- make_future_dataframe(m, periods = 12, freq = 'month')
tail(future,12)

forecast <- predict(m)
#tail(forecast(c('ds', 'yhat','yhat_lower','yhat_upper')))
plot(m,forecast, ylab = "Rate", xlab = "", main = "US Unemployment Forecast (Prophet)")
prophet_plot_components(m, forecast)
mape_prophet <- mean(abs(test_df$y - forecast$yhat)/test_df$y)
mape_prophet
rmse_prophet <- sqrt(mean((test_df$y - forecast$yhat)^2))
rmse_prophet
rmse_prophet_st <- sqrt(mean((test_df$y[1:2] - forecast$yhat[1:2])^2))
rmse_prophet_st

forecast <- tail(predict(m)$yhat,12)
plot_ly(data=test_1) %>%
  add_lines(x=~date, y=~y, name="Actual") %>%
  add_lines(x=~date, y=~yhat, name="Linear Regression", line=
              list(dash="dot")) %>%
  add_lines(x=~date, y=~pred_gbm, name="GBM", line=
              list(dash="dash")) %>% 
  add_lines(x=~date, y=~pred_rf, name="RF", line=
              list(dash="dash")) %>% 
  add_lines(x=~date, y=~forecast, name="Prophet", line=
              list(dash="dash")) %>% 
  layout(title="US Unemployment-Actual vs. Prediction (GBM)", 
         yaxis= list(title="Rate"), 
         xaxis=list(title="Month"))
