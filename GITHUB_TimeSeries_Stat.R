library(data.table)
library(neuralnet)
library(caret)
library(forecast)
library(dplyr)
library(ggplot2)
library(janitor)
library(padr)
library(lubridate)
library(tsfknn)
library(smooth)

#WARNING: This is row-wise time series data 
#Import file
df = read.csv("Sample_Market_SKU_data.csv")
n=ncol(df)
df[is.na(df)] <- 0
#Remove the non-numeric columns (1 & 2)
d_mat = as.matrix(df[,3:n])
#N rows in the matrix
n=nrow(d_mat)
#C columns in the matrix
c=ncol(d_mat)
#FORECAST HORIZON = h
h=18

#Create matrices to store the generated forecasts
ets = matrix(nrow=n, ncol=t)
ar = matrix(nrow=n, ncol=t)
theta = matrix(nrow=n, ncol=t)
stlf = matrix(nrow=n, ncol=t)
tbats = matrix(nrow=n, ncol=t)
knn= matrix(nrow=n, ncol=t)

################################ Test Train Split ############################
t=3 #size of test set
d_mat= d_mat[,1:(c-t)] #Note: This is a matrix not a time series object
x_test = d_mat[,((c-t+1):c)] #Note: Same as above
View(x_test)

################################ Generate Forecasts for Test set ############################

for(i in 1:n)
{
  ets[i,] = round(forecast(ets(ts(d_mat[i,1:ncol(d_mat)], frequency=12),model='ZZZ'), h=t)$mean)
  ar[i,] = round(forecast(auto.arima(ts(d_mat[i,1:ncol(d_mat)],frequency=12)), h=t)$mean)
  theta[i,] = round(thetaf(ts(d_mat[i,1:ncol(d_mat)], frequency=12),h=t)$mean)
  stlf[i,] = stlf(ts(d_mat[i,1:ncol(d_mat)], frequency=12), h=t)$mean
  tbats[i,]= forecast(tbats(ts(d_mat[i,1:ncol(d_mat)], frequency=12),use.box.cox=TRUE,
                            use.trend=TRUE,
                            use.damped.trend=TRUE,
                            biasadj=TRUE), h=t)$mean
  
  knn[i,] = round(knn_forecasting(ts(d_mat[i,1:ncol(d_mat)], frequency=12), h=t, k=6, msas = "recursive",cf="mean")$prediction)
  
  }
print(i)

#Generate ensemble forecasts
tbl.1 =  as.matrix((ets+ar)/2)
tbl.2 =  as.matrix((ar+theta)/2)
tbl.3 =  as.matrix((theta+stlf)/2)
tbl.4 =  as.matrix((stlf+ets)/2)
tbl.5 =  as.matrix((stlf+ar)/2)
tbl.6 =  as.matrix((knn+ets)/2)
tbl.7 =  as.matrix((knn+ar)/2)
tbl.8 =  as.matrix((knn+theta)/2)

################################ Calculate Accuracy of Ensemble Forecasts ############################

#Compare the Forecast for the next 3 months with the Actuals of last 3 months 
#for each ensemble model
tblList <- list(tbl.1, tbl.2, tbl.3, tbl.4,tbl.5, tbl.6,tbl.7,tbl.8)
#Create a matrix for each ensemble model's accuracy
acc=matrix(nrow=n,ncol=length(tblList))
colnames(acc) = c("ETS+AR","AR+THETA","THETA+STLF","STLF+ETS","STLF+AR","KNN+ETS","KNN+AR","KNN+THETA")

for (i in 1:length(tblList)) 
    {
  for (j in 1:n){
  tbl <- tblList[[i]]
  #Test accuracy for each row in each table with tbl.
  acc[j,i]=round(accuracy(ts(x_test[j,], frequency=1),(ts(tbl[j,], frequency=1)))[5], digits=1) #MAPE is the 5th column
  }
}
acc_df=as_data_frame(acc)
acc_df$RowMinColIndex = apply(acc_df, 1, which.min) #Find the column index of the min row value
#Reattach the non-numeric columns to Accuracy matrix
acc_full = cbind(df[,1:2],acc_df)

############################## Generate Forecasts Forecast Horizon=h ####################################

#Create matrices to store the generated forecasts
ets1 = matrix(nrow=n, ncol=h)
ar1 = matrix(nrow=n, ncol=h)
theta1 = matrix(nrow=n, ncol=h)
stlf1 = matrix(nrow=n, ncol=h)
tbats1 = matrix(nrow=n, ncol=h)
knn1= matrix(nrow=n, ncol=h)

for(i in 1:n)
{
  ets1[i,] = round(forecast(ets(ts(d_mat[i,1:ncol(d_mat)], frequency=12),model='ZZZ'), h=h)$mean)
  ar1[i,] = round(forecast(auto.arima(ts(d_mat[i,1:ncol(d_mat)],frequency=12)), h=h)$mean)
  theta1[i,] = round(thetaf(ts(d_mat[i,1:ncol(d_mat)], frequency=12),h=h)$mean)
  stlf1[i,] = stlf(ts(d_mat[i,1:ncol(d_mat)], frequency=12), h=h)$mean
  tbats1[i,]= forecast(tbats(ts(d_mat[i,1:ncol(d_mat)], frequency=12),use.box.cox=TRUE,
                            use.trend=TRUE,
                            use.damped.trend=TRUE,
                            biasadj=TRUE), h=h)$mean
  knn1[i,] = round(knn_forecasting(ts(d_mat[i,1:ncol(d_mat)], frequency=12), h=h, k=6, msas = "recursive",cf="mean")$prediction)
  
}
print(i)

#Generate final ensemble forecasts
mat.1 =  as.matrix((ets1+ar1)/2)
mat.2 =  as.matrix((ar1+theta1)/2)
mat.3 =  as.matrix((theta1+stlf1)/2)
mat.4 =  as.matrix((stlf1+ets1)/2)
mat.5 =  as.matrix((stlf1+ar1)/2)
mat.6 =  as.matrix((knn1+ets1)/2)
mat.7 =  as.matrix((knn1+ar1)/2)
mat.8 =  as.matrix((knn1+theta1)/2)


#Find the best ensemble forecast based on acc_df's last column(RowMinColIndex)
#Create a List. We will iterate through this best forecast algo list
#Not possible to create this list manually for large datasets. Need to create a loop for this
matList = list(mat.4,mat.4,mat.4,mat.4,mat.1,mat.1,mat.1,mat.5,mat.5,mat.1)

#Generate a forecast file with best forecast for each SalesProduct-Market combination
forecast = matrix(nrow=length(matList), ncol=h)
for (i in 1:length(matList))
{
  mat= matList[[i]]
  forecast[i,] = mat[i,]
}
forecast_full = cbind(df[1:2],forecast)
View(forecast_full)

write.table(forecast_full, file='R_TimeSeries_Predictions.csv', sep=',',row.names=FALSE)
