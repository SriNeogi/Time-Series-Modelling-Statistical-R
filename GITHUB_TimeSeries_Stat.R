library(data.table)
library(neuralnet)
library(caret)
library(forecast)
library(dplyr)
library(ggplot2)
library(janitor)
library(padr)
library(lubridate)


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
ets = matrix(nrow=n, ncol=h)
ar = matrix(nrow=n, ncol=h)
theta = matrix(nrow=n, ncol=h)
stlf = matrix(nrow=n, ncol=h)
tbats = matrix(nrow=n, ncol=h)

################################ Generate Forecasts ############################

for(i in 1:n)
{
  ets[i,] = round(forecast(ets(ts(d_mat[i,1:ncol(d_mat)], frequency=12),model='ZZZ'), h=h)$mean)
  ar[i,] = round(forecast(auto.arima(ts(d_mat[i,1:ncol(d_mat)],frequency=12)), h=h)$mean)
  theta[i,] = round(thetaf(ts(d_mat[i,1:ncol(d_mat)], frequency=12),h=h)$mean)
  stlf[i,] = stlf(ts(d_mat[i,1:ncol(d_mat)], frequency=12), h=h)$mean
  tbats[i,]= forecast(tbats(ts(d_mat[i,1:ncol(d_mat)], frequency=12),use.box.cox=TRUE,
                            use.trend=TRUE,
                            use.damped.trend=TRUE,
                            use.arma.errors= ,biasadj=TRUE), h=h)$mean
  }
print(i)

#Plot ETS and ARIMA for row 1, change row number for other rows
autoplot(forecast(ets(ts(d_mat[1,1:ncol(d_mat)], frequency=12),model='ZZZ')), h=h)
autoplot(forecast(auto.arima(ts(d_mat[1,1:ncol(d_mat)],frequency=12)), h=h))

#Generate ensemble forecasts
tbl.1 =  data.table((ets+ar)/2)
tbl.2 =  data.table((ar+theta)/2)
tbl.3 =  data.table((theta+stlf)/2)
tbl.4 =  data.table((stlf+ets)/2)
tbl.5 =  data.table((stlf+ar)/2)

################################ Calculate Accuracy ############################

#Compare the Forecast for the next 3 months with the Actuals of last 3 months 
#for each ensemble model
tblList <- list(tbl.1, tbl.2, tbl.3, tbl.4,tbl.5)
#Create a matrix for each ensemble model's forecast sum
f=matrix(nrow=1,ncol=5)
colnames(f) = c("ETS+AR","AR+THETA","THETA+STLF","STLF+ETS","STLF+AR")
#Create a matrix for each ensemble model's accuracy
acc=matrix(nrow=1,ncol=5)
colnames(acc) = c("ETS+AR","AR+THETA","THETA+STLF","STLF+ETS","STLF+AR")

#Calculate the sum of Actuals for last 3 months
a=sum(d_mat[,((c-2):c)])

#Accuracy=(sum(Actuals 3 months)- sum(Forecast 3 months))/sum(Actuals 3 months)*100
for (i in 1:5) {
  tbl <- tblList[[i]]
  # Do something with tbl.
  f[,i]=sum(tbl[,2:4])
  acc[,i]=((a-f[i])/a)*100
}
print(i)
#Choose the best ensemble forecast with lowest % error
View(acc)

#Reattach the non-numeric columns
combo2_full = cbind(df[,1:2],combo2)

write.table(combo2_full, file="2024_Nov_Forecast_Output.csv", sep=',', row.names = FALSE)