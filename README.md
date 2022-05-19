# Prediction-based-on-Load-Profiling
---------------
A predictive model based on load profiling is kind of a data driven model is able to forecast power consumption in building

first step : Load profiling using acutal dataset
second step : Forecasting daily power consumption using linear regression method
third step : Distribute predictive values according to the hourly load profile

test : Daily average temperature is used as input

# Requirements
---------------
```
matplotlib==3.5.1
numba==0.54.0
numpy==1.20.0
pandas==1.4.1
Pillow==9.0.1
scikit-learn==1.0.2
scipy==1.8.0
tslearn==0.5.2
```
User optional
```
input.csv : Actual data
test.csv : Test input
```
