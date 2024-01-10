# Day 13: Unsupervised Learning
-----------------------------------------------
### Date: 10/01/2024
### Topic:
	-Association
	-Frequent PAttern Analysis
	-Time series Analysis


### Frequent Pattern:
-----------------
	-a pattern 
	-set of items, subsequences,substructures etc.
	-that occurs frequently in a data set.
	
	-Used to find inherent regularities in the dataset.
	-i.e hidden pattern of dataset.

### Frequent Pattern Growth Rule:
-----------------------------

![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/0778b7c0-f2c5-4ee6-a02b-80f1e671a67f)

### Difference between Apriori and FP Growth:
----------------------------------------------
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/8638bb72-46bf-4530-9029-c40c42ccff2e)



### Time Series Analysis:
-----------------------
	-The analysis of data organized across units of time.
	
	-It is sresearch design problem statement.
	
	-Data for onr or more  variables is collected for many observations at different time periods.
		-Univariate: one variable description
		-Multivariate: casual explanation
		
### Steps to analyze the Time series:
-------------------------------------

    1. Collecting the data and clean it
    2. Prepare visualization withrespect to time vs key feature
    3. Observing the stationary data series
    4. Developing charts to understand its nature
    5. Model building - AR, MA, ARMA, ARIMA
    6. Extracting inference from the prediction.

### Data types in Time Series Model:
--------------------------------
	1. Stationary:
		-data must follow the thum rules without having tren, seasonality, cyclic and irregularity componenet of the time series.
		
			
	2. Non-stationary:
		-It will be the mean-variance or covariance, which is changing with respect to the non-stationary dataset.
		
### Method for stationary dataset:
------------------------------
	1. Augmented-Dickey -Fuller (ADF)
	2. Kwiatkowski-Phillips -Schmidt-Shin (KPSS)
	
### Method for non-tationary dataset:
-----------------------------------
	1. ARIMA
					AR + I + MA
		AR: Autoregression
		I:  Integrated
		MA: Moving average
		
### TSA used for:
-------------
	-forecasting predictions outcomes are based the statistical concept of serial coorelation, where past data points influence future data points.
	
### Advantage of TSA:
-----------------
	-Good for short term forecasting
	-Requirement for historial data
	-handle non-stationary data
	
### Limitation of TSA:
-------------------
	-Not built for long term forecasting/predictions.
	-computationaly expensive model
	-Model parameters are subjective in nature.
