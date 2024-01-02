# Day 6: Logistic Regression
-----------------------------------------------
## Date: 02/01/2024
## Topic:
	-Logistic Regression
	-Concept of Classification
	-Measures for classification

## Logistic Regression:
--------------------
	-Supervised learning
	-used for classification (binary classification)
	-logit function : probability that an instance belongs to a given class
	-powerful tool for decision making
	-statistical algorithm, which analyze the relationship between IV and DV
	-Binary classification with categorical and discrete data
	-activation function used is sigmoid function 
	-Type of data handled are: yes/No, Male/Female, 0/1, True/False
	
## Sigmoid function:
------------------
	-activation function, mathematical function
	-used to predict 0 or 1 , so that it can for 'S' curve
	-use threshold value which separates the probability either 0 or 1 

## Terms involved in Logistic Regression:
---------------------------------------
	-Independent variable:1/ more feature 
	-Dependent variable:target class/ labels for class
	-odds : ratio of occurring to not occuring i.e., (p/(1-p))
	-Log-odds: Logit function: natural log of the odds
	-coeff:weightd of features
	-Maximum Likelihood Estimation : MLE
		-method used to estimate the coefficient of the logistic regression model, which maximizes the likelihood of observing the data given in the model.
		
## Data used in Logistic Regression:
---------------------------------
	-Binomial : 2 possible values e.g., 0/1, Yes/No
	-Multinominal: 3/more possible dependent variables e.g. Age={young, middle, aged}
	-Ordinal: 3/more possible orderd type of dependent variable. e.g., (Poor, Good, Very good}
	
## Steps for Model building:
--------------------------
    1. Define the problem statement
    2. Get the Data
    3. Data preprocessing
    4. EDA
    5. Feature Selection
    6. Model Building
    7. Evalaution Measures
    8. Model Improvement (Fine tuning)
    9. Model deployment
    10. Conclusion 
        
# Home-work:

## Problem statement :
### Machine Learning Problem Statement: Income Prediction using Logistic Regression

### Objective:
    Develop a machine learning model to analyze a dataset and predict whether an individual's income is above or below $50,000 based on various demographic and socio-economic features. The model should utilize logistic regression and explore the suitability of different types of regression for this classification task.

### Dataset:
    The dataset contains the following parameters:

    AGE
    WORKCLASS
    FNLWGT
    EDUCATION
    EDUCATIONNUM
    MARITALSTATUS
    OCCUPATION
    RELATIONSHIP
    RACE
    SEX
    CAPITALGAIN
    CAPITALLOSS
    HOURSPERWEEK
    NATIVECOUNTRY
    ABOVE50K (Target variable)

### Tasks:

### Data Exploration:

    Explore the dataset to understand the distribution of features.
    Visualize the relationships between different features.
    Analyze the distribution of the target variable (ABOVE50K).
### Data Preprocessing:

    Handle missing values, outliers, and any data inconsistencies.
    Convert categorical variables like WORKCLASS, EDUCATION, MARITALSTATUS, OCCUPATION, RELATIONSHIP, RACE, SEX, and NATIVECOUNTRY into a format suitable for logistic regression (e.g., one-hot encoding).
### Feature Selection:

  Choose relevant features for predicting whether an individual's income is above or below $50,000.
### Logistic Regression:

  Implement a logistic regression model to predict the binary outcome of the ABOVE50K variable.
  Evaluate the model's performance using appropriate metrics (e.g., accuracy, precision, recall, F1 score).

### Evaluation Metrics:

  Utilize appropriate metrics for evaluating classification models, considering the nature of the task.
### Result Analysis:

    Analyze the model's predictions to understand how well it generalizes to new data.
    Interpret the coefficients and their significance in the logistic regression model.
### Deployment (Optional):

    If applicable, deploy the trained logistic regression model for making predictions on new data.

