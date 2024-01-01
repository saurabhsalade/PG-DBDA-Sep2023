
# Day 5: Regression
-----------------------------------------------
# Date: 01/01/2024
# Topic:
	-Regression
	-Types of Regression
		-Simple Linear Regression
		-Multiple Linear Regression
		-Polynomial Linear Regression
		-Ridge Regression
		-Lasso Regression
		-Elastic Net Regression
	-Logistic Regression
	
	
# Overfitting:
------------
	-phenomenon that occurs when a machine learning model is conatrained to the training set and not able to perform well on unseen data.
	-when model learns the noise in the training data as well, cases will be remembered by the training data set.
	-Result: Training accuracy increase, Testing accuracy decreases.
	

# Underfitting:
------------- 
	-phenomenon that occurs when a machine learning model is notable to learn even the basic patterns available in the dataset.
	-such model is unable to perform well even ontraining dataset.
	-Result: Training and Testing accuracy decreases.
	
	
# Bias:
----- 
	-refers to the error which occurs when we try to fit perfectly.
	-data fit will be high bias, where is unable to learn the patterns in the data and hence perform poorly.
	
# Variance:
---------
	-refers to the error which occurs when we try to make predictions using the unseen data.
	-data fit with high variance, where model learn noise that present in the data.

# Regularization:
---------------
	- technique used to reduce errors by fitting the function appropriately on the given training set & avoiding overfitting.

# Interview Questions
---------------------
    1. What are the different types of regression.
    2. Is regression a supervised learning? Why?
    3. Explain the ordinary least squares method for regression.
    4. What are linear, multinomial, and polynomial regressions.
    5. If model used for regression is y = a + b(x − 1)2; is it a multinomial regression? If not, what type of regression is it?
    6. What does the line of regression tell you?

# Problem statement:
----------------------
### Machine Learning Problem Statement: Iris Flower Species Prediction using Linear Regression

## Objective:
    Develop a machine learning model to analyze the Iris dataset and predict the species of iris flowers based on various features. The model should utilize linear regression and explore the suitability of different types of regression for this classification task.

### Dataset:
    The Iris dataset consists of measurements for three species of iris flowers (setosa, versicolor, and virginica). The features include sepal length, sepal width, petal length, and petal width.

## Tasks:

### Data Exploration:

    Explore the dataset to understand the distribution of features.
    Visualize the relationships between different features.
  
## Data Preprocessing:

    Check for missing values and outliers.
    Ensure that the data is suitable for linear regression (numeric features, linear relationships).

## Feature Selection:

    Choose relevant features for predicting iris species.
    Consider feature scaling if necessary.

# Linear Regression:

    Implement a linear regression model to predict the iris species based on the chosen features.
    Evaluate the model's performance using appropriate metrics (e.g., mean squared error).
    
# Type of Regression:

    Explore different types of regression models to compare their performance with linear regression.
    Select the most suitable regression model for the Iris species prediction task.
    
###  Hyperparameter Tuning:

    Optimize hyperparameters for the selected regression model to improve performance.
    
### Evaluation Metrics:

    Utilize appropriate metrics for evaluating regression models, considering the nature of the task.
    
### Result Analysis:

    Analyze the model's predictions to understand how well it generalizes to new data.
    Interpret the coefficients and their significance in the linear regression model.
    
### Visualization (Optional):

    Optionally, visualize the regression line and scatter plots to better understand the relationships between features and species.

### Deployment (Optional):

    If applicable, deploy the trained regression model for making predictions on new data.

### Monitoring and Updating (Optional):

    If deployed, establish a system for monitoring the model's performance over time.
    Plan for model updates or retraining based on new data.
