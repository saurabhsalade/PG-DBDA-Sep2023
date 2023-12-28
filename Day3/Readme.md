
# Day 3: Data Modelling
-----------------------------------------------
### Date: 28/12/2023
### Topic:
	-Data
	-Types of Attributes
	-Preprocessing
	-Transformations
	-Measures
	-Visualization

### ML Application Development:
---------------------------
    1. Task: Problem definition
    2. Collection of data
    3. Clean & process that data (Pre-processing)
    4. Feature Transformation
    5. Feature Selection
    6. Learning algorithm
    7. Evaluation, Visualization and Interpretation---> (Hidden information)
    8. Conclude

### Data: collection of data objects and their attributes
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/01e81091-32ae-4aa0-b101-ea0723f849ef)

### Types of Data:
    -Numerical : Made up of numbers
    	-Continuous: infinite real numbers
    	-Discrete: finite
    	
    -Categorical : Made up of words
    	-Ordinal : Distinct & ordered
    		Eg: eye color, zip codes
    	
    	-Nominal :Distinct
    		Eg:Ranking list
    		
    	-Interval : Range
    		Eg: Temperature
    		
    	-Ratio : All
    		Eg: Girls-Boys ratio
    	
    	Ex: Age: 15 to 78yrs
    		-<20yrs : Teen
    		-20 to 45yrs : Yng
    		->40yrs : Adult

### Types of dataset:
-----------------
  	-Record
  		-Data Matrix
  		-Document data
  		-Transaction data
  	-Graph
  		-World wide web
  		-Molecular data structure
  	-Ordered
  		-Spatial data
  		-Temporal(time series) data
  		-Sequential data
  		-Genetic data
  		
### Knowledge Discovery Process:
-------------------------------
    1. Goal:
    	-understanding the application domain and goals of KDD effort
    2. Data selection,acquisition, integration
    3. Data cleaning
    	-noise, missing data, outliers, etc.
    4. Exploratory data analysis
    	-Dimensionality reduction, transformation
    	-selection of appropriate model for analysis, hypothesis testing
    5. Data mining/Machine Learning
    	-selecting appropriate method that match set goals ( classification, regression,clustering)
    6. Testing and verification
    7. Interpretation
    8. Conclusions
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/021b520a-e93d-49f8-9ef5-28a3b8550dd3)

### Data Pre-processing steps:
----------------------
    1. Get the dataset
    2. Import libraries
    3. Import dataset
    4. Analyse the data & identify the missing data
    5. Apply feature transformation
    6. Split the dataset into training and testing data
    
 ![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/72158265-ee60-45f9-adec-6ed66fbcb2f6)
   
    
### Feature Transformation:
-----------------------------
  #### Label Encoding:
     Label encoding is a technique used in machine learning and data analysis to convert categorical variables into numerical format. It is particularly useful when working with algorithms that require numerical input, as most machine learning models can only operate on numerical data.
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/b838612d-c0ce-434e-9815-1309906d7d49)

  #### Onehot Encoding: 
    One Hot Encoding is a technique that is used to convert categorical variables into numerical format.
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/060dd00c-094f-4039-a9eb-d7924fd0b711)

### Feature Scaling:
-----------------
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/c8de4131-de9e-411d-8343-c43ace77a644)

![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/d40feda0-01af-4713-9d15-d7e01a173524)

# Homework:
------------

## Machine Learning Problem Statement: Titanic Survival Prediction

## Objective:
	Develop a machine learning model to analyze the Titanic dataset and predict passenger survival based on various features. The model should utilize techniques such as simple imputation for handling missing values, label encoding and one-hot encoding for categorical variables, and standardization for numerical features.

### Dataset:
	The dataset contains information about Titanic passengers, including features such as age, gender, class, ticket fare, and whether the passenger survived or not (target variable).

## Tasks:

### Data Exploration:

	Explore the dataset to understand the distribution of features.
	Identify missing values and outliers.
 
### Data Preprocessing:

	Handle missing values using simple imputation for features like age and embarked.
	Apply label encoding to convert categorical variables like gender into numerical representations.
	Use one-hot encoding for categorical variables with more than two categories, such as embarked.
	Standardize numerical features like age and fare to bring them to a common scale.
 
### Feature Engineering:

	Explore the creation of new features or interactions that might enhance predictive power.
	Consider combining or transforming existing features if necessary.
