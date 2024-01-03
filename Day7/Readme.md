# Day 7: Decision Tree
-----------------------------------------------
### Date: 03/01/2024
### Topic:
	
	-Decision Tree for classification
	-Decision Tree Regression
	-Random Forest

### Decision Tree:
---------------
    -DT most powerful tool for supervised learning algorithm.
    -DT is used for both classification and Regression.
    -DT looks like a flow chart same as tree structure, where branches denote the IF-THEN rules and the leaf node denotes the result of the algorithm.

### Terminologies in Decision Tree:
------------------------------
    1. Root Node: Starting point of DT
    2. Intermediate node( Decision node): decision making criteria
    3. Leaf node (Terminal node) : indicates the label (categorical/numeric)
    4. Splitting : splitting a node into 2/more sub-nodes(child node).
    5. Sub-Tree (branch) : sub section of the DT.
    6. Parent node: divide the node into child nodes.
    7. Child node: splitted node from intermediate/parent node.
    8. Impurity: measurement of the target variable's homogeneity.
    9. Information Gain: measure the reduction in impurity achieved by splitting a dataset on a particular feature of DT
    10. Pruning : process of removing branches from the tree that do not provide any additional information to overfitting.


### Decision Tree Assumptions:
--------------------------
### Binary Splits:
--------------
Decision trees typically make binary splits, meaning each node divides the data into two subsets based on a single feature or condition. This assumes that each decision can be represented as a binary choice.

### Recursive Partitioning:
-----------------------
Decision trees use a recursive partitioning process, where each node is divided into child nodes, and this process continues until a stopping criterion is met. This assumes that data can be effectively subdivided into smaller, more manageable subsets.

### Feature Independence:
---------------------
Decision trees often assume that the features used for splitting nodes are independent. In practice, feature independence may not hold, but decision trees can still perform well if features are correlated.

### Homogeneity:
-------------
Decision trees aim to create homogeneous subgroups in each node, meaning that the samples within a node are as similar as possible regarding the target variable. This assumption helps in achieving clear decision boundaries.

### Top-Down Greedy Approach:
---------------------------
Decision trees are constructed using a top-down, greedy approach, where each split is chosen to maximize information gain or minimize impurity at the current node. This may not always result in the globally optimal tree.

### Categorical and Numerical Features:
-----------------------------------
Decision trees can handle both categorical and numerical features. However, they may require different splitting strategies for each type.

### Overfitting:
------------
Decision trees are prone to overfitting when they capture noise in the data. Pruning and setting appropriate stopping criteria are used to address this assumption.

### Impurity Measures:
-------------------
Decision trees use impurity measures such as Gini impurity or entropy to evaluate how well a split separates classes. The choice of impurity measure can impact tree construction.

No Missing Values:
-------------------
Decision trees assume that there are no missing values in the dataset or that missing values have been appropriately handled through imputation or other methods.

### Equal Importance of Features:
-----------------------------
Decision trees may assume equal importance for all features unless feature scaling or weighting is applied to emphasize certain features.

### No Outliers:
-------------
Decision trees are sensitive to outliers, and extreme values can influence their construction. Preprocessing or robust methods may be needed to handle outliers effectively.

### Sensitivity to Sample Size:
------------------------------
Small datasets may lead to overfitting, and large datasets may result in overly complex trees. The sample size and tree depth should be balanced.

# Home Work

## Problem Statement: Income Prediction using Regression and Classification Techniques

### Machine Learning Problem Statement: Income Prediction using Regression and Classification Techniques

### Objective:
    Develop a machine learning model to analyze a dataset and predict an individual's income based on various demographic and socio-economic features. The task involves using both regression and classification techniques to predict the continuous income amount and classify individuals into income categories, such as above or below a certain threshold.

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
    SALARY (Target variable)

### Tasks:

### Data Exploration:

    Explore the dataset to understand the distribution of features.
    Visualize the relationships between different features.
    Analyze the distribution of the target variable (SALARY).
### Data Preprocessing:

    Handle missing values, outliers, and any data inconsistencies.
    Convert categorical variables like WORKCLASS, EDUCATION, MARITALSTATUS, OCCUPATION, RELATIONSHIP, RACE, SEX, and NATIVECOUNTRY into a format suitable for both regression and classification (e.g., one-hot encoding for classification).
### Regression Technique:

    Implement a regression model to predict the continuous income amount.
    Evaluate the regression model's performance using appropriate metrics (e.g., mean squared error).
### Classification Technique:

    Implement a classification model to classify individuals into income categories (e.g., above or below a certain threshold).
    Evaluate the classification model's performance using appropriate metrics (e.g., accuracy, precision, recall, F1 score).
### Ensemble Techniques (Optional):

    Explore ensemble techniques, such as random forests or gradient boosting, to improve the overall predictive performance.
### Hyperparameter Tuning:

    Optimize hyperparameters for both the regression and classification models to improve their performance.
### Result Analysis:

    Analyze the predictions of both models to understand how well they generalize to new data.
    Interpret the coefficients and their significance in the regression model.
### Deployment (Optional):

    If applicable, deploy the trained models for making predictions on new data.
### Monitoring and Updating (Optional):

    If deployed, establish a system for monitoring the models' performance over time.
    Plan for model updates or retraining based on new data.
