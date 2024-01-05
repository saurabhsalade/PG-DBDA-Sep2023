# Day 8: KNN
-----------------------------------------------
## Date: 03/01/2024
## Topic:
	
	-k-NN
	-Naive Bayes

### k-NN Algorithm:
----------------
	-simplest machine learning technique
	-assume similarity between the new testing data and available data. 
	-It always put new  testing data into the category that is most similar to the available categories.
	-used for regression and classification.
	-non-parametric algorithm, which means it does not make any assumption on underlying data.
	-It is lazy learner algorithm, because it stores the does not learn from the trainingdataset immediately instead it stores the data and at the time of classification, it performs an action on the dataset.

 ![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/bee19d37-1abe-40fb-8a20-5b7ee10c0710)

	
### Distance Measures:
-------------------
	1. Euclidean distance
	
	2. Manhatten distance
	
	3. Minkowski distance
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/e9f6452b-b9e8-4396-8c00-4e5361e32fc8)

### How to choose the value of k:
-------------------------------
	For k-NN algorithm:
		-k=very crucial in KNN algorithm to define the number of neighbours in the algorithm.
		-k=based on an input data 
		-if input data has more outliers or noise, a higher value of k would be better.
		-It is recommended to choose an odd value for it, to avoid ties inclassification.
		-Cross validation can also help in selecting the best k-value for the given dataset.
		
### Cross validation:
------------------
	-Technique used to evaluate the performance of a model on unseedn data.
	
### Different techniques used in cross-validation:
-----------------------------------------------
		1. Hold out validation:
			-Training data =50%
			-Testing data=50%
			-simple and quick model building
		
		2. :LOOCV (Leave one out cross validation)
			-Training data = 100%
			-Testing data = leaves one data
			-low bias data(beause using all datapoints)
		
		3. Stratified cross validation
			-Training data = data is divided into folds (100%)
			-Testing data = unseen data
		
		4. K-fold cross validation
			-Training data = divide the data into 'k' folds: 100%
			-Testing data = one subset from each fold will be use  for testing.

   ![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/ffc89e26-785a-46f7-8976-8b2835dd0aaf)

			
# Naive Bayes: 
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/90468fc6-42d3-4af2-9d48-c9471425f272)

## -Bayes theorem:
---------------
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/ee76f81e-9e5d-447f-95f8-4b44a026c180)

    Prior: The probability 'A' (Knowledge)
    Likelihood: The probability of 'B' given by 'A'
    Margin:The probability of 'B'
    Posterior:The probability of 'A' given by 'B'

### Classification Techniques:
-------------------------
    1. Statistical based method:
    	-Regression
    	-Baysian Classifier
    	
    2. Distance based classification:
    	-K-NN algorithm
    	
    3. Desicion Tree Classification
    	-ID3, CART, C 4.5
    
    4. Hyperplane classification
    	-SVM
    	
    5. Neural Network classification
    	-ANN

### Example:
    Link : https://aihubprojects.com/naive-bayes-algorithm-from-scratch/

### Types of distribution:
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/e882703a-5fee-4c12-994a-347bd002e214)

### Gaussian distribution:
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/82a6511c-dc04-4b08-a239-3333c6b8d2b6)

### Bernoli's distribution:
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/d39d9f12-8d84-4c50-9da1-e29a083cf984)

### Multinominal distribution:
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/6032e9c0-c2a1-411b-81c3-3d9d00297d3f)



# Homework
### Loan Defaulters
```
Home Owner	Marital Status	Annual Income	Defaulted Borrower
Yes	Single	$125,000	No
No	Married	$100,000	No
No	Single	$70,000	No
Yes	Married	$120,000	No
No	Divorced	$95,000	Yes
No	Married	$60,000	No
Yes	Divorced	$220,000	No
No	Single	$85,000	Yes
No	Married	$75,000	No
No	Single	$90,000	Yes
```
