# Day 10: SVM Algorithm
-----------------------------------------------
### Date: 10/01/2024
### Topic:


### Support Vector Machine:(SVM):
----------------------------
	-powerful machine learning algorithm used for linear and nonlinear classification, regression and even outlier detection tasks.
	-used classification  for both linear and non-linear.
	-uses non-linear mapping to transform the original training dataset into higher dimensions.
	-SVM is a supervised learning where hyperplane with maximum separating decision boundary is identified between different target classes.


	
### Best hyperplane:
-----------------
	-There can be infinite number of hyperplanes passing through data points and classifying the target classes perfectly.
	-Hence SVM finds the maximum margin between the hyperplanes that means maximum distance between the two classes.
	-Margin is indicated by 'd' i.e., d= d1+d2, where d1 and d2 are the margins of left and right side of the hyperplane.
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/cab90d52-ab5b-4911-929d-60896aa9dc60)

	
### Types of SVM:
--------------
	1. Linear SVM:
		-data is perfectly linearly separable.
		-datapoints can be classified into 2 separate classes by using strainght line.(specificaly in 2-D)
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/ea2e4c8f-d620-48e2-a2b4-7a73133d84d6)

	
	2. Non-linear SVM:
		-data is not linearly separable, then prefered to use non-linear SVM.
		-For non-linear separation, kernel techniques can be used to separate datapoints into differnt clases.
		
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/2c1c06fa-0473-4d97-bf74-f54a390f5494)

### Terminologies:
----------------
### Hyperplane: 
	-The decision boundary i.e., used to separate the data points of different classes in a feature space.
			
### Support Vector:
	-Data points that are closed to the hyperplane with a separating line.
	
### Margin:
	-The distance between the hyperplane and the observed closest support vector to the hyperplane.
	
### Hard Margin:
	-Maximum margin in hyperplane which separate the clasess without any misclassification.
	
### Soft Margin:
	-Maximum margin in hyperplane which separates the classes with misclassification.
	-Data points is not perfectly separated or contains outliers, then SVM permits a soft margin technich.
	-Data points includes 'slack variable' introduced by soft margin formulation.
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/f49d2159-6d4a-4fe4-9570-613dae4d2ed9)

### Slack variable:
	-It comprises between the margin and reducing violotions.
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/f7e8289e-9b00-4edd-87d9-b85641afc91b)
		
		
### Kernel Function:
	-It is the technique used to handle the non-linear separation.
	-It is a method used to take data as input and transform it into the required form of processing data.
	

### Types of Kernel function:
-----------------
	-The kernel function is applied on each data instance to map the original non-linear observations into a higher-dimensional space in which they are separable.
	-Linear 
	-Polynomial
	-Gaussian
	-Sigmoid
	-Neural net
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/bf7baecd-fd50-4a94-b97a-6694d8d1a78c)

![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/016d19f4-de46-4c37-af3e-13ca7430d3d8)


![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/0548fed0-3d4d-4ff1-aa64-bde928cfcac0)

