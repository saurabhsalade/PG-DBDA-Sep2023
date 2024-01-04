# Day 8: Random Forest
-----------------------------------------------
### Date: 03/01/2024
### Topic:
	
	-Decision Tree
	-Random Forest
	-Ensemble Learning
	
### Random Forest:
--------------
	-supervised machine learning, where there is a labeled target variables are used to classify.
	-used for solving regression(numeric target variable) and classification (categorical target variable) problems can be solved.
	-Generaly ensemble learning methods are combined to do the predictions.
	-Each of the smaller model in the random forest ensemble is a decision tree.
	

### Difference Between DT and RF:
------------------------------
### Decision Tree:
----------------
	1. Single tree
	2. Computation of single tree is faster
	3. majoritiliy face overfitting problem, which allows tree to grow without any control
	4. Dataset with features is taken as input and if-then rules can make predictions

### Random Forest:
----------------
	1. Multiple decision tree
	2. Computation of multiple decision is complex.
	3. created with the subsets od data and final output is based on avergae value or majority voting.
	4. Randomly selects observation and builds decision tree and ensemble learning techings will be used to do predictions.
	
### Ensemble learning:
--------------------
#### Techniques used in Ensemble learning:
	1. Bagging
		-Boostrap Aggregation
		-method involves training multiple models on random subsets of the training dataset
		-predictions from the inidividual models are combined by averaging.
		
	2. Boosting
		-method involves training a sequence of models, where each subsequent model focus on the errors made by the previous models.
		-predictions are combined using weighted voting method.
		
	3. Stacking
		-method involves using the predictions from the set of models as input feature from another model.
		-predictions is made by the next level.
		
	4. Random subset
		-method used with multiple models on random subsets of fetures.
		
	5. XGBoost:
		-optimized distributed gradient boosting library designed for efficient and scalable training of machine learning models.

# Visualising the Training set results
```
from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
					 np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier1.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
			 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

# Homewor:

Solved Example of Decision Tree:
Link 1: https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8
Link 2: https://www.vtupulse.com/machine-learning/id3-algorithm-decision-tree-solved-example/

### Solved Example: 
![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/4af6c961-1fda-45a6-949a-7ae6e8c5c568)

![image](https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/assets/72081819/58d05fca-fa9e-42ce-a237-328bf4d99afd)


### Question: Implement the Decision Tree and Random Forest for the following dataset.

Dataset: https://github.com/Kiranwaghmare123/PG-DBDA-Sep2023/blob/main/Dataset/classification.csv

