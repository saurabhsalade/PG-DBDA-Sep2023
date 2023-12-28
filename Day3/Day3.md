# Data Pre-processing

# Step 1: Import the libraries


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

# Step 2: Import dataset


```python
dataset = pd.read_csv('Day3.csv')
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Age</th>
      <th>Salary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>France</td>
      <td>44.0</td>
      <td>72000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>27.0</td>
      <td>48000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>30.0</td>
      <td>54000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spain</td>
      <td>38.0</td>
      <td>61000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>France</td>
      <td>35.0</td>
      <td>58000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spain</td>
      <td>NaN</td>
      <td>52000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>France</td>
      <td>48.0</td>
      <td>79000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>50.0</td>
      <td>83000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>France</td>
      <td>37.0</td>
      <td>67000.0</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Independent variable
X = dataset.iloc[:,:-1].values

#Dependent variable
y = dataset.iloc[:,3].values
```


```python
X
```




    array([['France', 44.0, 72000.0],
           ['Spain', 27.0, 48000.0],
           ['Germany', 30.0, 54000.0],
           ['Spain', 38.0, 61000.0],
           ['Germany', 40.0, nan],
           ['France', 35.0, 58000.0],
           ['Spain', nan, 52000.0],
           ['France', 48.0, 79000.0],
           ['Germany', 50.0, 83000.0],
           ['France', 37.0, 67000.0]], dtype=object)




```python
y
```




    array(['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'],
          dtype=object)



# Step 3: Handling the missing data
SimpleImputer is a scikit-learn class which is helpful in handling the missing data in the predictive model.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

```


```python
X
```




    array([['France', 44.0, 72000.0],
           ['Spain', 27.0, 48000.0],
           ['Germany', 30.0, 54000.0],
           ['Spain', 38.0, 61000.0],
           ['Germany', 40.0, 63777.77777777778],
           ['France', 35.0, 58000.0],
           ['Spain', 38.77777777777778, 52000.0],
           ['France', 48.0, 79000.0],
           ['Germany', 50.0, 83000.0],
           ['France', 37.0, 67000.0]], dtype=object)



# Step 4: Encoding categorical data
Labelencoder: encoding the levels of categorical feature into numeric values

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
```


```python
X
```




    array([[0, 44.0, 72000.0],
           [2, 27.0, 48000.0],
           [1, 30.0, 54000.0],
           [2, 38.0, 61000.0],
           [1, 40.0, 63777.77777777778],
           [0, 35.0, 58000.0],
           [2, 38.77777777777778, 52000.0],
           [0, 48.0, 79000.0],
           [1, 50.0, 83000.0],
           [0, 37.0, 67000.0]], dtype=object)


OnehotEncoder: Encode categorical data using a one-hot-K method.

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columntransformer = ColumnTransformer([('encoder', OneHotEncoder(),[0])], remainder='passthrough')
X = np.array(columntransformer.fit_transform(X),dtype=np.str)
```

    C:\Users\DELL\AppData\Local\Temp\ipykernel_17300\3087217293.py:4: DeprecationWarning: `np.str` is a deprecated alias for the builtin `str`. To silence this warning, use `str` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.str_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      X = np.array(columntransformer.fit_transform(X),dtype=np.str)
    


```python
X
```




    array([['1.0', '0.0', '0.0', '44.0', '72000.0'],
           ['0.0', '0.0', '1.0', '27.0', '48000.0'],
           ['0.0', '1.0', '0.0', '30.0', '54000.0'],
           ['0.0', '0.0', '1.0', '38.0', '61000.0'],
           ['0.0', '1.0', '0.0', '40.0', '63777.77777777778'],
           ['1.0', '0.0', '0.0', '35.0', '58000.0'],
           ['0.0', '0.0', '1.0', '38.77777777777778', '52000.0'],
           ['1.0', '0.0', '0.0', '48.0', '79000.0'],
           ['0.0', '1.0', '0.0', '50.0', '83000.0'],
           ['1.0', '0.0', '0.0', '37.0', '67000.0']], dtype='<U17')



# Step 5: Splitting the dataset into training sets and test sets


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/4, random_state=0)
```


```python
X_train
```




    array([['1.0', '0.0', '0.0', '37.0', '67000.0'],
           ['0.0', '0.0', '1.0', '27.0', '48000.0'],
           ['0.0', '0.0', '1.0', '38.77777777777778', '52000.0'],
           ['1.0', '0.0', '0.0', '48.0', '79000.0'],
           ['0.0', '0.0', '1.0', '38.0', '61000.0'],
           ['1.0', '0.0', '0.0', '44.0', '72000.0'],
           ['1.0', '0.0', '0.0', '35.0', '58000.0']], dtype='<U17')




```python
X_test
```




    array([['0.0', '1.0', '0.0', '30.0', '54000.0'],
           ['0.0', '1.0', '0.0', '50.0', '83000.0'],
           ['0.0', '1.0', '0.0', '40.0', '63777.77777777778']], dtype='<U17')




```python
y_train
```




    array(['Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes'], dtype=object)




```python
y_test
```




    array(['No', 'No', 'Yes'], dtype=object)



# Step 6: Feature Scaling


```python
from sklearn.preprocessing import StandardScaler
scale  = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
```


```python
X_train
```




    array([[ 0.8660254 ,  0.        , -0.8660254 , -0.2029809 ,  0.44897083],
           [-1.15470054,  0.        ,  1.15470054, -1.82168936, -1.41706417],
           [-1.15470054,  0.        ,  1.15470054,  0.08478949, -1.0242147 ],
           [ 0.8660254 ,  0.        , -0.8660254 ,  1.5775984 ,  1.62751925],
           [-1.15470054,  0.        ,  1.15470054, -0.04111006, -0.14030338],
           [ 0.8660254 ,  0.        , -0.8660254 ,  0.93011502,  0.94003267],
           [ 0.8660254 ,  0.        , -0.8660254 , -0.52672259, -0.43494049]])




```python
X_test
```




    array([[ 0.        ,  0.        ,  0.        , -1.22474487, -1.07298811],
           [ 0.        ,  0.        ,  0.        ,  1.22474487,  1.33431759],
           [ 0.        ,  0.        ,  0.        ,  0.        , -0.26132948]])




```python
y_train
```




    array(['Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes'], dtype=object)




```python
y_test
```




    array(['No', 'No', 'Yes'], dtype=object)




```python

```
