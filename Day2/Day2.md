# Numpy Library


```python
import numpy as np
import pandas as pd
```


```python
#Datatypes

dtypes = pd.DataFrame(
    {
        'Type': ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64', 'float128', 'complex64', 'complex128', 'bool', 'object', 'string_', 'unicode_'],
        'Type Code': ['i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f2', 'f4 or f', 'f8 or d', 'f16 or g', 'c8', 'c16', '', 'O', 'S', 'U']
    }
)
dtypes
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
      <th>Type</th>
      <th>Type Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>int8</td>
      <td>i1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>uint8</td>
      <td>u1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>int16</td>
      <td>i2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>uint16</td>
      <td>u2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>int32</td>
      <td>i4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>uint32</td>
      <td>u4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>int64</td>
      <td>i8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>uint64</td>
      <td>u8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>float16</td>
      <td>f2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>float32</td>
      <td>f4 or f</td>
    </tr>
    <tr>
      <th>10</th>
      <td>float64</td>
      <td>f8 or d</td>
    </tr>
    <tr>
      <th>11</th>
      <td>float128</td>
      <td>f16 or g</td>
    </tr>
    <tr>
      <th>12</th>
      <td>complex64</td>
      <td>c8</td>
    </tr>
    <tr>
      <th>13</th>
      <td>complex128</td>
      <td>c16</td>
    </tr>
    <tr>
      <th>14</th>
      <td>bool</td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td>object</td>
      <td>O</td>
    </tr>
    <tr>
      <th>16</th>
      <td>string_</td>
      <td>S</td>
    </tr>
    <tr>
      <th>17</th>
      <td>unicode_</td>
      <td>U</td>
    </tr>
  </tbody>
</table>
</div>




```python
arr = np.array([1,2,3], dtype='f4')
arr
print(arr.dtype)
```

    float32
    


```python
#Create an array
arr = np.array(range(10))
arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
#create an evenly spaced array
arr = np.arange(0,20,2)
arr
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])




```python
#create an array evenly spaced in specified interval
arr = np.linspace(0,10,25)
arr
```




    array([ 0.        ,  0.41666667,  0.83333333,  1.25      ,  1.66666667,
            2.08333333,  2.5       ,  2.91666667,  3.33333333,  3.75      ,
            4.16666667,  4.58333333,  5.        ,  5.41666667,  5.83333333,
            6.25      ,  6.66666667,  7.08333333,  7.5       ,  7.91666667,
            8.33333333,  8.75      ,  9.16666667,  9.58333333, 10.        ])




```python
#create the array of random values
arr = np.random.rand(3,3)
arr
```




    array([[0.78219908, 0.87953446, 0.11850086],
           [0.88495848, 0.8263806 , 0.21830464],
           [0.67545699, 0.68943854, 0.47023393]])




```python
#create an array of zeros
zeros = np.zeros((2,3), dtype='i4')
zeros
```




    array([[0, 0, 0],
           [0, 0, 0]])




```python
#create an array of ones
ones = np.ones((2,3))
ones
```




    array([[1., 1., 1.],
           [1., 1., 1.]])




```python
empty = np.empty_like(arr)
empty
```




    array([[0.78219908, 0.87953446, 0.11850086],
           [0.88495848, 0.8263806 , 0.21830464],
           [0.67545699, 0.68943854, 0.47023393]])




```python
#Create an array with constant value
a1 = np.full((2,3), 5)
a1

```




    array([[5, 5, 5],
           [5, 5, 5]])




```python
#create an array using repetation
arr = [0,1,2]
np.repeat(arr,3)
```




    array([0, 0, 0, 1, 1, 1, 2, 2, 2])




```python
#Create identity matrix
identity = np.eye(5)
identity
```




    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])




```python
#Create 2D matrix
arr = np.array([[1,2,3],[4,5,6],[1,2,3],[4,5,6]])
arr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [1, 2, 3],
           [4, 5, 6]])




```python
np.info(arr)
```

    class:  ndarray
    shape:  (4, 3)
    strides:  (12, 4)
    itemsize:  4
    aligned:  True
    contiguous:  True
    fortran:  False
    data pointer: 0x2bd27871180
    byteorder:  little
    byteswap:  False
    type: int32
    


```python
#Sampling method
#Set seed
a1 = np.random.seed(123)
a1
```


```python
# set random independent state 
rs = np.random.RandomState(321)
rs.rand(10)
```


```python
# Genrate a random sample frominterval [0,1] 
np.random.rand()
```


```python
#1-D
np.random.rand(3)
```


```python
#2-D
np.random.rand(3,3)
```


```python
np.random.randint(1,10,2, dtype='i8')
```


```python
#create float random values
np.random.rand(10)
```


```python
np.random.ranf(10)
```


```python
#Create 2-D array
arr = np.array([[1,2,3],[4,5,6],[7,8,9],[4,7,6]])
arr
```


```python
np.info(arr)
```


```python
arr.dtype
```


```python
arr.shape
```


```python
len(arr)
```


```python
arr/10
```


```python
arr/0
```


```python
#Exponential value
np.exp(arr)
```


```python
np.log(arr)
```


```python
np.log10(arr)
```


```python
np.log2(arr)
```


```python
np.sqrt(arr)
```


```python
np.sin(arr)
```


```python
np.cos(arr)
```


```python
np.sum(arr)
```


```python
np.sum(arr, axis=0)#Row-wise sum
```


```python
np.sum(arr, axis=1)#Column wise sum
```


```python
np.max(arr)
```


```python
np.max(arr, axis = 0)
```


```python
np.max(arr, axis = 1)
```


```python
#Sort an array
arr = np.random.rand(5,5)
arr
```


```python
np.sort(arr, axis=0)
```


```python
np.sort(arr, axis=1)
```


```python
np.transpose(arr)
```


```python
#Flatterning an array
arr.flatten()
```


```python
arr1 = np.random.rand(5,5)
arr2 = np.random.rand(5,5)
```


```python
arr1
```


```python
arr2
```


```python
#Method 1
arr1.dot(arr2)
```


```python
#Method 2
np.dot(arr1,arr2)
```


```python
#Method 3
arr1 @ arr2
```


```python
#Eigen values
arr = np.random.rand(5,5)
arr
```


```python
w, v = np.linalg.eig(arr)
```


```python
print(w)
print(v)
```


```python
np.linalg.det(arr)
```


```python
np.linalg.inv(arr)
```


```python
#Linear Equation
y = [1,2,3,4,5]
solution, residual, rank, singular = np.linalg.lstsq(arr,y)

```


```python
print(solution)
```


```python
print(residual)
```


```python
print(rank)
```


```python
print(singular)
```

# Pandas Library


```python
import pandas as pd
```


```python
data1 = {
    'day': ['1/1/2017','1/2/2017','1/3/2017','1/4/2017','1/5/2017','1/6/2017'],
    'temperature': [32,35,28,24,32,31],
    'windspeed': [6,7,2,7,4,2],
    'event': ['Rain', 'Sunny', 'Snow','Snow','Rain', 'Sunny']
}
```


```python
df = pd.DataFrame(data1)
df
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2017</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/4/2017</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2017</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1/6/2017</td>
      <td>31</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head()
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2017</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/4/2017</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2017</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/4/2017</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2017</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1/6/2017</td>
      <td>31</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[1:3]
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[1:-3]
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```


```python
df['day']
```




    0    1/1/2017
    1    1/2/2017
    2    1/3/2017
    3    1/4/2017
    4    1/5/2017
    5    1/6/2017
    Name: day, dtype: object




```python
df[['day','temperature']]
```


```python
#Calculate the maximum temperature in the day
df['temperature'].max()
```


```python
#Calculate the temperature greater that 31 
df[df['temperature']>31]
```


```python
#Calculate the day in which temperature is 32
df[df['temperature']==32]
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2017</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2017</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Calculate the event on which temparature was minimum.
df[df['temperature']==df['temperature'].min()]
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>1/4/2017</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Setting an index
df.set_index('day', inplace=True)
```


```python
df.index
```




    Index(['1/1/2017', '1/2/2017', '1/3/2017', '1/4/2017', '1/5/2017', '1/6/2017'], dtype='object', name='day')




```python
df.loc['1/1/2017']
```




    temperature      32
    windspeed         6
    event          Rain
    Name: 1/1/2017, dtype: object




```python
df.reset_index(inplace=True)
df.head()
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2017</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/4/2017</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2017</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.set_index('event',inplace=True)
df

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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
    </tr>
    <tr>
      <th>event</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rain</th>
      <td>1/1/2017</td>
      <td>32</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Sunny</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Snow</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Snow</th>
      <td>1/4/2017</td>
      <td>24</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Rain</th>
      <td>1/5/2017</td>
      <td>32</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Sunny</th>
      <td>1/6/2017</td>
      <td>31</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['Sunny']
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
    </tr>
    <tr>
      <th>event</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sunny</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Sunny</th>
      <td>1/6/2017</td>
      <td>31</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Example: Read CSV file
import pandas as pd
```


```python
df = pd.read_csv("D:/Test/Day11.csv")
df
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2017</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/4/2017</td>
      <td>24</td>
      <td>7</td>
      <td>Snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2017</td>
      <td>32</td>
      <td>4</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1/6/2017</td>
      <td>31</td>
      <td>2</td>
      <td>Sunny</td>
    </tr>
  </tbody>
</table>
</div>




```python
#HW: Convert dictionary, Tuple and list data into Dataframe
```


```python
#Reading Excel file
df=pd.read_excel("Day12.xlsx","Sheet1")
df
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-01-01</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-01-02</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-01-03</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Reading Dictionary
import pandas as pd
data2 = {
    'day': ['1/1/2017','1/2/2017','1/3/2017'],
    'temperature': [32,35,28],
    'windspeed': [6,7,2],
    'event': ['Rain', 'Sunny', 'Snow']
}
df = pd.DataFrame(data2)
df
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2017</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Reading tuples
data3 = [
    ('1/1/2017',32,6,'Rain'),
    ('1/2/2017',35,7,'Sunny'),
    ('1/3/2017',28,2,'Snow')
]
df = pd.DataFrame(data=data3, columns=['day','temperature','windspeed','event'])
df
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2017</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Reading Multiple dictionaries
data4 = [
    {'day': '1/1/2017', 'temperature': 32, 'windspeed': 6, 'event': 'Rain'},
    {'day': '1/2/2017', 'temperature': 35, 'windspeed': 7, 'event': 'Sunny'},
    {'day': '1/3/2017', 'temperature': 28, 'windspeed': 2, 'event': 'Snow'},
    
]
df = pd.DataFrame(data=data4, columns=['day','temperature','windspeed','event'])
df
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
      <th>day</th>
      <th>temperature</th>
      <th>windspeed</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2017</td>
      <td>32</td>
      <td>6</td>
      <td>Rain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2017</td>
      <td>35</td>
      <td>7</td>
      <td>Sunny</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2017</td>
      <td>28</td>
      <td>2</td>
      <td>Snow</td>
    </tr>
  </tbody>
</table>
</div>




```python
HW: Design an Student Result Analysis Application
```

# Series data structure


```python
obj = pd.Series([1,"Anish", 3.5, "Niraj"])
obj
```




    0        1
    1    Anish
    2      3.5
    3    Niraj
    dtype: object




```python
obj[1]
```




    'Anish'




```python
obj.values
```




    array([1, 'Anish', 3.5, 'Niraj'], dtype=object)




```python
obj1 = pd.Series([1,"Anish", 3.5, "Niraj"], index=["a","b","c","d"])
obj1
```




    a        1
    b    Anish
    c      3.5
    d    Niraj
    dtype: object




```python
obj1["b"]
```




    'Anish'




```python
#Example:
Marks = {"Nayan":90,"Prajwal":89,"Saket":75,"Ranjeet":78,"Mayur":98}
Result = pd.Series(Marks)
Result
```




    Nayan      90
    Prajwal    89
    Saket      75
    Ranjeet    78
    Mayur      98
    dtype: int64




```python
#Print the marks for Ranjeet
Result["Ranjeet"]
```




    78




```python
#Print the marks > 80
Result[Result>80]
```




    Nayan      90
    Prajwal    89
    Mayur      98
    dtype: int64




```python
#Assign Mayur = 60 marks
Result["Mayur"]=60
```


```python
Result
```




    Nayan      90
    Prajwal    89
    Saket      75
    Ranjeet    78
    Mayur      60
    dtype: int64




```python

```
