# Dataset:

### Ex:1
    dtypes = pd.DataFrame(
        {
            'Type': ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64', 'float128', 'complex64', 'complex128', 'bool', 'object', 'string_', 'unicode_'],
            'Type Code': ['i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f2', 'f4 or f', 'f8 or d', 'f16 or g', 'c8', 'c16', '', 'O', 'S', 'U']
        }
    )

### Ex:2

    
    data1 = {
        'day': ['1/1/2017','1/2/2017','1/3/2017','1/4/2017','1/5/2017','1/6/2017'],
        'temperature': [32,35,28,24,32,31],
        'windspeed': [6,7,2,7,4,2],
        'event': ['Rain', 'Sunny', 'Snow','Snow','Rain', 'Sunny']
    }

### Ex 3: Dictionary
    data2 = {
        'day': ['1/1/2017','1/2/2017','1/3/2017'],
        'temperature': [32,35,28],
        'windspeed': [6,7,2],
        'event': ['Rain', 'Sunny', 'Snow']
    }

### Ex 4: Tuples
    data3 = [
        ('1/1/2017',32,6,'Rain'),
        ('1/2/2017',35,7,'Sunny'),
        ('1/3/2017',28,2,'Snow')
    ]

### Ex 5: List
    data4 = [
        {'day': '1/1/2017', 'temperature': 32, 'windspeed': 6, 'event': 'Rain'},
        {'day': '1/2/2017', 'temperature': 35, 'windspeed': 7, 'event': 'Sunny'},
        {'day': '1/3/2017', 'temperature': 28, 'windspeed': 2, 'event': 'Snow'},
        
    ]

# HomeWork:

## Problem Statement:
### Machine Learning Problem Statement: Student Result Analysis and Prediction

#### Objective:
    Develop a machine learning model to analyze student academic performance and predict results based on historical data. The model should take into account various parameters such as student ID, student name, marks in individual subjects (Subject1, Subject2, Subject3, Subject4), total marks, percentage, and corresponding grades.

#### Dataset:
    A dataset is provided containing the following features for each student:

    Student ID
    Student Name
    Marks obtained in Subject1, Subject2, Subject3, Subject4
    Total marks
    Percentage
    Grade (Target variable)

### Tasks:

#### Exploratory Data Analysis (EDA):

    Perform exploratory data analysis to understand the distribution of marks, percentages, and grades.
    Identify any patterns or correlations between different features.

#### Data Preprocessing:

    Handle missing data and outliers appropriately.
    Convert categorical features (e.g., grades) into numerical representations.
    Normalize/standardize numerical features if necessary.
