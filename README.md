# Machine Learning Techniques to Predict Wheat Type from Kernel
The 'seeds' dataset provided is a csv file that contains information on different wheat kernels and the type of wheat they came from. The three types were: Kama, Rosa and Canadian. The independent variables used were area, perimeter, compactness, length of kernel, width of kernel, asymmetry coefficient and length of kernel groove. The dataset came from: https://archive.ics.uci.edu/dataset/236/seeds.

The aim of this project is to use three different machine learning methods; support vector machine, random forest and Knn to predict the wheat a kernel belongs to. The data will be prepared, preprocessed, trained and tested. Finally, a conclusion will be given.

All packages used:

```
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
from sklearn import set_config
import pickle
import matplotlib.pyplot as plt
```

Import data and check for na values.

```
df = pd.read_csv("seeds.csv")
df.head()

df.isna().sum()
```
Check for outliers:

```
df.loc[:,'area':'length of kernel groove'].boxplot(figsize=(20,5))
plt.show()
```

![boxplot display of outliers](outliers_boxplot.png)
```
df.loc[:,'area':'length of kernel groove'].hist(bins=10, figsize=(25, 20))
plt.show()
```
![histogram display of outliers](outliers_hist.png)

Area, perimeter, length of kernel and length of kernel groove appear to be
left skewed and could benefit from a log transform.

The next step is to prepare the data for modelling, this will be done by
converting the wheat to binary and splitting the data into two seperate components so it can be modelled. The binary split will be Kama vs Rosa and Canadian.

```
mapper = {1: 1, 2: 0, 3: 0}
df['class'] = df['type'].replace(mapper)
df['class'].value_counts()
df = df.drop('type', axis = 1)

y = df['class']
X = df.drop('class', axis=1)
```

Another important component to this preparation is the log transformation
mentioned earlier. Robust scaler will also be used to prepare the data for
modelling, this subtracts the median and divides by the interquartile range.
The importance of this is to make sure all data is on the same scale.

These preprocessing techniques are used as they may improve the performance
of machine learning models.

The preprocessing pipelines will be prepared next and the data will be divided
into train/test sets.

```
columns_left_skew = ['area', 'perimeter', 'length of kernel', 'length of kernel groove']

columns_other = [item for item in list(X.columns) 
                             if item not in columns_left_skew]

columns_left_skew_pipeline = Pipeline(
    steps = [
        ("log_transform", FunctionTransformer(np.log)), 
        ("scaler", RobustScaler())
    ]
)

columns_other_pipeline = Pipeline(
    steps = [
        ("scaler", RobustScaler())
    ]
)

preprocess_pipeline = ColumnTransformer(
    transformers = [
        ("left_skew", columns_left_skew_pipeline, columns_left_skew),
        ("other", columns_other_pipeline, columns_other)
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
```

Modelling will begin, three models will be compared; SVM, random forest and Knn.
For each model, the pipeline will be built, then a parameter grid will be created so the best
set of parameters can be searched. Finally the best parameters will be saved for use on the test data.
Note to avoid confusion - there are two types of test data. The first is implemented when cross validation
is used on the training set. The second is a randomly selected set of 20% that will be used at the end
to compare models.

Support Vector Machine

```
pipe = Pipeline(steps=[('preprocess', preprocess_pipeline), ('svm', svm.SVC(probability=True))])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],  
    'svm__gamma': [1, 0.1, 0.01, 0.001], 
    'svm__kernel': ['rbf', 'linear', 'poly']}

search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=5, refit=True)
search.fit(X_train, y_train)

print("Best CV score = %0.3f:" % search.best_score_)
print("Best parameters: ", search.best_params_)

# store the best params and best model for later use
SVM_best_params = search.best_params_
SVM_best_model = search.best_estimator_
```
# store the best params and best model for later use
