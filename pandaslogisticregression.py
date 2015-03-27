# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import csv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

%matplotlib inline                                  

# <codecell>

data = pd.read_csv('train.csv') # read in data

# <codecell>

[d for d in data.Embarked.unique() ]  #just experimenting with unique()
[d for d in data.Sex.unique() ]       # these function calls change nothing

# <codecell>

embark_dict = {'S': 1, 'C': 2, 'Q': 3}
sex_dict = {'male': 0, 'female': 1}
print embark_dict

# <codecell>

data['Embarked'] = data.Embarked.map(embark_dict)

# <codecell>

data['Sex'] = data.Sex.map(sex_dict)

# <codecell>

data['Name'] = data.Name.map(len)  # replace the name string with the length of the name

# <codecell>

nvs = pd.crosstab(data['Name'],data['Survived'])

# <codecell>

nvs.hist()
pl.show()

# <codecell>

data['Pclass'].hist(bins=3)


# <codecell>


data[data['Survived'] != 1]['Pclass'].hist(bins=3)
data[data['Survived'] == 1]['Pclass'].hist(bins=3)

# <codecell>


data = data.drop('Ticket',1)
data = data.drop('Cabin',1)
data = data.drop('Age',1)
data = data.drop('Embarked',1)

# <headingcell level=2>

# Logistic Regression !!

# <codecell>

import statsmodels.api as sm

# <codecell>

data['intercept'] = 1.0

train_cols = data.columns[2:]
 
logit = sm.Logit(data['Survived'], data[train_cols])
 
# fit the model
result = logit.fit()

# <codecell>

print result.summary()

# <codecell>

print result.conf_int()

# <codecell>

print np.exp(result.params)  # odds ratios only

# <headingcell level=2>

# Predictions!!!

# <codecell>

testdata = pd.read_csv('test.csv') # read in data

# <codecell>

testdata['Embarked'] = testdata.Embarked.map(embark_dict)
testdata['Sex'] = testdata.Sex.map(sex_dict)
testdata['Name'] = testdata.Name.map(len)  # replace the name string with the length of the name

testdata = testdata.drop('Ticket',1)
testdata = testdata.drop('Cabin',1)

# <codecell>

testdata['Age'] = testdata['Age'].fillna(method='pad')
testdata['Fare'] = testdata['Fare'].fillna(method='pad')

testdata['intercept'] = 1.0

# <codecell>

testdata.describe()

# <codecell>

testdata['Survived'] = result.predict(testdata[train_cols])

# <codecell>

train_cols

# <codecell>

testdata['Survived'] = [d for d in testdata.Survived < 0.5 ]  #just experimenting with unique()
bool_map = {True: 1, False: 0}
testdata['Survived'] = testdata.Survived.map(bool_map)

# <codecell>

testdata = testdata.drop('Name',1)
testdata = testdata.drop('SibSp',1)
testdata = testdata.drop('Parch',1)
testdata = testdata.drop('Sex',1)
testdata = testdata.drop('Fare',1)
testdata = testdata.drop('intercept',1)
testdata = testdata.drop('Pclass',1)
testdata = testdata.drop('Age',1)
testdata = testdata.drop('Embarked',1)

# <codecell>

testdata.to_csv("linearmodel.csv", cols=['PassengerId', 'Survived'], index=False)

# <codecell>


