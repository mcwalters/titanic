# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Using pandas and sklearn regression analysis

# <rawcell>


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

# <headingcell level=2>

# Training Data Input and Munging

# <codecell>

data = pd.read_csv('train.csv') # read in data

# <codecell>

# Translate Gender and Port of Embarkation into integers for statistical purposes
embark_dict = {'S': 1, 'C': 2, 'Q': 3}          # Order of Embarkation: Southhamton, Cherbourg, Queenstown
sex_dict = {'male': 0, 'female': 1}
#print embark_dict

# <codecell>

#Apply maps and evaluate names by their length
data['Embarked'] = data.Embarked.map(embark_dict)
data['Sex'] = data.Sex.map(sex_dict)
data['Name'] = data.Name.map(len)  # replace the name string with the length of the name

# <headingcell level=2>

# Logistic Regression !!

# <codecell>

import statsmodels.api as sm

# Get rid of non numerical data to prepare for Regression Analysis
data = data.drop('Ticket',1)
data = data.drop('Cabin',1)

# <codecell>

# Pad out missing Data with copies of neighboring data
data['Age'] = data['Age'].fillna(method='pad')
data['Fare'] = data['Fare'].fillna(method='pad')
data['Embarked'] = data['Embarked'].fillna(method='pad')
data['intercept'] = 1.0   # Required for regression algorithim.

# <codecell>

data.describe()

# <codecell>

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

# Predictions from Model built above

# <codecell>

testdata = pd.read_csv('test.csv') # read in data

# <codecell>

testdata['Embarked'] = testdata.Embarked.map(embark_dict)
testdata['Sex'] = testdata.Sex.map(sex_dict)
testdata['Name'] = testdata.Name.map(len)  # replace the name string with the length of the name

testdata = testdata.drop('Ticket',1)
testdata = testdata.drop('Cabin',1)

# <codecell>

# Pad out missing Data with copies of neighboring data
testdata['Age'] = testdata['Age'].fillna(method='pad')
testdata['Fare'] = testdata['Fare'].fillna(method='pad')
testdata['intercept'] = 1.0   # Required for regression algorithim.

# <codecell>

# Get predictions from linear regression module built above.
testdata['Survived'] = result.predict(testdata[train_cols])

# <codecell>

#Prep data for output
testdata['Survived'] = [d for d in testdata.Survived < 0.5 ]  #Convert probabilities to boolean
bool_map = {True: 1, False: 0}
testdata['Survived'] = testdata.Survived.map(bool_map)        #Convert boolean to 0, 1 for CSV output

# <codecell>

#Take slice of data which only includes ID and Survival and print to CSV for turning in.
printdata = testdata.loc[:,['PassengerId','Survived']]
printdata.to_csv("pandaslinear.csv",index=False)

# <codecell>


# <codecell>


