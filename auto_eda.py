import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataprep.eda import create_report

df = pd.read_csv('BostonHousing.csv')
print(len(df))
report = create_report(df, title='My Report')
report.save('report_boston_housing.html')

'''
As per the EDA report provided by dataprep, we see that columns "indus" and "tax" are
highly correlated with "nox" and "rad" respectively. So we choose to drop "indus" and 
"tax" columns to reduce redundancy.
'''
df = df.drop(columns=['indus', 'tax'])
print(df.head())

'''
From the plots of columns 'crim' and 'b' we can see that they contain outliers which
could affect model's performance. Therefore we remove these using the Inter Quartile Range
available in the Dataprep Report for these columns.
'''
# Removing outliers for column 'crim'. Q1 and Q3 values obtained from Dataprep Report
Q1 = 0.08204
Q3 = 3.6771
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

df = df.loc[(df['crim']>lower)&(df['crim']<upper)]

# Removing outliers for column 'b'. Q1 and Q3 values obtained from Dataprep Report
Q1 = 375.3775
Q3 = 396.225
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

df = df.loc[(df['b']>lower)&(df['b']<upper)]

df.to_csv('Cleaned_Data.csv')

'''
Standardizing Data
'''
X = df.drop('medv', axis=1)
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open('X_train.pkl','wb') as f:
    pkl.dump(X_train, f)

with open('y_train.pkl','wb') as f:
    pkl.dump(y_train, f)

with open('X_test.pkl','wb') as f:
    pkl.dump(X_test, f)

with open('y_test.pkl','wb') as f:
    pkl.dump(y_test, f)