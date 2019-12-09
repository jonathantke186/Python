# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 05:16:51 2019
This iA=s a small example of ML decision trees and random forest 

@author: jonat
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#i will be usnig a csv file that will be upload with this script

df= pd.read_csv('kyphosis.csv')

#sns.pairplot(df,hue='Kyphosis')

#the next import will start the ML
from sklearn.model_selection import train_test_split

X=df.drop('Kyphosis', axis=1)
y=df['Kyphosis']

X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print (confusion_matrix(y_test,predictions))
print( '\n')
print(classification_report(y_test, predictions))


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

print (confusion_matrix(y_test,rfc_pred))
print( '\n')
print(classification_report(y_test, rfc_pred))