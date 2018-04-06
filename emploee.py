#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:18:37 2018

@author: cherry
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
employees=pd.read_csv('turnover.csv')
employees.head()



#Renaming the column sales to department


employees=employees.rename(columns={'sales':'department'})


#lets make support,it as technical department

employees['department']=np.where(employees['department']=='support','technical',employees['department'])
employees['department']=np.where(employees['department']=='IT','technical',employees['department'])






x = employees.iloc[:,[0,1,2,3,4,5,7,8,9]].values
y = employees.iloc[:,6].values





#employees['left'].value_counts()
#employees.groupby('left').mean()


#some observations

#no of monthly hours of employees left is more than emploees in
#satisfcation of employees in is more than emploees left
#time spend of employees left is more than employees in
#no of projects of left is more than emploees in 
#no of promotions of employees left has less than employee in 


#so from the mean values we may infer that due to over load at work and not giving proper response in form of appraisals or 
#promotions from company might be the reason for employees leaving


#employees['department'].value_counts()
#employees.groupby('department').mean()





#Data Visualization 


pd.crosstab(employees.department,employees.left).plot(kind='bar')
plt.title('Turnover vs Department')
plt.xlabel('Department')
plt.ylabel('Left')
plt.savefig('department_Left bar_chart')

#there is more left  in department technical 

#salary vs left
pd.crosstab(employees.salary,employees.left).plot(kind='bar')
plt.title('Turnover vs salary')
plt.xlabel('salary')
plt.ylabel('Left')
plt.savefig('salary_Left bar_chart')



#Data Preprocessing 

#label encoder



from sklearn.preprocessing import LabelEncoder as le,OneHotEncoder


labelencoder_x=le()


x[:, 7]=labelencoder_x.fit_transform(x[:,7])



x[:, 8]=labelencoder_x.fit_transform(x[:,8])



onehotencoder=OneHotEncoder(categorical_features=[7])



x= onehotencoder.fit_transform(x).toarray()


#feature selection

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 10)
rfe = rfe.fit(x,y)
print(rfe.support_)
print(rfe.ranking_)


x=x[:,[0,2,3,7,8,9,10,12,13,14]] 







from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


#Logistic Regression


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(x_train, y_train)


#predicting test results

y_pred=logreg.predict(x_test)

#making confusion matrix to check whether model is learnt correctly

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)



#Accuracy


from sklearn.metrics import accuracy_score
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(x_test))))


#Logistic regression accuracy: 0.759





#Random Forest


from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier(n_estimators=30,random_state=0)
rfclassifier.fit(x_train, y_train)

#predicting test results

y_pred=rfclassifier.predict(x_test)

#making confusion matrix to check whether model is learnt correctly

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)



#Accuracy


from sklearn.metrics import accuracy_score
print('Random Forests accuracy: {:.3f}'.format(accuracy_score(y_test, rfclassifier.predict(x_test))))

#Random Forests accuracy: 0.989



#support vector machine 


from sklearn.svm import SVC
svmclassifier = SVC()
svmclassifier.fit(x_train, y_train)


#predicting test results

y_pred=svmclassifier.predict(x_test)

#making confusion matrix to check whether model is learnt correctly

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)



#Accuracy


from sklearn.metrics import accuracy_score
print('svm classifier accuracy: {:.3f}'.format(accuracy_score(y_test, svmclassifier.predict(x_test))))

#svm classifier accuracy: 0.952




# the Random forest has shown the highest accuracy


#cross validation to avoid over fitting 

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = RandomForestClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, x_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))



#10-fold cross validation average accuracy: 0.984

#As cross validation result is very near to random forest random forest is not over fitted and giving highest accuracy






