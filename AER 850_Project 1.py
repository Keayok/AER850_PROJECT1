# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 12:31:29 2025

@author: quich
"""
# from IPython import get_ipython
# import matplotlib.pyplot as plt
# get_ipython().magic('clear')     # clears console
# get_ipython().magic('reset -f')  # clears all variables
# plt.close('all')                 # closes all figures



import pandas as pd
import pathlib as path
import numpy as np
import math
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
import joblib
#mport scikit as sk

## Predict maintainance Step based on Coordinates provided
## Read the CSV/xlsx file
df = pd.read_csv("Project 1 Data.csv")



print(df.head())
print(df.columns)
print(df.X)
print(df.Y)
print(df.Z)
print(df.Step)
# df.hist()


## Correlation Matrix 
# Target Variable is the step and training data 

# import and use Stratified split. This reduces the risk of leakage which would skew results
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

my_splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 42)

for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_data_train = df.iloc[train_index].reset_index(drop=True)
    strat_data_test = df.iloc[test_index].reset_index(drop=True)

# strat_data_train = strat_data_train.drop(columns=["Step"], axis = 1)
# strat_data_test = strat_data_test.drop(columns=["Step"], axis = 1)

print(df.shape)
print(strat_data_train.shape)
print(strat_data_test.shape)


## Train data 
# Identify y and X. Remember the goal is to find f(.) such that y=f(X)
y_train = strat_data_train['Step']
X_train = strat_data_train.drop(columns=['Step'])
y_test = strat_data_test['Step']
X_test = strat_data_test.drop(columns=['Step'])

## Standard correlation also Pearson correlation 
corr_matrix = strat_data_train.corr()

import matplotlib.pyplot as plt
import seaborn as sns

# plt.figure()
# sns.heatmap(np.abs(corr_matrix))


# Drom correlation matrix, we identify colinear variables, and select one from them
# Usually, we keep the variable with the highest correlation with y, but this
# does not generate the best results all the time. So, trial and error is needed.
print(np.abs(y_train.corr(X_train['X'])))
print(np.abs(y_train.corr(X_train['Y'])))
print(np.abs(y_train.corr(X_train['Z'])))
print("\n")

 # from the correlations between y train(house value) and the other values

# # Based on correlation values, we drop the following from X_train
# X_train = X_train.drop(columns=['Y'])
# X_train = X_train.drop(columns=['Z'])

# # We can also drop the above columns from X_test. This is safe to do so, because
# # we decided the columns to drop based on train data only.
# X_test = X_test.drop(columns=['Y'])
# X_test = X_test.drop(columns=['Z'])

# Training and Evaluating Data set based on models

from sklearn.linear_model import LogisticRegression as LogisticReg
from sklearn.tree import DecisionTreeClassifier as DecClass
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV as RCV
from sklearn.pipeline import Pipeline as pipl
from sklearn.preprocessing import StandardScaler
# # Evaluate each classifier using various metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


## Pipelines
# Logistic Regression
Pipl_LogReg = pipl([('scale', StandardScaler()),('model',LogisticReg(max_iter=1000, random_state=42))])

# Decision Trees
Pipl_DecTree = pipl([('model',DecClass( random_state=42))])

# Support Vector Machine
Pipl_SVM = pipl([('model',SVC(random_state=42))])

# Random Forest with RandomSearch CV
RNDM_Forest = RandomForestClassifier(random_state=42,n_jobs=-1)
parameters = {'n_estimators': randint(50, 200),'max_depth': randint(2, 10),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)}
Pipl_RNDM_SRCH_CV = pipl([('model',RCV(RNDM_Forest,parameters,n_iter=5,cv=3,scoring='accuracy',random_state=42,n_jobs=-1))])


# Stacking Classifier 
from sklearn.ensemble import StackingClassifier

# Two of the best performing Models Random forest and Logistic Regression
estim1   = [('Logisitic Regression', Pipl_LogReg),('Random Forest',Pipl_RNDM_SRCH_CV)]
estim2   = [('SVM', Pipl_SVM),('Random Forest',Pipl_RNDM_SRCH_CV)]
estim3   = [('Logisitic Regression', Pipl_LogReg),('SVM',Pipl_SVM)]
estim4   = [('Decision Tree', Pipl_DecTree),('SVM',Pipl_SVM)]
estim5   = [('Decision Tree', Pipl_LogReg),('Logisitic Regression',Pipl_LogReg)]
estim6   = [('Decision Tree', Pipl_LogReg),('Random Forest',Pipl_RNDM_SRCH_CV)]


stk_class1 = StackingClassifier(estimators=estim1,cv=3,passthrough=False,n_jobs=-1) 
stk_class2 = StackingClassifier(estimators=estim2,cv=3,passthrough=False,n_jobs=-1) 
stk_class3 = StackingClassifier(estimators=estim3,cv=3,passthrough=False,n_jobs=-1) 
stk_class4 = StackingClassifier(estimators=estim4,cv=3,passthrough=False,n_jobs=-1) 
stk_class5 = StackingClassifier(estimators=estim5,cv=3,passthrough=False,n_jobs=-1) 
stk_class6 = StackingClassifier(estimators=estim6,cv=3,passthrough=False,n_jobs=-1) 


# preallocation
y_pred_ = {}
cm_ = {}
scores_ = {}
precision_={}
recall_ = {}
f1_ = {}

# handles all the models in 1 loop 

for name, model in [('Logisitic Regression',Pipl_LogReg),('Decision Tree',Pipl_DecTree) ,('SVM',Pipl_SVM) ,('Random Forest',Pipl_RNDM_SRCH_CV), ('Stacked Classifier 1', stk_class1 ),('Stacked Classifier 2', stk_class2 )
                    ,('Stacked Classifier 3', stk_class3 ),('Stacked Classifier 4', stk_class4 ),('Stacked Classifier 5', stk_class5 ),('Stacked Classifier 6', stk_class6 )]:
    
    model.fit(X_train, y_train)
    print(f"{name} Training Accuracy:", model.score(X_train, y_train))
    print(f"{name} Test Accuracy:", model.score(X_test, y_test))
    print("\n")
    
    y_pred_[name] = model.predict(X_test)
    ## Confusion Matric for each
    cm_[name] = confusion_matrix(y_test, y_pred_[name])
    print("Confusion Matrix:")
    # print(cm_LogRec)
    plt.figure()
    sns.heatmap(np.abs(cm_[name]))
    print("\n")
    # Performance Metrics 
    precision_[name] = precision_score(y_test, y_pred_[name],average='weighted',zero_division=0)
    recall_[name] = recall_score(y_test, y_pred_[name], average='weighted',zero_division=0)
    f1_[name] = f1_score(y_test, y_pred_[name], average='weighted', zero_division=0)
    print(f"{name}:")
    print("Precision:", precision_[name])
    print("Recall:", recall_[name])
    print("F1 Score:", f1_[name])

    print("\n")
    

# Save as ajoblib file

# joblib.dump(stk_class1,'Inverter_Maintainance_Steps_Model.joblib')
# print('File Saved') 
    
# LogReg = LogisticReg(max_iter=1000, random_state=42)
# LogReg.fit(X_train, y_train)
# # print("Predictions:", LogReg.predict(X_test))
# print("Logistic Regression:")
# print("Training accuracy:", LogReg.score(X_train, y_train))
# print("Test accuracy:", LogReg.score(X_test, y_test))
# print("\n") 

# DecTree = DecClass( random_state=42)
# DecTree.fit(X_train, y_train)
# # print("Predictions:", DecTree.predict(X_test))
# print("Decision Tree Model:")
# print("Training accuracy:", DecTree.score(X_train, y_train))
# print("Test accuracy:", DecTree.score(X_test, y_test))
# print("\n") 


# SVM = SVC(random_state=42)
# SVM.fit(X_train, y_train)
# # print("Predictions:", SVM.predict(X_test))
# print("SVM Model:")
# print("Training accuracy:", SVM.score(X_train, y_train))
# print("Test accuracy:", SVM.score(X_test, y_test))
# print("\n") 

# # Implementing RandomSearchCV

# RNDM_Forest = RandomForestClassifier(random_state=42)
# parameters = {'n_estimators': randint(50, 200),'max_depth': randint(2, 10),
#     'min_samples_split': randint(2, 10),
#     'min_samples_leaf': randint(1, 5)}

# RNDM_SRCH_CV = RCV(
#     RNDM_Forest,parameters,n_iter=20,
#     cv=5,
#     scoring='accuracy',
#     random_state=42)


# RNDM_SRCH_CV.fit(X_train, y_train)
# # print("Predictions:", RNDM_SRCH_CV.predict(X_test))
# print("Random Forest Model:")
# print("Training accuracy:", RNDM_SRCH_CV.score(X_train, y_train))
# print("Test accuracy:", RNDM_SRCH_CV.score(X_test, y_test))
# print("\n")

# # # Evaluate each classifier using various metrics
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# # Predicitions for the 4 models 
# y_pred_LogReg = LogReg.predict(X_test)
# y_pred_DecTree = DecTree.predict(X_test)
# y_pred_SVM = SVM.predict(X_test)
# y_pred_RNDM_Forest = RNDM_SRCH_CV.predict(X_test)

# ## Confusion Matric for each

# # Logistic Regression
# cm_LogRec = confusion_matrix(y_test, y_pred_LogReg)
# print("Confusion Matrix:")
# # print(cm_LogRec)
# plt.figure()
# sns.heatmap(np.abs(cm_LogRec))
# print("\n")
# # Decision Tree
# cm_DecTree = confusion_matrix(y_test, y_pred_DecTree)
# print("Confusion Matrix:")
# # print(cm_DecTree)
# plt.figure()
# sns.heatmap(np.abs(cm_DecTree))
# print("\n")

# # Support vector Machine

# cm_SVM = confusion_matrix(y_test, y_pred_SVM)
# print("Confusion Matrix:")
# # print(cm_SVM)
# plt.figure()
# sns.heatmap(np.abs(cm_SVM))
# print("\n")

# # Random Forest 
# cm_RNDM_Forest = confusion_matrix(y_test, y_pred_RNDM_Forest)
# print("Confusion Matrix:")
# # print(cm_RNDM_Forest)
# plt.figure()
# sns.heatmap(np.abs(cm_RNDM_Forest))
# print("\n")


# Other Metric such as F1 and recall

# # Logistic Regression
# precision_LogReg = precision_score(y_test, y_pred_LogReg,average='weighted',zero_division=0)
# recall_LogReg = recall_score(y_test, y_pred_LogReg, average='weighted',zero_division=0)
# f1_LogReg = f1_score(y_test, y_pred_LogReg, average='weighted', zero_division=0)
# print("Logistic Regression:")
# print("Precision:", precision_LogReg)
# print("Recall:", recall_LogReg)
# print("F1 Score:", f1_LogReg)

# print("\n")

# # Decision Tree
# precision_DecTree = precision_score(y_test, y_pred_DecTree,average= "weighted",zero_division=0)
# recall_DecTree = recall_score(y_test, y_pred_DecTree,average= "weighted",zero_division=0)
# f1_DecTree = f1_score(y_test, y_pred_DecTree,average= "weighted",zero_division=0)
# print("Decision Tree Model:")
# print("Precision:", precision_DecTree)
# print("Recall:", recall_DecTree)
# print("F1 Score:", f1_DecTree)
# print("\n")

# # Support Vector Machine
# precision_SVM = precision_score(y_test, y_pred_SVM,average= "weighted",zero_division=0)
# recall_SVM = recall_score(y_test, y_pred_SVM,average= "weighted",zero_division=0)
# f1_SVM = f1_score(y_test, y_pred_SVM,average= "weighted",zero_division=0)
# print("SVM Model:")
# print("Precision:", precision_SVM)
# print("Recall:", recall_SVM)
# print("F1 Score:", f1_SVM)
# print("\n")

# # Random Forest 
# precision_RNDM_Forest = precision_score(y_test, y_pred_RNDM_Forest, average= "weighted",zero_division=0)
# recall_RNDM_Forest = recall_score(y_test, y_pred_RNDM_Forest,average= "weighted",zero_division=0)
# f1_RNDM_Forest = f1_score(y_test, y_pred_RNDM_Forest,average= "weighted",zero_division=0)
# print("Random Forest Model:")
# print("Precision:", precision_RNDM_Forest)
# print("Recall:", recall_RNDM_Forest)
# print("F1 Score:", f1_RNDM_Forest)
# print("\n")



# # # # Data scaling
# # # # NOTE: To use K-fold cross validation, use cross val score 
# from sklearn.model_selection import cross_val_score
# sc = cross_val_score()
# sc.fit(X_train)

# pd.DataFrame(X_train).to_csv("UnscaledoriginalData.csv")
# X_train = sc.transform(X_train)
# pd.DataFrame(X_train).to_csv("NowScaledData_Proj1.csv")


# # At this point, we can also scale X_test. This is safe to do so, because the 
# # the standard scaler above has been fit to the train data only.
# X_test = sc.transform(X_test)

# # Build the first classifier using logistic regression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression  # Simplest classifier to use
# clf1 = Pipeline([
#     ("scaler", StandardScaler()),
#     ("clf", LogisticRegression(max_iter=1000, random_state=42)) # same state 
# ])
# clf1.fit(X_train, y_train)
# print("Training accuracy:", clf1.score(X_train, y_train))
# print("Test accuracy:", clf1.score(X_test, y_test))

# # Evaluate the classifier using various metrics
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
# y_pred_clf1 = clf1.predict(X_test)
# cm_clf1 = confusion_matrix(y_test, y_pred_clf1)
# print("Confusion Matrix:")
# print(cm_clf1)
# precision_clf1 = precision_score(y_test, y_pred_clf1)
# recall_clf1 = recall_score(y_test, y_pred_clf1)
# f1_clf1 = f1_score(y_test, y_pred_clf1)
# print("Precision:", precision_clf1)
# print("Recall:", recall_clf1)
# print("F1 Score:", f1_clf1)



## Grave 

# y1_train = strat_data_train
# X1_train = strat_data_train
# y1_test = strat_data_test
# X2_test = strat_data_test


# Df2 = [df.X;df.Y;df.Z]

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     Df2, df.Step, test_size=0.2, random_state=42, stratify=df.Step)


# # Mask correlation values above a threshold
# masked_corr_matrix = np.abs(corr_matrix) < 0.8
# #less independent variables by getting rid of collinear variables - Repeated info?highly correlated values - Get more unique data
 
# sns.heatmap(masked_corr_matrix)


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y)

