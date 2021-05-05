#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('Crop_recommendation.csv')


# In[3]:


df.head()


# In[4]:


df['label'].nunique()


# In[5]:


df.corr()


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.countplot(data=df, x='label')
ax.set(xlabel='Crop')
ax.set(ylabel='Count')
plt.xticks(rotation=45)

plt.title('Count of crop', fontsize = 20, color='black')
plt.show()


# In[8]:


fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(df.corr(), annot=True)
ax.set(xlabel='features')
ax.set(ylabel='features')

plt.title('Correlation between different features', fontsize = 20, color='black')
plt.show()


# In[9]:


import warnings
warnings.filterwarnings("ignore")


# In[10]:


columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


fig, ax = plt.subplots(7, 1, figsize=(20, 40), sharex=True)

i = 0
for j in columns:
    sns.swarmplot(data=df, x='label', y=j, ax=ax[i])
    plt.xticks(rotation=45)
    i = i + 1

plt.title('Correlation between different features', fontsize = 20, color='black')
plt.show()


# In[11]:


features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']


# In[12]:


acc = []
model = []


# In[13]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# In[14]:


from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree


# In[15]:


from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(Ytest,predicted_values))


# In[16]:


from sklearn.model_selection import cross_val_score


# In[17]:


score = cross_val_score(DecisionTree, features, target,cv=5)
score


# In[18]:


from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[19]:


score = cross_val_score(NaiveBayes,features,target,cv=5)
score


# In[20]:


from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[21]:


score = cross_val_score(LogReg,features,target,cv=5)
score


# In[22]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[23]:


score = cross_val_score(RF,features,target,cv=5)
score


# In[24]:


#!pip install xgboost


# In[25]:


#!pip install --upgrade xgboost


# In[26]:


import xgboost as xgb
XB = xgb.XGBClassifier()
XB.fit(Xtrain,Ytrain)

predicted_values = XB.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('XGBoost')
print("XGBoost's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[27]:


score = cross_val_score(XB,features,target,cv=5)
score


# In[28]:


acc


# In[29]:


plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')


# In[30]:


import numpy as np


# In[33]:


data = np.array([[100,18, 30, 23.603016, 60.3, 20, 20]])
prediction = RF.predict(data)
print(prediction)


# In[34]:


import pickle
XB_pkl_filename = 'XGBoost.pkl'
XB_Model_pkl = open(XB_pkl_filename, 'wb')
pickle.dump(XB, XB_Model_pkl)
XB_Model_pkl.close()


# In[35]:


RF_pkl_filename = 'RandomForest.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
RF_Model_pkl.close()


# In[ ]:




