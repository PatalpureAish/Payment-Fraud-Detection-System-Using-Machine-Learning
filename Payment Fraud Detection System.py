#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#Import important libraries
import pandas as pd
import numpy as np

#Read data set
data = pd.read_csv("Fraud Detection Data set.csv")


# In[2]:


print(data.head())


# In[3]:


#Null values
print(data.isnull().sum())


# In[4]:


print(data.shape)


# In[5]:


#exploring transaction type
print(data.type.value_counts())


# In[ ]:





# In[6]:


type = data["type"].value_counts()
transactions = type.index
quantity = type.values
import plotly.express as px
figure = px.pie(data, values=quantity, names=transactions, hole=0.5, title="Distribution of transaction")
figure.show()


# In[7]:


# check correlation b/w the feature of the data with the isFraud coulmn
#checking correlation
correlation= data.corr()
print (correlation["isFraud"].sort_values(ascending=False))


# In[8]:


data["type"] = data["type"].map({"CASH_OUT": 1,"PAYMENT": 2,"CASH_IN": 3,"TRANSFER": 4,"DEBIT":5})
data["isFraud"] = data["isFraud"].map({0:"No Fraud", 1:"Fraud"})
print(data.head())


# In[ ]:





# In[15]:


#spitting the data
from sklearn.model_selection import train_test_split
x=np.array(data[["type","amount","oldbalanceOrg","newbalanceOrig"]])
y=np.array(data[["isFraud"]])


# In[16]:


#tranining a machine leaning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[17]:


#prediction
# feature=[type,amount,oldbalanceOrg,newbalanceOrig]
features= np.array([[4,9000.60,9000.60,0.0]])
print(model.predict(features))


# In[ ]:




