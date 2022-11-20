#!/usr/bin/env python
# coding: utf-8

# # TSF - Data Science & Business Analytics Internship
# 
# 
# 
# ### TASK-1 Predicting the score of a student using Supervised ML technique
# 
# 
# 
# #### Author - Mohammad Haris

# ### Step 1- Importing all the required libraries

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#To ignore warnings 
import warnings as wg
wg.filterwarnings("ignore")


# In[17]:


#Reading data from the link provided by TSF

url = "http://bit.ly/w-data"
df = pd.read_csv(url)


# In[10]:


#Observing the dataset
df.head()


# In[11]:


df.tail()


# In[14]:


df.info()


# In[19]:


df.shape


# In[16]:


df.describe()


# In[17]:


#To check if our dataset contains any null or missing values
df.isnull().sum()


# ## Step 2- Visualising the dataset

# In[32]:


#Plotting the dataset
df.plot(x='Hours', y='Scores', style='o', color='blue', markersize=10)
plt.title('Hours vs Score')
plt.xlabel('Hours Studied')
plt.ylabel('% Score')
plt.grid()
plt.show()


# In[33]:


# The above graph shows there is a linear relationship between the two quantities, therefore we can use '.corr()' 
# to determine the correlation between the variables.

df.corr()


# ## Step 3- Preparing the data

# In[34]:


df.head()


# In[35]:


# splitting the data using iloc function
X = df.iloc[:, :1].values
Y = df.iloc[:, 1:].values


# In[36]:


X


# In[37]:


Y


# ## Step 4- Training the Algorithm

# In[43]:


# Splitting data into traning and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[46]:


# Training the algorithm
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, Y_train)


# ## Step 5- Visualizing our trained model

# In[49]:


line = model.coef_*X + model.intercept_

#Plotting for the trained data
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X, line, color = 'green');
plt.xlabel('Hours Studied')
plt.ylabel('% Score')
plt.grid()
plt.show()


# ## Step 6- Predicting the score on the basis of no. of hrs studied

# In[50]:


# Predicting the score based on the number of hrs studied
print(X_test)
Y_pred = model.predict(X_test)


# In[51]:


# Comparing actual data VS predicted data
Y_test


# In[52]:


Y_pred


# In[53]:


# Comparing Actual VS Predicted data

comp = pd.DataFrame({'Actual':[Y_test], 'Predicted':[Y_pred]})
comp


# In[57]:


# testing with your own data

hours = 9.25
own_pred = model.predict([[hours]])
print("The predicted score if a student studies for", hours," hours is", own_pred[0])


# ## Step 7- Evaluating the model

# In[58]:


from sklearn import metrics
print("Mean Absolute Error:", metrics.mean_absolute_error(Y_test, Y_pred))

