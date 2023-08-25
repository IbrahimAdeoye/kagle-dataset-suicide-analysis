#!/usr/bin/env python
# coding: utf-8

# In[213]:


import pandas as pd
import matplotlib.pyplot as plt
import re


# In[214]:


df = pd.read_csv('/Users/adeoyedipo/downloads/master.csv')
df.info()


# In[215]:


df.head()


# In[216]:


df.rename(columns={'gdp_per_capita ($)':'gdp_per_capita',' gdp_for_year ($) ':'gdp_for_year'},inplace=True)


# In[217]:


# to convert the gdp_for_year column from string to int
df['gdp_for_year'] = df['gdp_for_year'].apply(lambda x:int(re.sub(r'[^\w]','',x)))


# In[ ]:


# Creating the target variable/feature/attribute
y = df['suicides_no']


# In[219]:


# spliting the data in training set and test set and the test set is 20% of the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.2,random_state=42)


# In[220]:


X_train.columns


# In[221]:


# drop some columns due to lack or relevance or lack of information. droping suicides/100k pop cause it's 
# basically cheating,dropping country-year because i already have coumns for country and year, dropping HDI for year
# because too many missing values, HDI was introduced in the 1990s amd our data starts from 1987

for set_ in (X_train,X_test):
    set_.drop(columns=['country-year','HDI for year','suicides/100k pop','suicides_no'],inplace=True)


# In[222]:


# just plotting total number of suicide per age group
# data insight i guess
# it shows age 35-54 commits the most suicide year and since the is more of how money affects suicide data set
# it makes sense that it would affects the age 35-54 most

d = [df[df['age']==i]['suicides_no'].sum() for i in df.age.unique()]
print(d)
print(df.age.unique())


plt.plot([1,2,3,4,5,6],d)
plt.scatter([1,2,3,4,5,6],d,marker='X',c='black',s=100)
plt.xticks(range(1,7),['15-24 years', '35-54 years', '75+ years', '25-34 years',
       '55-74 years', '5-14 years'])
plt.grid()
plt.show()


# In[267]:


# i have checked if there are 
# now we have to performance data preprocessing, there are many categorical features in this dataset, and many 
# different scaled data features

# country,sex,age and generation are categorical
# year,population,gdp_for_year and gdp_per_capita

from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,PolynomialFeatures
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer,make_column_selector,make_column_transformer
import numpy as np
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import mean_squared_error


# In[226]:


preprocessing = ColumnTransformer([('category',OneHotEncoder(handle_unknown='ignore'),
                                    ['country','sex','age','generation']),
                                  ('scaler',StandardScaler(),['year','population','gdp_for_year','gdp_per_capita'])],
                                 remainder='passthrough')


# In[397]:


# we import and train three Regression algorithms  

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

linear_pipeline = make_pipeline(preprocessing,LinearRegression())
tree_pipeline = make_pipeline(preprocessing,DecisionTreeRegressor(random_state=42))
forest_pipeline = make_pipeline(preprocessing,RandomForestRegressor(random_state=42))

linear_pipeline.fit(X_train,y_train)
tree_pipeline.fit(X_train,y_train)
forest_pipeline.fit(X_train,y_train)


# In[ ]:





# In[395]:


# print prediction of the the trained algorithms on the first 10 instances in the training set
print(linear_pipeline.predict(X_train[:10]),"\n")

print(np.round(forest_pipeline.predict(X_train[:10])),"\n")

print(tree_pipeline.predict(X_train[:10]),"\n")


# In[313]:


# we evaluate each algorithm using cross valuation 

linear_cv =-cross_val_score(linear_pipeline,X_train,y_train,cv=10,scoring="neg_root_mean_squared_error")
tree_cv =-cross_val_score(tree_pipeline,X_train,y_train,cv=10,scoring="neg_root_mean_squared_error")
forest_cv = -cross_val_score(forest_pipeline,X_train,y_train,cv=10,scoring="neg_root_mean_squared_error")


print(linear_cv)
print(forest_cv)
print(tree_cv)


# In[344]:


# from my evaluation with k-fold cross-validation the random forest performanced the best
#let's fine tune the model to even make it performace better than before

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
final_pipeline = Pipeline([('preprocessin',preprocessing),
                           ('forest',RandomForestRegressor(n_jobs=-1,random_state=42))])
param_grid = {'forest__n_estimators':[200,300],
             'forest__max_depth':[35,40,45],
             'forest__min_samples_split':[3,4,5]
             }

grid_search = GridSearchCV(final_pipeline,param_grid=param_grid,scoring="neg_root_mean_squared_error",cv=10)
grid_search.fit(X_train,y_train)


# In[396]:


# best hyperparamters pairing
grid_search.best_params_


# In[361]:


# random forest has a method to show that shows the importances of each feature
sorted(zip(grid_search.best_estimator_['forest'].feature_importances_,
grid_search.best_estimator_["preprocessin"].get_feature_names_out()),
reverse=True)


# In[383]:


# Time to evaluate on the test set
final_model = grid_search.best_estimator_
testset_prediction = final_model.predict(X_test)

test_set_rmse = mean_squared_error(y_test,testset_prediction,squared=False)
test_set_rmse


# In[384]:


from scipy import stats


# In[381]:


# we calculate the 95% confidence interval, it will give you the 95 % range where the
# generalization error(difference between the y_test and the prediction) fall under
squared_error = (y_test-testset_prediction)**2
confidence = 0.95
np.sqrt(stats.t.interval(confidence,len(squared_error) - 1,
                 loc=squared_error.mean(),scale=stats.sem(squared_error)))


# In[ ]:




