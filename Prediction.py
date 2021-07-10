#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import statsmodels.api as sm


# In[4]:


car = pd.read_csv('https://stepik.org/media/attachments/lesson/387691/cars.csv')


# In[5]:


# Подготавливаем данные
car['company'] = car.CarName.apply(lambda x: x.split(' ')[0])

car.drop(['car_ID','CarName'], axis='columns', inplace=True)

car.company = car.company.apply(lambda x: x.replace('vw', 'volkswagen'))

car.company = car.company.apply(lambda x: x.replace('vokswagen', 'volkswagen'))

car.company = car.company.apply(lambda x: x.replace('maxda', 'mazda'))

car.company = car.company.apply(lambda x: x.replace('toyouta', 'toyota'))

car.company = car.company.apply(lambda x: x.lower())

car.company = car.company.apply(lambda x: x.replace('porcshce', 'porsche'))


# In[9]:


# Считаем корреляцию между данными
car.corr().round(2)


# In[7]:


cars = car[['company', 
            'fueltype', 
            'aspiration',
            'carbody', 
            'drivewheel', 
            'wheelbase', 
            'carlength',
            'carwidth', 
            'curbweight', 
            'enginetype', 
            'cylindernumber', 
            'enginesize', 
            'boreratio',
            'horsepower', 
            'price']]


# In[12]:


cars_dumm = pd.get_dummies(data=cars[['fueltype', 
                                    'aspiration', 
                                    'carbody', 
                                    'drivewheel', 
                                    'enginetype', 
                                    'cylindernumber', 
                                    'company']], 
                                    drop_first = True)

cars_lr = pd.concat([cars.drop(['fueltype', 
                                'aspiration', 
                                'carbody', 
                                'drivewheel', 
                                'enginetype', 
                                'cylindernumber', 
                                'company'],
                                axis='columns'), cars_dumm], axis=1)


# In[13]:


smf.ols('price ~ horsepower', cars_lr).fit().summary()


# **В первой модели мы используем все имеющиеся предикторы**

# In[14]:


X = cars_lr.drop(['price'], axis='columns')
X = sm.add_constant(X)
y = cars_lr['price']


# In[15]:


model_1 = sm.OLS(y, X).fit().summary()


# In[16]:


model_1


# In[17]:


ctk = cars_lr.columns[~cars_lr.columns.str.startswith('company_')]
ctk


# **Во второй модели мы используем все предикторы, кроме марок машин**

# In[18]:


X2 = cars_lr[ctk].drop(['price'], axis='columns')
X2 = sm.add_constant(X2)

model_2 = sm.OLS(y, X2).fit().summary()

model_2


# **Оставляем вторую модель так как в ней меньше предикторов, а R2 изменился не очень сильно, часть марок вообще не значима.**

# Выбранная модель объясняет примерно 90 % дисперсии.
# 
# Среди предикторов 10 из 27 оказались не значимыми (p > 0.05). 
# 
# Пример интерпретации: при единичном изменении показателя horsepower, цена возрастает на 86.8164.

# In[ ]:




