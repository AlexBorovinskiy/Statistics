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


# Вы работаете в приложении по доставке готовых продуктов. К вам пришел коллега с результатами двух тестов:
# 
# В первом **тестировали разрешение фотографий блюд** в приложении: пользователям показывались либо прямоугольные, либо новые квадратные.
# 
# Во втором: **была обновлена кнопка заказа**, и часть юзеров видела старый вариант, а часть – новый.
# 
# Коллега пришел к вам с просьбой: он посмотрел на графики и предположил, что среди групп могут встретиться различия. Ваша задача – помочь ему проверить гипотезы, сделать соответствующие выводы на основе статистических тестов и принять решения.

# In[2]:


meal_1 = pd.read_csv('https://stepik.org/media/attachments/lesson/385920/5_task_1.csv')

meal_2 = pd.read_csv('https://stepik.org/media/attachments/lesson/385920/5_task_2.csv')


# In[3]:


meal_1


# In[5]:


meal_A = meal_1.query('group == "A"').events
meal_B = meal_1.query('group == "B"').events
meal_C = meal_1.query('group == "C"').events


# С помощью теста Левена определяем, являются ли дисперсии внутри групп примерно одинаковыми (гомогенными).

# In[6]:


stats.levene(meal_A, meal_B, meal_C)


# **p-value > 0.05, не отклоняем нулевую гипотезу**

# In[7]:


stats.shapiro(meal_A.sample(1000, random_state=17))


# In[8]:


stats.shapiro(meal_B.sample(1000, random_state=17))


# In[9]:


stats.shapiro(meal_C.sample(1000, random_state=17))


# **Данные распределены нормально**

# In[10]:


stats.f_oneway(meal_A, meal_B, meal_C)


# Для сравнения средних я использую однофакторный дисперсионный анализ (f_oneway).
# 
# Значение статистики равно 2886, 
# 
# р-уровень значимости составил 0.0

# In[11]:


print(pairwise_tukeyhsd(meal_1['events'], groups=meal_1['group']).summary())


# **Между всеми группами есть статистически значимые различия.** Используем картинки В

# In[4]:


meal_2


# Так как требуется проверить, как пользователи отреагируют на изменение формата кнопки оформления заказа, с разбивкой по сегменту клиента, мы используем **многофакторный дисперсионный анализ**. 

# In[12]:


sns.distplot(meal_2.query('group == "test"').events)


# In[13]:


sns.distplot(meal_2.query('group == "control"').events)


# In[14]:


meal_2.query('group == "control" and segment == "high"').describe()


# In[15]:


meal_2.query('group == "control" and segment == "low"').describe()


# In[16]:


meal_2.query('group == "test" and segment == "low"').describe()


# In[17]:


meal_2.query('group == "test" and segment == "high"').describe()


# In[18]:


model = smf.ols(formula='events ~ segment + group + segment:group', data=meal_2).fit()
aov_table = anova_lm(model, typ=2)


# In[19]:


print(round(aov_table, 2))


# In[20]:


meal_2['combination'] = meal_2.group + ' \ ' + meal_2.segment


# In[21]:


meal_2


# In[22]:


print(pairwise_tukeyhsd(meal_2['events'], groups=meal_2['combination']).summary())


# **Разница между значением у тестовой группы сегмента low и контрольной группой этого же сегмента равна примерно 13**
# 
# **Для обоих сегментов показатели статистически значимо увеличились по сравнению с контрольной группой**
# 
# **Разница между control/high и test/high составила около 10**

# In[24]:


sns.pointplot(x='group', y='events', hue='segment', data=meal_2, capsize= .1)


# Фактор group оказался значимым,
# фактор segment – значимым,
# их взаимодействие – значимо. 
# 
# Судя по графику, для всех групп среднее значение events увеличилось, поэтому решение:
#     
# **выкатываем новую версию**
# 

# In[ ]:




