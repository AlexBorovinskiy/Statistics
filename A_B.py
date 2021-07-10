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


# # Задание
# Представьте, что вы работаете аналитиком в очень крупной компании по доставке пиццы над приложением для курьеров (да, обычно в таких компаниях есть приложение для курьеров и отдельно приложение для потребителей).
# 
# У вас есть несколько ресторанов в разных частях города и целый штат курьеров. Но есть одна проблема, к вечеру скорость доставки падает из-за того, что курьеры уходят домой после рабочего дня, а количество заказов лишь растет. Это приводит к тому, что в момент пересмены наша доставка очень сильно проседает в эффективности. 
# 
# Наши data scientist-ы придумали новый алгоритм, который позволяет курьерам запланировать свои последние заказы перед окончанием рабочего дня так, чтобы их маршрут доставки совпадал с маршрутом до дома. То есть, чтобы курьеры доставляли последние свои заказы за день как бы "по пути" домой. 
# 
# Вы вместе с командой решили раскатить A/B тест на две равные группы курьеров. Часть курьеров использует старый алгоритм без опции "по пути", другие видят в своем приложении эту опцию и могут ее выбрать. Ваша задача – проанализировать данные эксперимента и помочь бизнесу принять решение о раскатке новой фичи на всех курьеров.

# # Гипотезы для проверки: 
# **Нулевая гипотеза (H0)**: Разницы между средним временем доставки в тестовой и контрольной группе нет
# 
# **Альтернативная гипотеза (H1)**: Разница между средним временем доставки в тестовой и контрольной группе есть

# In[2]:


delivery = pd.read_csv('https://stepik.org/media/attachments/lesson/385916/experiment_lesson_4.csv')


# In[3]:


delivery


# In[4]:


delivery.query('experiment_group == "control"').groupby('district').delivery_time.hist()


# In[5]:


delivery.query('experiment_group == "test"').groupby('district').delivery_time.hist()


# In[7]:


n_c = delivery.query('experiment_group == "control"').order_id.count()

n_t = delivery.query('experiment_group == "test"').order_id.count()

sd_t = delivery.query('experiment_group == "test"').delivery_time.std()

sd_c = delivery.query('experiment_group == "control"').delivery_time.std()


# In[8]:


round(sd_t,2)


# In[9]:


round(sd_c,2)


# In[10]:


d_t = stats.shapiro(delivery[delivery['experiment_group'] == 'test']['delivery_time'].sample(1000, random_state=17))

d_c = stats.shapiro(delivery[delivery['experiment_group'] == 'control']['delivery_time'].sample(1000, random_state=17))


# Для того, чтобы проверить нормальность распределения я использую тест Шапиро-Уилка. Этот тест показывает, что значения в тестовой группе распределены 
# нормально. В контрольной группе распределение является нормальным.
# 
# Стандартное отклонение времени доставки в тесте равно 9.88
# 
# Стандартное отклонение времени доставки в контроле равно 9.99
# 

# In[11]:


stats.ttest_ind(delivery.query('experiment_group == "test"').delivery_time, delivery.query('experiment_group == "control"').delivery_time)


# Для сравнения средних в данных экспериментальных группах я использую тест Стьюдента. 
# 
# Статистика в тесте равна -43
# 
# p-value <= 0.05
# 

# In[13]:


c_mean = delivery.query('experiment_group == "control"').delivery_time.mean()

t_mean = delivery.query('experiment_group == "test"').delivery_time.mean()


# In[14]:


perc = (t_mean - c_mean) / c_mean * 100

round(perc,2)


# Время доставки в тестовой группе изменилось на 13,35 %

# **Новый способ доставки доказал эффективность, можно раскатывать на всех.**

# In[ ]:




