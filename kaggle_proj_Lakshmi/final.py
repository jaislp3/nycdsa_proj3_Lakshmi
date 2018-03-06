
# coding: utf-8

# In[ ]:


#superbend_1 from https://www.kaggle.com/tunguz/blend-of-blends-1/output
 

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import minmax_scale

submission_1 = pd.read_csv("kaggle_Lakshmi_working/4.ensembling/mysubmission11.csv")                  #0.9851+
submission_2 = pd.read_csv("kaggle_Lakshmi_working/4.ensembling/superblend_1.csv")                   #0.9854

blend = submission_2.copy()
col = blend.columns

col = col.tolist()
col.remove('id')

a = 0.5
b = 0.5


blend[col] = a*minmax_scale(submission_1[col].values) + b*minmax_scale(submission_2[col].values) 
blend.to_csv("mysubmission12.csv", index=False)
 #0.98754

