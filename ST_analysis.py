import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# ST4000DM000
data = pd.read_csv('.\\data\\ST4000DM000_data.csv') # 236690, 49

data.drop(list(data.filter(regex="raw$")), axis=1, inplace=True)
data.drop(['Unnamed: 0', 'date', 'model'], inplace=True, axis=1) # dont need model, also can't smote string col
y = data['failure']
X = data.drop(['failure'], axis=1).fillna(0) # replace NAs with 0

#print(data.groupby('failure').size())
# 1.681M ok, 139 failures. Data is very imbalanced, so we will use SMOTE to make it more balanced by oversampling minotiry class

sm = SMOTE(random_state=99)
X_res, y_res = sm.fit_resample(X, y)

# 1681334 1's and 0's
# print(sum(y_res==1))
# print(sum(y_res==0))


