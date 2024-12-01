import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\minyo\\OneDrive\\Desktop\\harddrive.csv") # 3,179,295 x 95
#data = pd.read_csv('.\\data\\hitachi_data.csv') # subset, 100,000 x 95

# remove all raw columns
data.drop(list(data.filter(regex="raw$")), axis=1, inplace=True)
data.drop(['serial_number'], inplace=True, axis=1)
#data.drop(['Unnamed: 0', 'serial_number'], inplace=True, axis=1)

# create model_counts.xlsx
#a = data.groupby('model').size()
#a.to_csv('.\\data\\model_counts.csv')

# save all hitachi model data into separate csv
#hitachi = pd.read_csv('.\\data\\Hitachi.csv', header=None)
#x = data[data['model'].isin(hitachi.iloc[:,0])]
#x.to_csv('.\\data\\hitachi_data.csv')

# Single hitachi model data
#x = data[data['model'] =='Hitachi HDS5C3030ALA630']
#x.to_csv('.\\data\\hitachi_data.csv')

# data for ST4000DM000
x = data[data['model'] =='ST4000DM000']
x.to_csv('.\\data\\ST4000DM000_data.csv')
