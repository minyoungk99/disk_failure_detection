import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data = pd.read_csv(".\\data\\harddrive.csv") # 3,179,295 x 95
data = pd.read_csv(".\\data\\harddrive_small.csv") # subset, 100,000 x 95

# remove all raw columns
data.drop(list(data.filter(regex="raw$")), axis=1, inplace=True)
data.drop(['Unnamed: 0', 'serial_number'], inplace=True, axis=1)
print(data.head())
print(data.shape)
print(data.columns)


