import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hitachi HDS5C3030ALA630
data = pd.read_csv('.\\data\\hitachi_data.csv') # 236690, 49

data.drop(list(data.filter(regex="raw$")), axis=1, inplace=True)
data.drop(['Unnamed: 0'], inplace=True, axis=1)
data.sort_values('date', inplace=True)

print(data.groupby('failure').size())


