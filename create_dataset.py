import pandas as pd

data = pd.read_csv("C:\\Users\\minyo\\OneDrive\\Desktop\\disk temp\\harddrive.csv") # 3,179,295 x 95

data.drop(list(data.filter(regex="raw$")), axis=1, inplace=True)

# counts of data by model
#a = data.groupby('model').size()
#a.to_csv('.\\data\\model_counts.csv')

# create dataset of single model as csv
model = 'WDC WD20EFRX'
x = data[data['model'] == model]
x.to_csv('.\\data\\' + model + '_data.csv')


model = 'WDC WD1600AAJS'
x = data[data['model'] == model]
x.to_csv('.\\data\\' + model + '_data.csv')
