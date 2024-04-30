import pandas as pd
import torch
import numpy as np

#load the data
df = pd.read_excel("Mine_Dataset.xls", sheet_name="Normalized_Data").to_numpy()
np.random.shuffle(df)
df = df.transpose()

#create train test valid set
train_frac = 0.6
valid_frac = 0.2
train_size = int(df.shape[1] * train_frac)
valid_size = int(df.shape[1] * valid_frac)
train_X = df[:3, :train_size]
train_y = df[3, :train_size]
valid_X = df[:3, train_size:(train_size + valid_size)]
valid_y = df[3, train_size:(train_size + valid_size)]
test_X = df[:3, (train_size + valid_size):]
test_y = df[3, (train_size + valid_size):]

#test the shape
print(train_X.shape)
print(train_y.shape)
print(valid_X.shape)
print(valid_y.shape)
print(test_X.shape)
print(test_y.shape)




