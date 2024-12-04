import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("train.csv")

data

data.info()

data.dropna(inplace = True)

data.info()

from sklearn.model_selection import train_test_split

X=data.drop(['Median_house_value',axis=1]);
y=data['median_house_value']
