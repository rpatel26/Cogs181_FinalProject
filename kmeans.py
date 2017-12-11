#!/home/lightningclaw001/anaconda3/bin/python

import numpy as np
import pandas as pd
import sys

if (len(sys.argv) != 2):
   print("Usage: kmeans.py training_data.csv")


data = pd.read_csv(sys.argv[1], delimiter = ',', comment = '%', skiprows=132)

labels = data.iloc[:,-1]
one_hot = pd.get_dummies(labels)

data = data.iloc[:, :-1]

y_data = one_hot.as_matrix()
x_data = data.as_matrix()
