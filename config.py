'''
This will set up the model and model configuration that will be used by the main program

Author: Abhoy, Yuvraj
'''


from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

import pandas as pd

# preprocessing

train_file_path='SP_Train.xlsx'
output_file_path='SP_Output.xlsx'

model=RandomForestRegressor(max_depth=50)