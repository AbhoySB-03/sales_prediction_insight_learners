'''
This will set up the model and model configuration that will be used by the main program

Author: Abhoy, Yuvraj
'''


from sklearn.ensemble import RandomForestRegressor
import pandas as pd

train_file_path='SP_Train.xlsx'
output_file_path='SP_Output.xlsx'

model=RandomForestRegressor(100)