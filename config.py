'''
This will set up the model and model configuration that will be used by the main program

Author: Abhoy, Yuvraj
'''

# Load Necessary Libraries
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

train_file_path='SP_Train.xlsx'
output_file_path='Insight_Learners_SalesPredictionTask_submission.xlsx'

target_column='Item_Outlet_Sales'   # The feature that is to be predicted

# Choosing the best model
model=RandomForestRegressor(max_depth=110, min_samples_leaf=41, min_samples_split=10, n_estimators=161)