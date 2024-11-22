import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.impute import SimpleImputer

def knn_outlier_detection(X, k, threshold):
    """
    Identifies outliers in the dataset using KNN.

    Parameters:
        X (array-like): The data matrix.
        k (int): Number of nearest neighbors to consider.
        threshold (float): The outlier threshold.

    Returns:
        array: A boolean array indicating outliers (True) and inliers (False).
    """

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X) 
    distances, _ = nbrs.kneighbors(X)
    outlier_scores = distances[:, k]  # Distance to the kth neighbor
    if threshold is None:
        threshold = np.percentile(outlier_scores, 95)

    print(threshold)
    is_outlier = outlier_scores > threshold
    return is_outlier

def data_preprocessing(x_data,y_data):

    numerical_features = ['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']
    print('With Outliers: ',x_data.shape[0])
    
    is_outlier = knn_outlier_detection(np.array(x_data[numerical_features]), k=2, threshold=None)
    x_data = x_data[~is_outlier].reset_index(drop=True)
    y_data = y_data[~is_outlier].reset_index(drop=True)

    print(f"Dataset size after outlier removal: {x_data.shape[0]}")
    categorical_data = ['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Type','Outlet_Size','Outlet_Location_Type']
    
    x_data = pd.get_dummies(x_data,columns=categorical_data,drop_first=True)

    return x_data, y_data

def cleaning_data(data):
    for item in data.columns:
        print(f'No. of Empty items in {item} is: {data[item].isnull().sum()}')
    imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
    data.iloc[:,1]=imputer.fit_transform(data.iloc[:,1].values.reshape(-1, 1))
    imputer_string = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    data.iloc[:,8]=imputer_string.fit_transform(data.iloc[:,8].values.reshape(-1, 1))

    # data.to_csv("./cleaned_dataset.csv", index=False) 

    return data