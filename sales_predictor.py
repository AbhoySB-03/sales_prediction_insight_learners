'''
Main program for predicting the sales

Author: Abhoy, Yuvraj
'''
from config import *
import argparse as agp
from preprocessing import *
import numpy as np

DEBUG=True

def load_data(fp):
    '''
    load_data
    ===
    Load the dataset from a csv file or xlsx file based on its extension
    '''
    if fp.endswith(".csv"):
        return pd.read_csv(fp)
    elif fp.endswith('.xlsx') or fp.endswith('.xls'):
        return pd.read_excel(fp)

def save_data(data: pd.DataFrame,fp):
    '''
    save_data
    ===
    Save the dataset to a csv file or xlsx file based on its extension
    '''
    if fp.endswith(".csv"):
        return data.to_csv(fp, index=False)
    elif fp.endswith('.xlsx') or fp.endswith('.xls'):
        return data.to_excel(fp,index=False)
    
def extract_from_dataset(data: pd.DataFrame, target_column=None, remove_columns=[]):
    '''
    extract_from_dataset
    ===
    Extract X and y from the given data_frame based on its target column, if it is provided, otherwise just returns X. Also removes any unncessary columns
    
    Parameters
    ---
    data : DataFrame
    
    target_column : Any
    
    remove_column: List
        list of columns to be removed
    '''
    if target_column is not None:
        remove_columns+=[target_column]
        y=data[target_column]

    X=data.drop(columns=remove_columns)

    return X,y if target_column is not None else X

def main():

    # Argument Parser for passing train and test file locations as arguments
    parser=agp.ArgumentParser()
    parser.add_argument('--test_file', type=str, help='File Path to Test Dataset')
    parser.add_argument('--train_file', type=str, default=None, help='File Path to Training Dataset')

    args=parser.parse_args()
    test_file_path=args.test_file

    ######################
    # Load training data #
    ######################

    # Use default train file if argument not provided    
    if args.train_file is not None:
        train_file_path=args.train_file
    try:
        data=load_data(train_file_path)
    except FileNotFoundError:
        if args.train_file == None:
            print('The training file provided was not found. Please make sure the file path in config.py is correct.')
        else:
            print(f'{train_file_path} was not found')

    # Fill up missing data
    data=missing_data_processing(data)

    id_column='Item_Identifier'

    # Get X,y from dataframe
    X,y=extract_from_dataset(data, target_column,[id_column])

    # Perform preprocessing
    X,y=data_preprocessing(X,y)
    
    # apply PCA
    X,pca_obj=apply_pca(X,return_pca_object=True)

    # Fit the model
    model.fit(X,y)

    ######################
    # Load the Test Data #
    ######################


    data_test=(load_data(test_file_path))
    data_test=missing_data_processing(data_test)

    # Load Data and Perform necessary preprocessing
    if DEBUG:
        if not target_column in data_test.columns:
            raise KeyError(f'{target_column} not found. Perhaps use a test dataset with {target_column} present. Otherwise set DEBUG to False')
        X_inp,y_inp=extract_from_dataset(data_test, target_column, [id_column])
        X_inp, y_inp=data_preprocessing(X_inp, y_inp, remove_outlier=False)
    else:
        X_inp,_=extract_from_dataset(data_test,remove_columns=[id_column])        
        X_inp=data_preprocessing(X_inp, remove_outlier=False)


    X_inp=pca_obj.transform(X_inp)

    # Perform prediction
    y_out=model.predict(X_inp)

    
    out_data=pd.DataFrame(data_test[id_column])
    out_data=out_data.assign(**{target_column:y_out})

    if DEBUG:
        y_true=y_inp
        print(f"Root Mean Square Error: {np.sqrt(np.mean((y_out-y_true)**2))}")
        print(f"Normalized Root Mean Square Error: {np.sqrt(np.mean((y_out-y_true)**2))/(max(y_true)-min(y_true))}")

    # Save the predicted output in the 'output_file_path' file
    save_data(out_data, output_file_path)
    print(f'Output generated in {output_file_path}')


if __name__=="__main__":
    main()