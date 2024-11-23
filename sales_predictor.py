'''
Main program for predicting the sales data

Author: Abhoy, Yuvraj
'''
from config import *
import argparse as agp
from preprocessing import *
import numpy as np


def load_data(fp):
    if fp.endswith(".csv"):
        return pd.read_csv(fp)
    elif fp.endswith('.xlsx') or fp.endswith('.xls'):
        return pd.read_excel(fp)

def save_data(data: pd.DataFrame,fp):
    if fp.endswith(".csv"):
        return data.to_csv(fp, index=False)
    elif fp.endswith('.xlsx') or fp.endswith('.xls'):
        return data.to_excel(fp,index=False)
    
def extract_from_dataset(data: pd.DataFrame, target_column=None, remove_columns=[]):
    if target_column is not None:
        remove_columns+=[target_column]
        y=data[target_column]

    X=data.drop(columns=remove_columns)

    return X,y if target_column is not None else X

def main():

    parser=agp.ArgumentParser()
    parser.add_argument('--test_file', type=str, help='File Path to Test Dataset')
    parser.add_argument('--train_file', type=str, default=None, help='File Path to Training Dataset')

    args=parser.parse_args()
    test_file_path=args.test_file

    if args.train_file is not None:
        train_file_path=args.train_file

    data=load_data(train_file_path)
    data=missing_data_processing(data)

    target_column='Item_Outlet_Sales'
    id_column='Item_Identifier'

    X,y=extract_from_dataset(data, target_column,[id_column])

    X,y=data_preprocessing(X,y)
    
    X,pca_obj=apply_pca(X,True)
    model.fit(X,y)

    data_test=(load_data(test_file_path))
    data_test=missing_data_processing(data_test)

    # X_inp,y_inp=extract_from_dataset(data_test, target_column, [id_column])
    # X_inp, y_inp=data_preprocessing(X_inp, y_inp)

    X_inp,_=extract_from_dataset(data_test,remove_columns=[id_column])
    X_inp=data_preprocessing(X_inp, remove_outlier=False)

    x_inp_indices=X_inp.index
    X_inp=pca_obj.transform(X_inp)
    y_out=model.predict(X_inp)
    out_data=pd.DataFrame(data_test[id_column][x_inp_indices])
    out_data=out_data.assign(**{target_column:y_out})

    # y_true=y_inp
    # print(f"Root Mean Square Error: {np.sqrt(np.mean((y_out-y_true)**2))}")
    # print(f"Normalized Root Mean Square Error: {np.sqrt(np.mean((y_out-y_true)**2))/(max(y_true)-min(y_true))}")
    save_data(out_data, output_file_path)
    print(f'Output generated in {output_file_path}')



if __name__=="__main__":
    main()