'''
Main program for predicting the sales data

Author: Abhoy, Yuvraj
'''
from config import *
import argparse as agp

parser=agp.ArgumentParser()
parser.add_argument('--test_file', type=str, help='File Path to Test Dataset')
parser.add_argument('--train_file', type=str, default=None, help='File Path to Training Dataset')

args=parser.parse_args()
test_file_path=args.test_file
if args.train_file is not None:
    train_file_path=args.train_file

def preprocess(data: pd.DataFrame):
    return data.dropna()

def load_data(fp):
    if fp.endswith(".csv"):
        return pd.read_csv(fp)
    elif fp.endswith('.xlsx') or fp.endswith('.xls'):
        return pd.read_excel(fp)

def main():
    data=load_data(train_file_path)

    target_column='Item_Outlet_Sales'
    id_column='Item_Identifier'
    remove_columns=[]

    X=data.drop(columns=remove_columns+[target_column, id_column])
    y=data[target_column]

    X,y=preprocess(X,y)
    model.fit(X,y)

    data_test=(load_data(test_file_path))

    X_inp=data_test.drop(columns=remove_columns+[target_column, id_column])
    y_out=model.predict(X_inp)

    X_inp,y_out=preprocess(X_inp, y_out)
    out_data=pd.DataFrame([data_test[id_column],y_out])

    print(out_data)



if __name__=="__main__":
    main()