import numpy as np
import pandas as pd
working_dir = '~/Project/pred-future-sales'

def fill_na(df):
    for col in df.columns:
        if ('lag' in col) & (df[col].isnull().any()):
            df[col].fillna(0, inplace=True)
    return df

if __name__ == '__main__':
    train_test = pd.read_csv(working_dir + '/data/interim/train_test.csv')
    train_test['item_shop_first_sale'] = train_test['date_block_num'] - train_test.groupby(['item_id', 'shop_id'])['date_block_num'].transform('min')
    train_test['item_first_sale'] = train_test['date_block_num'] - train_test.groupby(['item_id'])['date_block_num'].transform('min')

    train_test = train_test[train_test['date_block_num'] > 11]

    train_test = fill_na(train_test)
    
    train_test.to_pickle(path=working_dir + '/data/interim/train_test.pkl')
