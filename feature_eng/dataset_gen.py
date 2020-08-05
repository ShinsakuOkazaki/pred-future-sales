import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import product
working_dir ='~/Project/pred-future-sales'
if __name__ == '__main__':
    sales_train = pd.read_csv(working_dir + '/data/raw/sales_train.csv')
    test = pd.read_csv(working_dir + '/data/raw/test.csv')
    shops = pd.read_csv(working_dir + '/data/raw/shops.csv')
    items = pd.read_csv(working_dir + '/data/raw/items.csv')
    item_categories = pd.read_csv(working_dir + '/data/raw/item_categories.csv')



    ##############sales_train#################
    sales_train = sales_train.astype({'date': 'datetime64', 'date_block_num': 'int64', 'shop_id': 'int64', 
                            'item_id': 'int64', 'item_price': 'float64', 'item_cnt_day': 'float64'})
    # Removed rows whose "item_price"s are more than 100000 and "item_cnt_day"s are more than 1000.
    # Replaced a value which is item_price with -1 to median of item_price.
    sales_train = sales_train[(sales_train.item_price <= 100000) & (sales_train.item_cnt_day <= 1000)]
    sales_train.loc[sales_train.item_price < 0, 'item_price'] = np.median(sales_train.item_price.values)
    sales_train['revenue'] = sales_train['item_price'] * sales_train['item_cnt_day']

    ##############shops#################
    # cleaning up and
    # create city and city_code feature
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split().map(lambda x: x[0])
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops['city_code'] = LabelEncoder().fit_transform(shops['city'])


    ##############item_categories#################
    # create type and subtype feature
    split_temp = item_categories['item_category_name'].str.split(' - ')
    item_categories['type'] = split_temp.map(lambda x: x[0])
    item_categories['subtype'] = split_temp.map(lambda x: x[1] if len(x) > 1 else x[0])
    item_categories['type_code'] = LabelEncoder().fit_transform(item_categories['type'])
    item_categories['subtype_code'] = LabelEncoder().fit_transform(item_categories['subtype'])

    ##############test#################
    test['date_block_num'] = 34
    test['date_block_num'] = test['date_block_num'].astype(np.int32)
    test['shop_id'] = test['shop_id'].astype(np.int32)
    test['item_id'] = test['item_id'].astype(np.int32)

    ##############train_test set generation#################
    train_test = []
    cols = ['date_block_num', 'shop_id', 'item_id']
    for i in range(34):
        window = sales_train[sales_train['date_block_num'] == i]
        train_test.append(np.array(list(product([i], window['shop_id'].unique(), window['item_id'].unique())), dtype='int32'))
    train_test = pd.DataFrame(np.vstack(train_test), columns=cols)
    train_test.astype(dtype={'date_block_num': 'int32', 'shop_id': 'int32', 'item_id': 'int32'})
    train_test.sort_values(cols, inplace=True)
    group = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
    group.columns = ['item_cnt_month'].reset_index(replace=True)
    train_test = pd.merge(train_test, group, on=cols, how='left')
    train_test['item_cnt_month'] = (train_test['item_cnt_month']
                                    .fillna(0)
                                    .clip(0,20) # NB clip target here
                                    .astype(np.float64))

    train_test = pd.merge(train_test, shops, on=['shop_id'], how='left')
    train_test = pd.merge(train_test, items, on=['item_id'], how='left')
    train_test = pd.merge(train_test, item_categories, on=['item_category_id'], how='left')
    train_test.drop(['shop_name', 'city', 'item_name', 'item_category_name', 'type', 'subtype'],axis=1, inplace=True)

    ################save files###########################
    sales_train.to_csv(path_or_buf=working_dir + 'data/interim/sales_train.csv', index=False)
    shops.to_csv(path_or_buf=working_dir + 'data/interim/shops.csv', index=False)
    item_categories.to_csv(path_or_buf=working_dir + 'data/interim/item_categories.csv', index=False)
    train_test.to_csv(path_or_buf=working_dir + 'data/interim/train_test.csv', index=False)