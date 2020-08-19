import numpy as np
import pandas as pd
working_dir = '~/Project/pred-future-sales'

# function to create lag freatures for 'col' based on 'date_block_num', 'shop_id', 'col'
def lag_feature(df, lags, col):
    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lag_' + str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df

# function to create dataframe which contains average lag features
def average_lag(df, group_keys, lag_nums, feature_names):
    size = len(group_keys)
    for i in range(size):
        group = df.groupby(group_keys[i]).agg({'item_cnt_month': ['mean']})
        group.columns = [feature_names[i]]
        group.reset_index(inplace=True)
        df = pd.merge(df, group, on=group_keys[i], how='left')
        df = lag_feature(df, lag_nums[i], feature_names[i])
        df.drop([feature_names[i]], axis=1, inplace=True)
    return df  

# function to select suitable trend feature
def select_trend(row):
    for i in lag_nums:
        if row['delta_item_price_lag_' + str(i)]:
            return row['delta_item_price_lag_' + str(i)]
        return 0

if __name__ == '__main__':
    sales_train = pd.read_csv(working_dir + '/data/interim/sales_train.csv')
    train_test = pd.read_pckle(working_dir + '/data/interim/train_test.pkl')


    # create item_cnt_month_lag_i (i = 1, 2, 3, 6, 12)
    train_test = lag_feature(train_test, [1, 2, 3, 6, 12], 'item_cnt_month')


    # create average lag features 
    group_keys = [['date_block_num'], ['date_block_num', 'item_id'], ['date_block_num', 'shop_id'], \
                ['date_block_num', 'item_category_id'], ['date_block_num', 'type_code'], ['date_block_num', 'subtype_code'], \
                ['date_block_num', 'city_code'], ['date_block_num', 'shop_id', 'item_category_id'], ['date_block_num', 'shop_id', 'type_code'], \
                ['date_block_num', 'shop_id', 'subtype_code'], ['date_block_num', 'item_id', 'city_code']\
                 ]
    lag_nums = [ [1], [1, 2, 3, 6, 12], [1, 2, 3, 6, 12], \
                [1, 2, 3, 6, 12], [1], [1], \
                [1], [1], [1], \
                [1], [1]\
                ]
    feature_names = ['item_cnt_month_avg', 'item_cnt_month_item_avg', 'item_cnt_month_shop_avg', \
                    'item_cnt_month_category_avg', 'item_cnt_month_type_avg', 'item_cnt_month_subtype_avg',\
                    'item_cnt_month_city_avg', 'item_cnt_month_shop_category_avg', 'item_cnt_month_shop_type_avg', \
                    'item_cnt_month_shop_subtype_avg', 'item_cnt_month_item_city_avg'\
                    ]
    train_test = average_lag(train_test, group_keys, lag_nums, feature_names)


    # create item_price_month_lag_i (i = 1, 2, 3, 4, 5, 6)
    group = sales_train.groupby(['item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_price_total_avg']
    group.reset_index(inplace=True)
    train_test = pd.merge(train_test, group, on=['item_id'], how='left')

    group = sales_train.groupby(['item_id', 'date_block_num']).agg({'item_price': ['mean']})
    group.columns = ['item_price_month_avg']
    group.reset_index(inplace=True)
    train_test = pd.merge(train_test, group, on=['item_id', 'date_block_num'], how='left')
    lag_nums = [1, 2, 3, 4, 5, 6]
    train_test = lag_feature(train_test, lag_nums, 'item_price_month_avg')

    # calculate delta_item_price_lag_i (i = 1, 2, 3, 4, 5 6)
    for i in lag_nums:
        train_test['delta_item_price_lag_' + str(i)] = (train_test['item_price_month_avg_lag_' + str(i)] - train_test['item_price_total_avg']) \
                                                        / train_test['item_price_total_avg']

    # create delta_item_price_lag selecting suitable delta_item_price_lag_i
    train_test['delta_item_price_lag'] = train_test.apply(select_trend, axis=1)
    train_test['delta_item_price_lag'] = train_test['delta_item_price_lag'].astype(np.float64)

    # drop features ('item_price_month_avg_lag_i', 'delta_item_price_lag_i')
    features_to_drop = ['item_price_total_avg', 'item_price_month_avg']
    for i in lag_nums:
        features_to_drop.extend(['item_price_month_avg_lag_' + str(i), 'delta_item_price_lag_' + str(i)])
    train_test.drop(features_to_drop, axis=1, inplace=True)

    ## create shop revenue trend features

    # calculate shop_month_revenue
    group = sales_train.groupby(['shop_id', 'date_block_num']).agg({'revenue': ['sum']})
    group.columns = ['shop_month_revenue']
    group.reset_index(inplace=True)
    train_test = pd.merge(train_test, group, on=['shop_id', 'date_block_num'], how='left')

    # calculate shop_month_avg_revenue
    group = train_test.groupby(['date_block_num']).agg({'shop_month_revenue': ['mean']})
    group.columns = ['shop_month_avg_revenue']
    group.reset_index(inplace=True)
    train_test = pd.merge(train_test, group, on=['date_block_num'], how='left')

    # calculate delta_shop_month_revenue_lag_1
    train_test['delta_shop_month_revenue'] = (train_test['shop_month_revenue'] - train_test['shop_month_avg_revenue']) / train_test['shop_month_avg_revenue']
    train_test = lag_feature(train_test, [1], 'delta_shop_month_revenue')
    # drop features 
    train_test.drop(['shop_month_revenue', 'shop_month_avg_revenue', 'delta_shop_month_revenue'], axis=1, inplace=True)

    # save dataframe     
    train_test.to_pickle(path=working_dir + '/data/interim/train_test.pkl')


