import pandas as pd


def preprocess():
    print("\n================================== 读入数据 ==================================")
    df_train = pd.read_csv('dataset/train.csv')
    df_test = pd.read_csv('dataset/test.csv')
    print(df_train.info())
    print(df_test.info())
    
    # 1. 合并数据集
    print("\n================================== 合并数据 ==================================")
    df_feat = pd.concat([df_train, df_test], ignore_index=True)
    df_feat = df_feat.drop(['PassengerId', 'Name'], axis=1)
    print(df_feat.info())
    print(df_feat)
    
    # 2. 分离 Cabin 列
    print("\n================================== 分离 Cabin 列 ==================================")
    df_feat[['Deck','Num','Side']] = df_feat['Cabin'].str.split('/', expand=True)
    df_feat = df_feat.drop(['Cabin'], axis=1)
    print(df_feat.info())

    # 3. 处理数值型数据
    print("\n================================== 处理数值型数据 ==================================")
    # 提取类型为数值的列
    num_cols = df_feat.columns[df_feat.dtypes != 'object']
    # 归一化（均值为0，方差为1）
    df_feat[num_cols] = df_feat[num_cols].apply(lambda x: (x - x.mean()) / x.std())
    # 缺失值填充为0
    df_feat[num_cols] = df_feat[num_cols].fillna(0)
    print(df_feat.info())
    print(df_feat.describe())
    
    # 4. 处理类别型数据
    print("\n================================== 处理类别型数据 ==================================")
    cate_cols = df_feat.columns[df_feat.dtypes == 'object']
    # 用整数编码替换类别，NAN值用-1替换
    df_feat[cate_cols] = df_feat[cate_cols].apply(lambda x: pd.Categorical(x).codes)
    print(df_feat.info())
    print(df_feat)
    
    # 5. 分离训练集和测试集
    print("\n================================== 分离训练集和测试集 ==================================")
    df_train_processed = df_feat.iloc[:len(df_train)]
    df_test_processed = df_feat.iloc[len(df_train):]
    # 重新设置id
    df_train_processed['PassengerId'] = df_train['PassengerId'].values
    df_test_processed['PassengerId'] = df_test['PassengerId'].values
    print(df_train_processed.info())
    print(df_train_processed)
    print(df_test_processed.info())
    print(df_test_processed)
    
    return df_train_processed, df_test_processed

def save_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    df_train.to_csv('dataset/train_processed.csv', index=False)
    df_test.to_csv('dataset/test_processed.csv', index=False)


if __name__ == '__main__':
    train_df, test_df = preprocess()
    save_data(train_df, test_df)
