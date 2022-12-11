import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


global_vars = {
    "DATASET_FILE_PREFIX" : "C:\\Users\\sapta\\Documents\\git\\DeepLearning\\QuickML_API\\_docs\\dataset\\",

    "split_profile" : {
        'independent_vars' : ['Country', 'Age', 'Salary'],
        'dependent_vars' : ['Purchased'],
        'train_size' : 0.7
    },

    "impute_strategy" : {
            'missing_value' : np.nan,
            'strategy': 'mean'
    },

    "categorical_cols" : {
        'independent' : ['Country'],
        'dependent' : ['Purchased']
    },

    "debug" : False
}

def print_tab(df : pd.DataFrame, head = False) -> None:
    """prints pandas dataframe in tabular form 

    Args:
        df (pd.DataFrame): dataframe to print
        head (bool): if set to False (default) prints whole table
    """

    if head:
        print(tabulate(df.head(), headers = 'keys', tablefmt = 'pretty'))
    else:
        print(tabulate(df, headers = 'keys', tablefmt = 'pretty'))


def load_dataset(dataset_filename):
    """_summary_

    Args:
        dataset_filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    DATASET_FILE_PREFIX = global_vars['DATASET_FILE_PREFIX']
    source_dataframe = pd.read_csv(DATASET_FILE_PREFIX + dataset_filename)
    return source_dataframe

def get_DI_split(source_dataframe : pd.DataFrame, conv_to_np = False) -> dict:
    """_summary_

    Args:
        source_dataframe (pd.DataFrame): _description_
        conv_to_np (bool, optional): _description_. Defaults to False.

    Returns:
        dict: _description_
    """
    
    split_profile = global_vars['split_profile']
    X = source_dataframe[ split_profile['independent_vars'] ]
    y = source_dataframe[ split_profile['dependent_vars'] ]

    if conv_to_np:
        X = X.to_numpy()
        y = y.to_numpy()
    
    return {'X':X , 'y':y}

def get_col_list_with_null(dataframe : pd.DataFrame) -> list:
    """_summary_

    Args:
        dataframe (pd.DataFrame): _description_

    Returns:
        list: _description_
    """

    col_list_with_null = [] 
    for column in dataframe.columns:
        if dataframe[column].isnull().any():
            col_list_with_null.append(column)
    return col_list_with_null

def remove_missing_data(dataframe : pd.DataFrame ) -> pd.DataFrame:
    """_summary_

    Args:
        dataframe (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    impute_strategy = global_vars['impute_strategy']

    # instantiate imputer object
    si = SimpleImputer(missing_values=impute_strategy['missing_value'], 
                       strategy=impute_strategy['strategy'])

    # get all column names with missing data
    col_list_with_null = get_col_list_with_null(dataframe=dataframe)

    # replace all missing values 
    df_copy = dataframe.copy()
    df_copy[col_list_with_null] = si.fit(df_copy[col_list_with_null]).transform(df_copy[col_list_with_null])

    return df_copy

def  cat_endcoing(dataframe : pd.DataFrame, categorical_cols : list, ohe = True):
    """_summary_

    Args:
        dataframe_X (pd.DataFrame): _description_
        categorical_cols (list): _description_
        ohe (bool): _description__

    Returns:
        _type_: _description_
    """
    df = dataframe.copy()
    if ohe:  # one hot encoding 
        for column in categorical_cols:
            dummy_df = pd.get_dummies(df[column], columns=[column], prefix=column)  # dummy var table for each encoding 
            df = df.join(dummy_df).drop(column, axis = 1) # upate dataframe by joining 

    else:    # label encoding
        label_encoder = LabelEncoder() 
        for column in categorical_cols:
            df[column] = label_encoder.fit_transform(df[column])
    
    return df

def tt_split(dataframe_X : pd.DataFrame, dataframe_y : pd.DataFrame) -> dict :
    """performs train-test-split on the given dataset

    Args:
        dataframe (pd.DataFrame): source dataframe 
        split_profile (dict): dictionary must cotain a key 'tt_ratio' (float) [0.0 - 1.0] denotes training proportion 

    Returns:
        dict: _description_
    """
    X_train, X_test, y_train, y_test = train_test_split(dataframe_X, dataframe_y, train_size=global_vars['split_profile']['train_size'], random_state = 42)

    return {
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test
    }

def feature_scalling(tt_splitted_dfs : dict , scale_dummies = True) ->dict:
  """_summary_

  Args:
      tt_splitted_dfs (dict): _description_
      scale_dummies (bool, optional): _description_. Defaults to True.

  Returns:
      dict: _description_
  """
    
  sc_X = StandardScaler()
  if not scale_dummies:
    # non-categoriccal independent vars
    scalable_features = list(set(global_vars['split_profile']['independent_vars']) - set(global_vars['categorical_cols']['independent']))

  return{
    'X_train' : sc_X.fit_transform(tt_splitted_dfs['X_train']),
    'X_test' : sc_X.transform(tt_splitted_dfs['X_test']),
    'y_train' : tt_splitted_dfs['y_train'],
    'y_test' : tt_splitted_dfs['y_test']
  }

def data_pre_processing():
    # [1/]load dataset
    print(' Step 1/6 Loading Data... ')
    
    source_dataframe = load_dataset(dataset_filename="data_preprocessing.csv")
    
    print_tab(source_dataframe)if global_vars['debug'] else None

    # [2/]DI split
    print(' Step 2/6 Splitting Dependent and Independent samples... ') 
    
    splitted_data = get_DI_split(source_dataframe=source_dataframe)

    if global_vars['debug']:
        print('Independent (X)')
        print_tab(splitted_data["X"])
        print('Dependent (y)')
        print_tab(splitted_data["y"])

    # [3/]Replace missing data
    print(' Step 3/6 Replacing Missing data... ') 
    splitted_data["X"] = remove_missing_data(dataframe=splitted_data["X"])
    
    if global_vars['debug']:
        print_tab(splitted_data["X"])
        print_tab(splitted_data["y"])

    # [4/]Encode categorical variables
    print(' Step 4/6 Encoding categotical vatiables... ')
    df_X = cat_endcoing(dataframe=splitted_data["X"], categorical_cols=global_vars["categorical_cols"]["independent"])
    df_y = cat_endcoing(dataframe=splitted_data["y"], categorical_cols=global_vars["categorical_cols"]["dependent"], ohe=False)

    if global_vars['debug']:
        print_tab(df_X)
        print_tab(df_y)

    # [5/]Train-Test Split
    print(' Step 5/6 Splitting Training and Test Samples... ')
    tt_splitted_data = tt_split(dataframe_X=df_X, dataframe_y=df_y)

    if global_vars['debug']:
        for key in tt_splitted_data.keys():
            print('key')
            print_tab(tt_splitted_data[key])

    # [6/]Feature Scalling
    print(' Step 5/6 Scaling features... ')
    scalled_dfs = feature_scalling(tt_splitted_dfs=tt_splitted_data, scale_dummies=False)

    if global_vars['debug']:
        for key in scalled_dfs.keys():
            print(key)
            print_tab(scalled_dfs[key])
    
    return scalled_dfs