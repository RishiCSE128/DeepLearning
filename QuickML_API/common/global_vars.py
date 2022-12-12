import numpy as np
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