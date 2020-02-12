import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df0 = pd.read_csv('./adult.data')
df1 = pd.read_csv('./adult.data', header=None)
df2 = pd.read_csv('./adult.data',
                  names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country', 'Label'])
mappings = {}

for col_name in df2.columns:
    if df2[col_name].dtype == 'object':
        df2[col_name] = df2[col_name].astype('category')
        df2[col_name], mapping_index = pd.Series(df2[col_name]).factorize()
        mappings[col_name] = {}
        for i in range(len(mapping_index.categories)):
            mappings[col_name][i] = mapping_index.categories[i]
    else:
        mappings[col_name] = 'continuous'
