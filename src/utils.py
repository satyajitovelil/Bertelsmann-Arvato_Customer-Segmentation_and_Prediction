import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def load_info(metaPath, **kwargs):
    attributes_df = pd.read_excel(metaPath, **kwargs).drop('Unnamed: 0', axis=1)

    if 'Information level' in attributes_df.columns:
        attribute_info['Information level'] = attribute_info['Information level'].ffill()
    else:
        attributes_df[['Attribute', 'Description']] = attributes_df[['Attribute', 'Description']].ffill()

    attributes_df['Missing'] = attributes_df['Meaning'].apply(lambda x: 'unknown' in x if type(x)==str else x)
    return attributes_df

def get_attributes(substring, attributes_df):
    return list(attributes_df[attributes_df.Attribute.str.contains(substring)].Attribute.unique())

def get_attribute_info(attribute, attributes_df):
    return attributes_df[attributes_df.Attribute == attribute]

def get_categorical_attributes_info(attributes_df):
    categorical_attributes_info = attributes_df.loc[attributes_df.Value.apply(lambda x: (type(x)==str) & (x!='-1, 0') & (x!='-1, 9')), :]
    categorical_attributes_info = categorical_attributes_info.reset_index()
    categorical_attributes_info = categorical_attributes_info[categorical_attributes_info.Value!='…']
    return categorical_attributes_info

def get_binary_attributes_info(attributes_df):    
    binary_attrib = attributes_df.query("Missing == False").groupby('Attribute').size()
    binary_attrib_index = binary_attrib[attrib_binary == 2].index
    binary_attributes_info = attributes_df.loc[attributes_df.Attribute.apply(lambda x: x in binary_attrib_index), :].query("Missing==False")
    return binary_attributes_info

def get_typ_class_attrib_info(attributes_df):  
    typ_attributes = attributes_df.loc[attributes_df.Attribute.str.contains('TYP'), :]
    klasse_attributes = attributes_df.loc[attributes_df.Attribute.str.contains('KLASSE'), :]
    cols_typ = attributes_df.loc[attributes_df.Description.str.contains('typ'), :]
    cols_class = attributes_df.loc[attributes_df.Description.str.contains('class'), :]
    return pd.concat([typ_attributes, klasse_attributes, attributes_typ_descr, attributes_class_descr], axis=0) 


def find_columns(col_string, df):
    '''Search for columns in df that contain the substring col_string'''
    return df.columns[df.columns.str.contains(col_string)]
    
def get_unique_vals(df, column):
    print('Unique Values in Column {}: '.format(column), df.loc[:, column].unique())
    
# def replace_vals(df, col, func):
#     df.loc[:, col] = df.loc[:, col].apply(func)

def create_missing_val_list(x):
    """
    Converts the input string to a list of integers if it isn't already an integer.

    Parameters:
    x (int or str): The input argument. If x is already an integer, it is wrapped in a list and returned.
                     If x is a string, it is split at each comma, converted into integers and returned as a list.

    Returns:
    A list of integer values. 
    """
    if type(x)!=int:
        return [int(i) for i in x.split(',')]
    else:
        return [x]

def get_cols_to_drop(df, threshold=0.3):
    """
    This function calculates the percentage of missing values for each column of the DataFrame df 
    and returns a list containing column names with missing 
    values greater than threshold.
    
    Args:
    df (Pandas DataFrame): the data on which to calculate missing values
    threshold (float, optional): the minimum threshold for selecting columns, defaults to 0.3 if no value provided.
    
    Returns:
    cols_to_drop (list): list of column names with missing values greater than threshold
    
    """
    missing_val = df.isnull().sum() / len(df)
    missing_val = missing_val[missing_val>0]
    cols_to_drop = missing_val[missing_val > threshold].index
    return cols_to_drop

def get_missing_rows_percent(df, missingno):
    '''Returns proportion of missing rows'''
    df_missing_rows = df.isnull().sum(axis=1)
    df_missing_rows = df_missing_rows[df_missing_rows>0]
    missing_percent = (len(df_missing_rows[df_missing_rows>missingno])/df.shape[0])*100
    return missing_percent

def plot_evr(pca_i):
    """
    Plots a graph of the explained variance ratio (EVR) of the principal components for a given
    PCA object 'pca_i' that has already been fit to a dataset.

    Parameters
    ----------
    pca_i : PCA object
        An instance of the PCA class that has already been fit to a dataset.

    Returns
    -------
    None
        The function displays the plot but does not return anything.
    """
    var = pca_i.explained_variance_ratio_
    cum_var = np.cumsum(var)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(1, len(var)+1), cum_var,label='Cumulative explained variance')
    ax1.set_ylabel('Explained variance ratio')
    ax1.set_xlabel('Principal component index')
    ax2 = ax1.twinx()
    ax2.bar(range(1, len(var)+1), var, alpha=0.5, align='center', label='Individual explained variance')
    ax2.set_ylabel('Explained variance ratio')
    fig.legend()
    fig.tight_layout(pad=4)
    plt.show();

def get_feature_weights(pca_var, df, component):
    """
    This function takes in the results of PCA analysis, a pandas dataframe, and a specific principal component and returns a pandas dataframe containing feature weights for that component. 

    Args:
    - pca_var: results of PCA analysis
    - df: pandas dataframe for PCA analysis
    - component: principal component for which feature weights are required

    Returns:
    - feature_weights_df: pandas dataframe containing feature weights for specified principal component.
    """
    features = df.columns.values
    featureWeights = pca_var.components_[component]

    feature_weight_dict = {feature: weight for feature, weight in zip(features, featureWeights)}

    feature_weights_df = pd.DataFrame({'weights':feature_weight_dict}).reset_index().sort_values('weights', ascending=False)

    return feature_weights_df

def plot_feature_weights(feature_weights_i):
    """Plots a horizontal bar graph for the feature importance weights with most important three features on the left 
    and least important three features on the right.
    """
    plt.barh(feature_weights_i.head(3)['index'], feature_weights_i.head(3).weights)
    plt.barh(feature_weights_i.tail(3)['index'], feature_weights_i.tail(3).weights)
    plt.show();

def plot_clusters(df1, df2):
    '''Creates a Bar plot of number of observation in each cluster for each df'''
    cluster_count = pd.merge(pd.Series(df1, name='azdias').value_counts(), 
                             pd.Series(df2, name='customers').value_counts(),
                         left_index=True,
                         right_index=True
                        )
    cluster_count.sort_index(inplace=True)
    cluster_plot_data = cluster_count/cluster_count.sum()
    cluster_plot_data.plot.bar(figsize=(12,6), secondary_y='customers');