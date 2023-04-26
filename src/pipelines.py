import pandas as pd
import types


class Clean:
    '''
    Tracks all re-encodings and transformations applied to the DataFrame in order to clean it

    '''
    def __init__(self, df):
        """
        Initializes an instance of the Clean class.

        Parameters:
        df (pandas.DataFrame): The DataFrame to be cleaned.
        """
        self.clean_proc = {}
        self.df = df
        # self.cols_dropped = []
        
    def remap(self, col, map_dict):
        """
        Replaces values in a column according to a provided mapping dictionary.
        Parameters:
        col (str): The name of the column to be cleaned.
        map_dict (dict): A dictionary where keys are the current values in the column and values are the new values to replace the current ones.

        Returns:
        pandas.Series: The cleaned column.
        """
        if col in self.clean_proc.keys():
            self.clean_proc[col].append(('replace', map_dict))
        else:
            self.clean_proc[col] = [('replace', map_dict)]
        return self.df[col].replace(map_dict)
    
    def reapply(self, col, transformation):
        """
        Applies a function to a column to transform its values.

        Parameters:
        col (str): The name of the column to be cleaned.
        transformation (function): The function to be applied to the column's values.

        Returns:
        pandas.Series: The cleaned column.
        """
        if col in self.clean_proc.keys():
            self.clean_proc[col].append(('apply', transformation))
        else:
            self.clean_proc[col] = [('apply', transformation)]
        return self.df[col].apply(transformation)

    # def drop_cols(self, df, col):
    #     self.cols_dropped.append(col)
    #     df.drop(col, axis=1, inplace=True)

    def fit(self, col, map_transform):
        """
        Fits a transformation to a column.

        Parameters:
        col (str): The name of the column to be cleaned.
        map_transform (function or dict): The transformation to be applied to the column's values.

        """
        if type(map_transform) == types.FunctionType:
            self.reapply(col, map_transform)
        if type(map_transform) == dict:
            self.remap(col, map_transform)

    def transform(self, df=None):
        """
        Transforms the DataFrame using previously fitted transformation(s).

        Parameters:
        df (pandas.DataFrame, optional): The DataFrame to be transformed. If None, the initial DataFrame is used.

        Returns:
        pandas.DataFrame: The transformed DataFrame.
        """
        if isinstance(df, pd.DataFrame):
            df_transformed = df
        elif df == None:
            df_transformed = self.df
        for col, map_transforms in self.clean_proc.items():
            # if ((col in self.cols_dropped)|(col not in df_transformed.columns)):
            if (col not in df_transformed.columns):
                continue
            else:
                for map_transform in map_transforms:
                    if map_transform[0] == 'replace':
                        df_transformed[col] = df_transformed[col].replace(map_transform[1])
                    if map_transform[0] == 'apply':
                        df_transformed[col] = df_transformed[col].apply(map_transform[1])
        # return df_transformed

    def fit_transform(self, col, map_transform):
        """
        Fits and applies a transformation to a column.

        Parameters:
        col (str): The name of the column to be cleaned.
        map_transform (function or dict): The transformation to be applied to the column's values.

        Returns:
        pandas.DataFrame: The transformed DataFrame.
        """
        if type(map_transform) == types.FunctionType:
            self.df[col] = self.reapply(col, map_transform)
        if type(map_transform) == dict:
            self.df[col] = self.remap(col, map_transform)


class FeatureEngineer:
    
    def __init__(self, *args):
        """
        Constructor: Creates an instance of Feature Engineer class
        
        Parameters:
        args : list
            List of Pandas dataframes
        """
        self.transformations_dict = {}
        self.args = args

    def apply_transform(self, new_col, from_col, transformation):
        """
        Method to apply transformation to column of a dataframe.
        
        Parameters:
        new_col : str
            Name of new column created by transformation
        from_col : str
            Name of column to be transformed
        transformation : function
            Function to be applied to transform the data in from_col
        
        Returns:
        None
            This method only modifies transformations_dict and Pandas dataframes passed to the constructor
        """
        if from_col in self.transformations_dict.keys():
            self.transformations_dict[from_col].append({new_col:transformation})
        else:
            self.transformations_dict[from_col] = [{new_col:transformation}]
        for df in self.args:
            df[new_col] = df[from_col].apply(transformation)

    def apply_remap(self, new_col, from_col, map_dict):
        """
        Method to remap values of a column of a dataframe
        
        Parameters:
        new_col : str
            Name of new column created with the remap
        from_col : str
            Name of column to be remapped\n        map_dict : dict
            A dictionary containing original, new value pairs to be remapped
        
        Returns:
        None
            This method only modifies transformations_dict and Pandas dataframes passed to the constructor
        """
        if from_col in self.transformations_dict.keys():
            self.transformations_dict[from_col].append({new_col:map_dict})
        else:
            self.transformations_dict[from_col] = [{new_col:map_dict}]
        for df in self.args:
            df[new_col] = df[from_col].replace(map_dict)


    def transform(self, df):
        """
        Method to apply all transformations and maps in `transformations_dict` to given dataframe
        
        Parameters:
        df : pandas.DataFrame
            Dataframe to transform
        
        Returns:
        pandas.DataFrame
            Transformed dataframe
        """
        for from_col, new_col_trans_list in self.transformations_dict.items():
            for new_col_trans in new_col_trans_list:
                # transtype, new_col_trans = new_col_trans_tuple[0], new_col_trans_tuple[1]
                new_col, map_transform = list(new_col_trans.keys()), list(new_col_trans.values())
                if type(map_transform[0]) == dict:
                    df[new_col[0]] = df[from_col].replace(map_transform[0])
                if type(map_transform[0]) == types.FunctionType:
                    df[new_col[0]] = df[from_col].apply(map_transform[0])