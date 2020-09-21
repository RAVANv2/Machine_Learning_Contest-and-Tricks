from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from tqdm import tqdm

"""
- Label Encoding
- One hot encoding
- Binarization
"""

class categoricalFeature:
    """
    df: pandas dataframe
    categorical_features: List of categorical columns .. e.g.["ord 1","ord 2","nom 1","nom 3"]
    encoding_type: label, binary, ohe 
    """
    def __init__(self, df, categorical_feature, encoding_type, handling_nan=False):
        self.df = df
        self.cat_features = categorical_feature
        self.enc_type = encoding_type
        self.label_encoder = dict()
        self.label_binarizer = dict()
        self.handling_nan = handling_nan

        '''Handling The Missing values'''
        if self.handling_nan:
            print("Intitiating Missing value Processes")
            for c in tqdm(self.cat_features):
                self.df[c] = self.df[c].astype(str).fillna("-999999")
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in tqdm(self.cat_features):
            le = LabelEncoder()
            le.fit(self.df[c].values)
            self.output_df[c] = le.transform(self.df[c].values)
            self.label_encoder[c] = le
        return self.output_df

    def _label_binarization(self):
        for c in tqdm(self.cat_features):
            lb = LabelBinarizer()
            lb.fit(self.df[c].values)
            val = lb.transform(self.df[c].values) # Return array with binary code
            self.output_df = self.output_df.drop(c,axis=1)
            for j in range(val.shape[1]):
                new_column_name = c + f"__bin_{j}"
                self.output_df[new_column_name] = val[:, j]
            self.label_binarizer[c] = lb

        return self.output_df

    def _one_hot(self):
        ohe = OneHotEncoder()
        ohe.fit(self.df[self.cat_features].values)
        return ohe.transform(self.output_df[self.cat_features].values)


    def fit_transform(self):
        if self.enc_type == "label":
            print("Intitiating Label Encoder Process.....")
            return self._label_encoding()
        elif self.enc_type == "binary":
            print("Intitiating Binary Label Encoder Process.....")
            return self._label_binarization()
        elif self.enc_type == "ohe":
            print("Initiating One Hot Encoder Process .....")
            return self._one_hot()
        else:
            raise Exception("Encoding type is Incorrect")
    
    '''For test Dataset '''
    def transform(self,dataframe):

        if self.handling_nan:
            for c in tqdm(self.cat_features):
                dataframe = dataframe.astype(str).fillna("-999999")
        
        if self.enc_type == "label":
            for c, le in self.label_encoder.items():
                dataframe[c] = le.transform(dataframe[c].values)
            return dataframe
        
        elif self.enc_type == "binary":
            for c, lb in self.label_binarizer.items():
                val = lb.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)

                for j in range(val.shape[1]):
                    new_column_name = c + "__bin_{j}"
                    dataframe[new_column_name] = val[:j]
            return dataframe
        


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv').head(500)
    df_test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv').head(500)

    df_test['target'] = -1
    train_idx = df['id'].values
    test_idx = df_test['id'].values

    full_data = pd.concat([df,df_test])

    cols = [c for c in df.columns if c not in ["id","target"]]
    print(cols)
    cat_object = categoricalFeature(df = full_data,
                                    categorical_feature = cols,
                                    encoding_type = "binary",
                                    handling_nan = True)
    
    full_data_transformed = cat_object.fit_transform()

    train_df = full_data_transformed[full_data_transformed['id'].isin(train_idx)].reset_index(drop=True)
    test_df = full_data_transformed[full_data_transformed['id'].isin(test_idx)].reset_index(drop=True)

    print(train_df.shape, test_df.shape)