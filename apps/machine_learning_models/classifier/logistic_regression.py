import re
import string

import nltk
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer

class LogisticRegression():
    
    def preprocessing(self, data):
        material = dict()

        data = pd.DataFrame(data, index=[0])
        sheet_name="Input 1_Conduit Data"
        data = data[["Short Desc", "Long Desc", "Size", "Length", "Material"]]
        data.columns = data.columns.str.lower().str.replace(" ", "_")
        data.rename(columns={"size": "size_", "length": "length_"}, inplace=True)
        for column, num in zip(data["material"].unique().tolist(), range(0, 38)):
            material[column]=num

        def material_column(column_value):
            for key, value in material.items():
                if key == column_value:
                    return value

        data["material_"] = data["material"].apply(material_column)
        data.dropna(inplace=True)

        return data
