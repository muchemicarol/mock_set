import re
import string

import nltk
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer

class LogisticRegression():
    def preprocessing(self, data):
        data = pd.DataFrame(data, sheet_name="Input 1_Conduit Data")
        data = data[["Short Desc", "Long Desc", "Size", "Length", "Material"]]
        data.columns = data.columns.str.lower().str.replace(" ", "_")
        data.rename(columns={"size": "size_", "length": "length_"}, inplace=True)

        return data