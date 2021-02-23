import re
import string

import nltk
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer

class LogisticRegression():
    def __init__(self):
        self.data = pd.read_excel(
            "/home/wambui/Fiverr/Python/notebooks/sanveohr/mockdata_set.xlsx", 
            sheet_name="input_1_conduit_data")
    
    def preprocessing(self):
        material = dict()
        stopwords = nltk.corpus.stopwords.words("english")
        word_net = nltk.WordNetLemmatizer()

        # data = pd.DataFrame(data, index=[0])
        data = self.data
        print(data.head())
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

        def clean_text(text):
            text = str(text).lower()
            text = "".join(word for word in str(text) if word not in string.punctuation)
            tokens = re.split("\W+", text)
            lemmatized = [word_net.lemmatize(word) for word in tokens if word not in stopwords]
            return lemmatized

        count_vector = CountVectorizer(analyzer=clean_text)
        count_vector_ = CountVectorizer(analyzer=clean_text)
        vector = count_vector.fit_transform(data["short_desc"])
        vector_ = count_vector_.fit_transform(data["long_desc"])

        

        print(pd.DataFrame(vector.todense(), columns=count_vector.get_feature_names()))
        print(pd.DataFrame(vector_.todense(), columns=count_vector_.get_feature_names()))
        
        return True       

# alg = LogisticRegression()
# alg.preprocessing(data = {
#             "Short Desc": "Metallic Liquidtight Conduit, Flexible, LA, Galvanized Steel (Inner Core), PVC (Jacket)",
#             "Long Desc": "_x000D_Type LA_x000D_",
#             "Size": "2-1/2 in.",
#             "Length": "100 ft.",
#             "Material": "PVC Coated Galvanized Steel"
#         })

# alg = LogisticRegression()
# alg.preprocessing(data = {
#     "Short Desc": {
#         "0": "Metallic Liquidtight Conduit, Flexible, LA, Galvanized Steel (Inner Core), PVC (Jacket)",
#         "1": "Metallic Liquidtight Conduit, Flexible, Stainless Steel, 0 to 275 °F, 14 in Bend Radius, 25 ft L, 2 in."
#     },
#     "Long Desc": {
#         "0": "_x000D_Type LA_x000D_",
#         "1": "Metallic Liquidtight Conduit, Flexible, Stainless Steel, 0 to 275 °F, 14 in, 25 ft, 2 in"
#     },
#     "Size": {
#         "0": "2-1/2 in.",
#         "1": "2 in."
#     },
#     "Length": {
#         "0": "100 ft.",
#         "1": "25 ft."
#     },
#     "Material": {
#         "0": "PVC Coated Galvanized Steel",
#         "1": "stainless steel" 
#     }
# })
