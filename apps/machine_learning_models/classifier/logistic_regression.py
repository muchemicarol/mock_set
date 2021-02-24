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
        """
        Preprocess the data:
            - import data
            - rename columns
            - encode material column
            - drop null columns
            - remove stopwords, punctuations and lemmatize text
            - vectorize the data to feed into ml model
        """
        material = dict()
        stopwords = nltk.corpus.stopwords.words("english")
        word_net = nltk.WordNetLemmatizer()
        
        try:
            # data = pd.DataFrame(data, index=[0])
            data = self.data
            print(data.head())
            data = data[["Short Desc", "Long Desc", "Size", "Length", "Material"]]
            data.columns = data.columns.str.lower().str.replace(" ", "_")
            data.rename(columns={"size": "size_", "length": "length_"}, inplace=True)
            for column, num in zip(data["material"].unique().tolist(), range(0, 38)):
                material[column]=num

            def material_column(column_value):
                """
                encode the material column (high ordinality)
                """
                for key, value in material.items():
                    if key == column_value:
                        return value

            data["material_"] = data["material"].apply(material_column)
            data.dropna(inplace=True)

            def clean_text(text):
                """
                natural language processing
                """
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
        
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return {"status": "OK"}