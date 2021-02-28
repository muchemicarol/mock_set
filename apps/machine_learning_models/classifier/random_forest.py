import re
import string

import joblib
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class RandomForest():
    def __init__(self):
        # self.data = pd.read_excel(
        #     "/home/wambui/Fiverr/Python/notebooks/sanveohr/mockdata_set.xlsx",
        #     sheet_name="input_1_conduit_data")
        self.model = joblib.load("/home/Fiverr/Python/notebooks/sanveohr/random_forest.joblib")

    def preprocessing(self, data):
        """
        Preprocess the data:
            - import data
            - rename columns
            - encode material column
            - drop null columns
            - remove stopwords, punctuations and lemmatize text
            - vectorize the data to feed into ml model
        """
        stopwords = nltk.corpus.stopwords.words("english")
        word_net = nltk.WordNetLemmatizer()

        try:
            # data = pd.DataFrame(data, index=[0])
            data = self.data
            data = data[["Short Desc", "Long Desc", "Size", "Length", "Material", "Type"]]
            data.columns = data.columns.str.lower().str.replace(" ", "_")
            data.rename(columns={"size": "size_", "length": "length_"}, inplace=True)

            data.dropna(inplace=True)

            labelencoder = LabelEncoder()

            for column in ["size", "length", "material", "type"]:
                mock_data[column] = mock_data[column].astype(str)
                mock_data[column + "_"] = labelencoder.fit_transform(mock_data[column])

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

            short_desc_df = pd.DataFrame(vector.todense(), columns=count_vector.get_feature_names())
            long_desc_df = pd.DataFrame(vector_.todense(), columns=count_vector_.get_feature_names())

            short_long_desc_df = pd.concat([short_desc_df, long_desc_df], axis=1)

        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return short_long_desc_df

    def predict(self, data):
        return self.model.predict(data)

    def postprocessing(self, prediction):
        """
        Decode the encoded outputs for consumption by users
        """

    def compute_prediction(self, material_df, size_df, length_df, type_df):
        try:
            data = self.preprocessing(data)
            predicted_output = self.predict(data)
            predicted_output = self.postprocessing(predicted_output)

        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return predicted_output