import ast
import re
import string

import joblib
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class RandomForest:
    def __init__(self):
        # self.data = pd.read_excel(
        #     "/home/wambui/Fiverr/Python/notebooks/sanveohr/mockdata_set.xlsx",
        #     sheet_name="input_1_conduit_data")
        path = ("/home/wambui/Fiverr/Python/mock_set/mock/apps/machine_learning_models/notebooks/sanveohr/{}")
        self.material_model = joblib.load(
                                    path.format(
                                        "material_random_forest.joblib"))
        self.length_model = joblib.load(
                                    path.format("length_random_forest.joblib"))
        self.type_model = joblib.load(
                                    path.format("type_random_forest.joblib"))
        self.size_model = joblib.load(
                                    path.format("/size_random_forest.joblib"))
        self.empty_dataframe = pd.read_excel(
                                    path.format("empty_dataframe.xlsx"))

        try:
            with open(path.format("type_.txt"), "r") as type_, open(path.format("material.txt"), "r") as material, open(path.format("length.txt"), "r") as length, open(path.format("size.txt"), "r") as size:
                self.type_ = type_.read()
                self.material = material.read()
                self.length = length.read()
                self.size = size.read()

        except Exception as e:
            return {"status": "Error", "message": str(e)}

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
        empty_dataframe = self.empty_dataframe

        try:
            data = pd.DataFrame(data, index=[0])
            data.columns = data.columns.str.lower().str.replace(" ", "_")
            data.rename(columns={"size": "size_", "length": "length_"}, inplace=True)

            data.dropna(inplace=True)

            labelencoder = LabelEncoder()

            def clean_text(text):
                """
                natural language processing
                """
                text = str(text).lower()
                text = str(text).replace("\n", " ")
                text = "".join(word for word in str(text) if word not in string.punctuation)
                tokens = re.split("\W+", text)
                lemmatized = [word_net.lemmatize(word) for word in tokens if word not in stopwords]
                return lemmatized

            count_vector = CountVectorizer(analyzer=clean_text)
            count_vector_ = CountVectorizer(analyzer=clean_text)
            vector = count_vector.fit_transform(data["short_desc"])
            vector_ = count_vector_.fit_transform(data["long_desc"])

            short_desc_df = pd.DataFrame(
                vector.todense(), columns=count_vector.get_feature_names()
                )
            long_desc_df = pd.DataFrame(
                vector_.todense(), columns=count_vector_.get_feature_names()
                )


            short_long_desc_df = pd.concat(
                [short_desc_df, long_desc_df], axis=1
                )
            short_long_desc_df = short_long_desc_df.groupby(
                short_long_desc_df.columns, axis=1).sum()


            short_long_desc_df = short_long_desc_df.drop(
                columns=list(
                    set(short_long_desc_df.columns.tolist())
                    - set(empty_dataframe.columns.tolist()))
                )

            empty_dataframe.drop(columns=["Unnamed: 0"], inplace=True)


            input_data = empty_dataframe.append(short_long_desc_df)
            input_data.fillna(value=0, inplace=True)

        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return input_data

    def predict(self, data):
        """
        Prediction on input data based on trained models
        """
        return self.material_model.predict(data),self.size_model.predict(data),self.length_model.predict(data),self.type_model.predict(data)

    def material_postprocessing(self, prediction):
        """
        Decode the encoded outputs for consumption by users
        """
        available_materials = self.material
        available_materials = ast.literal_eval(available_materials)

        for pred in available_materials:
            if prediction == pred:
                material = available_materials[pred]

        return {"prediction": prediction, "label": material, "status": "OK"}

    def size_postprocessing(self, prediction):

        available_sizes = self.size
        available_sizes = ast.literal_eval(available_sizes)

        for pred in available_sizes:
            if prediction == pred:
                size = available_sizes[pred]

        return {"prediction": prediction, "label": size, "status": "OK"}

    def length_postprocessing(self, prediction):
        available_lengths = self.length
        available_lengths = ast.literal_eval(available_lengths)

        for pred in available_lengths:
            if prediction == int(pred):
                length = available_lengths[pred]

        return {"prediction": prediction, "label": length, "status": "OK"}

    def type_postprocessing(self, prediction):
        available_types = self.type_
        available_types = ast.literal_eval(available_types)

        for pred in available_types:
            if prediction == pred:
                types = available_types[pred]

        return {"prediction": prediction, "label": types, "status": "OK"}

    def compute_prediction(self, data):
        """
        Apply preprocessing, prediction and post processing of data
        """
        try:
            data = self.preprocessing(data)
            material_predicted_output = self.predict(data)[0]
            size_predicted_output = self.predict(data)[1]
            length_predicted_output = self.predict(data)[2]
            type_predicted_output = self.predict(data)[3]

            material_predicted_output = self.material_postprocessing(material_predicted_output)
            size_predicted_output = self.size_postprocessing(
                                            size_predicted_output)
            length_predicted_output = self.length_postprocessing(
                                            length_predicted_output)
            type_predicted_output = self.type_postprocessing(
                                            type_predicted_output)

        except Exception as e:
            print(e)
            return {"status": "Error", "message": str(e)}

        return material_predicted_output, size_predicted_output, length_predicted_output, type_predicted_output