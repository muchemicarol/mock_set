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
        path = ("/home/wambui/Fiverr/Python/notebooks/sanveohr/")
        self.material_model = joblib.load(
                                    path + "material_random_forest.joblib")
        self.length_model = joblib.load(path + "length_random_forest.joblib")
        self.type_model = joblib.load(path + "type_random_forest.joblib")
        self.size_model = joblib.load(path + "/size_random_forest.joblib")
        self.empty_dataframe = pd.read_excel("/home/wambui/Fiverr/Python/notebooks/sanveohr/empty_dataframe.xlsx", )

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
        available_materials = {
            23: 'PVC Coated Galvanized Steel', 35: 'stainless steel', 
            21: 'PVC', 18: 'Nylon', 8: 'Galvanized Steel', 
            33: 'Stainless Steel', 3: 'Aluminium', 26: 'Plenum-PVDF', 
            27: 'Riser-Nylon', 28: 'Riser-PVDF', 11: 'HDPE-Schedule 80', 
            10: 'HDPE', 31: 'Schedule 80', 29: 'Schedule 40', 
            6: 'Corrosion Resistant Plated Steel, PVC', 
            13: 'Hot Dipped Galvanized Steel, PVC', 
            15: 'Hot-Dippedâ Galvanizedâ Steel (Inner Core); PVC', 
            9: 'Galvanized Steel, Thermoplastic PVC (Jacket)', 
            22: 'PVC (Jacket), Plated Steel (Inner Core)', 
            24: 'PVC-Coated Galvanized Steel', 16: 'Non-Metallic', 
            5: 'Carbon Strip Steel', 7: 'Electroplated Steel', 
            14: 'Hot Galvanized Steel', 2: '316 Stainless Steel', 
            34: 'Steel', 4: 'Aluminum', 1: '304 Stainless Steel', 
            12: 'High Density Polyethylene', 0: '0', 19: 'Nylon Resin', 
            30: 'Schedule 40 PVC', 32: 'Schedule 80 PVC', 
            20: 'Nylon, Polyamide', 17: 'Non-Metallic/PVC', 25: 'PVDF'
            }
        for pred in available_materials:
            if prediction == pred:
                material = available_materials[pred]

        return {"prediction": prediction, "label": material, "status": "OK"}

    def size_postprocessing(self, prediction):

        available_sizes = {
            7: '2-1/2 in.', 6: '2 in.', 5: '1/2 in.', 11: '3/4 in.', 
            3: '1-1/2 in.', 9: '3 in.', 4: '1-1/4 in.', 2: '1 in.', 
            13: '4 in.', 10: '3-1/2 in.', 14: '5 in.', 15: '6 in.', 
            1: '1 1/4 in.', 8: '21 mm', 12: '4', 0: '0'
            }
        for pred in available_sizes:
            if prediction == pred:
                size = available_sizes[pred]

        return print({"prediction": prediction, "label": size, "status": "OK"})

    def length_postprocessing(self, prediction):
        available_lengths = {
            4: '100 ft.', 16: '25 ft.', 5: '1000 ft.', 26: '50 ft.', 
            37: 'Cut Reel', 27: '500 ft.', 12: '200 ft.', 2: '10 ft.', 
            11: '20 ft.', 25: '400 ft.', 15: '25 Ft.', 21: '3 ft.', 
            35: '8 ft.', 22: '3.50 ft.', 9: '15 ft.', 33: '750 ft.', 
            17: '250 ft.', 36: '8000 ft.', 32: '700 ft.', 8: '1400 ft.', 
            31: '6500 ft.', 28: '5000 ft.', 34: '7500 ft.', 18: '2500 ft.', 23: '3000 ft.', 29: '600 ft.', 19: '2700 ft.', 3: '10,000 ft.', 
            0: '0', 6: '118.5 in.', 7: '119 in.', 1: '10 Ft.', 20: '2730 ft.', 30: '6100 ft.', 10: '20 Ft.', 38: 'Multiple', 24: '350 ft.', 
            14: '225 ft.', 13: '2000 ft.'
            }
        for pred in available_lengths:
            if prediction == pred:
                length = available_lengths[pred]

        return print({"prediction": prediction, "label": length, "status": "OK"})

    def type_postprocessing(self, prediction):
        available_types = {
            15: 'LFMC', 16: 'LFNC', 4: 'EMT', 6: 'FMC', 11: 'GRC', 13: 'IMC', 20: 'PVCC', 21: 'RMC', 5: 'ENT', 7: 'FNC', 14: 'Innerduct', 
            19: 'PVC', 10: 'Flexible, Liquidtight', 8: 'Flexible', 1: 'ATLA', 18: 'Liquidtight Flexible', 0: '0', 9: 'Flexible Metallic', 
            17: 'LL', 23: 'Smoothwall', 12: 'HDPE Conduit', 2: 'Corrugated', 3: 'Corrugated HDPE', 22: 'Rise Raceway'
            }
        for pred in available_types:
            if prediction == pred:
                types = available_types[pred]

        return print({"prediction": prediction, "label": types, "status": "OK"})

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

RandomForest().compute_prediction(data = {
            "Short Desc": "Steel Conduit Hot-Dippedâ Galvanizedâ Steel (Inner Core); PVC (Liquidtightâ Jacket), 1/2 in.",
            "Long Desc": "1/2 in. PVC-coated galvanized steel type ATLA grey liquid-tight conduit. Conduit is 1000 ft."
        })