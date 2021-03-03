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
        return self.material_model.predict(data), 
                self.size_model.predict(data), 
                self.length_model.predict(data), 
                self.type_model.predict(data)

    def material_postprocessing(self, prediction):
        """
        Decode the encoded outputs for consumption by users
        """
        material = 0
        if prediction == 0:
            material = 0
        
        elif prediction == 1:
            material = "304 Stainless Steel"

        elif prediction == 2:
            material = "316 Stainless Steel"

        elif prediction == 3:
            material = "Aluminium"

        elif prediction == 4:
            material = "Aluminum"

        elif prediction == 5:
            material = "Carbon Strip Steel"

        elif prediction == 6:
            material = "Corrosion Resistant Plated Steel, PVC"
        
        elif prediction == 7:
            material = "Electroplated Steel"

        elif prediction == 8:
            material = "Galvanized Steel"

        elif prediction == 9:
            material = "Galvanized Steel, Thermoplastic PVC (Jacket)"

        elif prediction == 10:
            material = "HDPE"

        elif prediction == 11:
            material = "HDPE-Schedule 80"

        elif prediction == 12:
            material = "High Density Polyethylene"

        elif prediction == 13:
            material = "Hot Dipped Galvanized Steel, PVC"

        elif prediction == 14:
            material = "Hot Galvanized Steel"

        elif prediction == 15:
            material = "Hot-Dippedâ Galvanizedâ Steel (Inner Core); PVC"

        elif prediction == 16:
            material = "Non-Metallic"

        elif prediction == 17:
            material = "Non-Metallic/PVC"

        elif prediction == 18:
            material = Nylon

        elif prediction == 19:
            material = "Nylon Resin"

        elif prediction == 20:
            material = "Nylon, Polyamide"

        elif prediction == 21:
            material = "PVC"

        elif prediction == 22:
            material = "PVC (Jacket), Plated Steel (Inner Core)"

        elif prediction == 23:
            material = "PVC Coated Galvanized Steel"

        elif prediction == 24:
            material = "PVC-Coated Galvanized Steel"

        elif prediction == 25:
            material = "PVDF"

        elif prediction == 26:
            material = "Plenum-PVDF"

        elif prediction == 27:
            material = "Riser-Nylon"

        elif prediction == 28:
            material = "Riser-PVDF"

        elif prediction == 29:
            material = "Schedule 40"

        elif prediction == 30:
            material = "Schedule 40 PVC"

        elif prediction == 31:
            material = "Schedule 80"

        elif prediction == 32:
            material = "Schedule 80 PVC"

        elif prediction == 33:
            material = "Stainless Steel"

        elif prediction == 34:
            material = "Steel"

        elif prediction == 35:
            material = "stainless steel"

        return print({"prediction": prediction, "label": material, "status": "OK"})

    def size_postprocessing(self, prediction):
        size = 0
        if prediction == 0:
            size = 0

        if prediction == 1:
            size = "1 1/4 in."

        if prediction == 2:
            size = "1 in."

        if prediction == 3:
            size = "1-1/2 in."

        if prediction == 4:
            size = "1-1/4 in."

        if prediction == 5:
            size = "1/2 in."

        if prediction == 6:
            size = "2 in."

        if prediction == 7:
            size = "2-1/2 in."

        if prediction == 8:
            size = "21 mm"

        if prediction == 9:
            size = "3 in."

        if prediction == 10:
            size = "3-1/2 in."

        if prediction == 11:
            size = "3/4 in."

        if prediction ==12:
            size = "4"

        if prediction == 13:
            size = "4 in."

        if prediction ==14:
            size = "5 in."
        
        if prediction == 15:
            size = "6 in."

        return print({"prediction": prediction, "label": size, "status": "OK"})

    def length_postprocessing(self, prediction):
        length = 0
        if prediction == 0:
            length = "0"

        if prediction == 1:
            length = "10 Ft."

        if prediction == 2:
            length = "10 ft."

        if prediction == 3:
            length = "10,000 ft."

        if prediction == 4:
            length = "100 ft."

        if prediction == 5:
            length = "1000 ft."

        if prediction == 6:
            length = "118.5 in."

        if prediction == 7:
            length = "119 in."

        if prediction == 8:
            length = "1400 ft."

        if prediction == 9:
            length = "15 ft."

        if prediction == 10:
            length = "20 Ft."

        if prediction == 11:
            length = "20 ft."

        if prediction == 12:
            length = "200 ft."

        if prediction == 13:
            length = "2000 ft."

        if prediction == 14:
            length = "225 ft."

        if prediction == 15:
            length = "25 Ft."

        if prediction == 16:
            length = "25 ft."

        if prediction == 17:
            length = "250 ft."

        if prediction == 18:
            length = "2500 ft."

        if prediction == 19:
            length = "2700 ft."

        if prediction == 20:
            length = "2730 ft."

        if prediction == 21:
            length = "3 ft."

        if prediction == 22:
            length = "3.50 ft."

        if prediction == 23:
            length = "3000 ft."

        if prediction == 24:
            length = "350 ft."

        if prediction == 25:
            length = "400 ft."

        if prediction == 26:
            length = "50 ft."

        if prediction == 27:
            length = "500 ft."

        if prediction == 28:
            length = "5000 ft."

        if prediction == 29:
            length = "600 ft."

        if prediction == 30:
            length = "6100 ft."

        if prediction == 31:
            length = "6500 ft."

        if prediction == 32:
            length = "700 ft."

        if prediction == 33:
            length = "750 ft."

        if prediction == 34:
            length = "7500 ft."

        if prediction == 35:
            length = "8 ft."

        if prediction == 36:
            length = "8000 ft."

        if prediction == 37:
            length = "Cut Reel"

        if prediction == 38:
            length = "Multiple"

        return print({"prediction": prediction, "label": length, "status": "OK"})

    def type_postprocessing(self, prediction):
        types = 0
        if prediction == 0:
            types = 0

        if prediction == 1:
            types = "ATLA"

        if prediction == 2:
            types = "Corrugated"

        if prediction == 3:
            types = "Corrugated HDPE"

        if prediction == 4:
            types = "EMT"

        if prediction == 5:
            types = "ENT"

        if prediction == 6:
            types = "FMC"

        if prediction == 7:
            types = "FNC"

        if prediction == 8:
            types = "Flexible"

        if prediction == 9:
            types = "Flexible Metallic"

        if prediction == 10:
            types = "Flexible, Liquidtight"

        if prediction == 11:
            types = "GRC"

        if prediction == 12:
            types = "HDPE Conduit"

        if prediction == 13:
            types = "IMC"

        if prediction == 14:
            types = "Innerduct"

        if prediction == 15:
            types = "LFMC"

        if prediction == 16:
            types = "LFNC"

        if prediction == 17:
            types = "LL"

        if prediction == 18:
            types = "Liquidtight Flexible"

        if  prediction == 19:
            types = "PVC"

        if prediction == 20:
            types = "PVCC"

        if prediction == 21:
            types = "RMC"

        if prediction == 22:
            types = "Rise Raceway"

        if prediction == 23:
            types = "Smoothwall"

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

            material_predicted_output = self.material_postprocessing(
                                            material_predicted_output)
            size_predicted_output = self.size_postprocessing(
                                            size_predicted_output)
            length_predicted_output = self.length_postprocessing(
                                            length_predicted_output)
            type_predicted_output = self.type_postprocessing(
                                            type_predicted_output)

        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return material_predicted_output, 
                size_predicted_output, 
                length_predicted_output, 
                type_predicted_output
