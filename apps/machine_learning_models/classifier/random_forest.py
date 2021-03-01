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
        self.material_model = joblib.load(path + "material_random_forest.joblib")
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

            short_desc_df = pd.DataFrame(vector.todense(), columns=count_vector.get_feature_names())
            long_desc_df = pd.DataFrame(vector_.todense(), columns=count_vector_.get_feature_names())


            short_long_desc_df = pd.concat([short_desc_df, long_desc_df], axis=1)
            short_long_desc_df = short_long_desc_df.groupby(short_long_desc_df.columns, axis=1).sum()


            # print(set(short_long_desc_df.columns.tolist()) - set(empty_dataframe.columns.tolist()))
            short_long_desc_df = short_long_desc_df.drop(
                columns=list(set(short_long_desc_df.columns.tolist()) - set(empty_dataframe.columns.tolist()))
                )

            empty_dataframe.drop(columns=["Unnamed: 0"], inplace=True)


            input_data = empty_dataframe.append(short_long_desc_df)


            input_data.fillna(value=0, inplace=True)

        except:
            raise Exception
            # return {"status": "Error", "message": str(e)}

        return input_data

    def predict(self, data):
        """
        Prediction on input data based on trained models
        """
        return self.material_model.predict(data)

    def postprocessing(self, prediction):
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

        return {"prediction": prediction, "label": material, "status": "OK"}


    def compute_prediction(self, data):
        """
        Apply preprocessing, predictiona and post processing of data
        """
        try:
            data = self.preprocessing(data)
            predicted_output = self.predict(data)
            predicted_output = self.postprocessing(predicted_output)

        except Exception as e:
            return print({"status": "Error", "message": str(e)})

        return predicted_output

rf = RandomForest()
rf.compute_prediction(data = {
        "Short Desc": 
        "PVDF Resin Plenum Innerduct, 3/4 in.",
        "Long Desc": 
        "Product Overview:_x005F_x000D_3/4 in. PVDF Resin Plenum innerduct with 900 lb. pull line, white, the length is 3000 ft._x005F_x000D_ENDOCOR'S corrugated design provides high tensile strength with low weight per foot for ease of handling and significantly longer put ups that can be obtained with smoothwall or ribbed innerduct. PVDF resin plenum.",
        })
# rf = RandomForest()
# rf.postprocessing(prediction = {
#             "Short Desc": "Metallic Liquidtight Conduit, Flexible, LA, Galvanized Steel (Inner Core), PVC (Jacket)",
#             "Long Desc": "_x005F_x000D_Type LA_x005F_x000D_",
#         })