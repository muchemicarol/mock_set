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
        self.material_model = joblib.load("/home/wambui/Fiverr/Python/notebooks/sanveohr/material_random_forest.joblib")
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

            # print(set(short_long_desc_df.columns.tolist()) - set(empty_dataframe.columns.tolist()))
            short_long_desc_df = short_long_desc_df.drop(
                columns=list(set(short_long_desc_df.columns.tolist()) - set(empty_dataframe.columns.tolist()))
                )
            # print(short_long_desc_df) 

            # print(short_long_desc_df)
            
            empty_dataframe.drop(columns=["Unnamed: 0"], inplace=True)
            # print(empty_dataframe.head())
            short_long_desc_df = short_long_desc_df.loc[~short_long_desc_df.index.duplicated(keep="first")]
            empty_dataframe = empty_dataframe.loc[~empty_dataframe.index.duplicated(keep="first")]


            input_data = empty_dataframe.append(short_long_desc_df)
            # print(empty_dataframe)

            input_data.fillna(value=0, inplace=True)

        except Exception as e:
            print(e)
            # return {"status": "Error", "message": str(e)}

        return input_data

    def predict(self, data):
        return self.material_model.predict(data)

    def postprocessing(self, prediction):
        """
        Decode the encoded outputs for consumption by users
        """
        label = 200
        if prediction == 0:
            label = 0
        
        elif prediction == 1:
            label = "304 Stainless Steel"

        elif prediction == 2:
            label = "316 Stainless Steel"

        elif prediction == 3:
            label = "Aluminium"

        elif prediction == 4:
            label = "Aluminum"

        elif prediction == 5:
            label = "Carbon Strip Steel"

        elif prediction == 6:
            label = "Corrosion Resistant Plated Steel, PVC"
        
        elif prediction == 7:
            label = "Electroplated Steel"

        elif prediction == 8:
            label = "Galvanized Steel"

        elif prediction == 9:
            label = "Galvanized Steel, Thermoplastic PVC (Jacket)"

        elif prediction == 10:
            label = "HDPE"

        elif prediction == 11:
            label = "HDPE-Schedule 80"

        elif prediction == 12:
            label = "High Density Polyethylene"

        elif prediction == 13:
            label = "Hot Dipped Galvanized Steel, PVC"

        elif prediction == 14:
            label = "Hot Galvanized Steel"

        elif prediction == 15:
            label = "Hot-Dippedâ Galvanizedâ Steel (Inner Core); PVC"

        elif prediction == 16:
            label = "Non-Metallic"

        elif prediction == 17:
            label = "Non-Metallic/PVC"

        elif prediction == 18:
            label = Nylon

        elif prediction == 19:
            label = "Nylon Resin"

        elif prediction == 20:
            label = "Nylon, Polyamide"

        elif prediction == 21:
            label = "PVC"

        elif prediction == 22:
            label = "PVC (Jacket), Plated Steel (Inner Core)"

        elif prediction == 23:
            label = "PVC Coated Galvanized Steel"

        elif prediction == 24:
            label = "PVC-Coated Galvanized Steel"

        elif prediction == 25:
            label = "PVDF"

        elif prediction == 26:
            label = "Plenum-PVDF"

        elif prediction == 27:
            label = "Riser-Nylon"

        elif prediction == 28:
            label = "Riser-PVDF"

        elif prediction == 29:
            label = "Schedule 40"

        elif prediction == 30:
            label = "Schedule 40 PVC"

        elif prediction == 31:
            label = "Schedule 80"

        elif prediction == 32:
            label = "Schedule 80 PVC"

        elif prediction == 33:
            label = "Stainless Steel"

        elif prediction == 34:
            label = "Steel"

        elif prediction == 35:
            label = "stainless steel"

        return {"prediction": prediction, "label": label, "status": "OK"}


    def compute_prediction(self, data):
        try:
            data = self.preprocessing(data)
            print(data)
            predicted_output = self.predict(data)
            predicted_output = self.postprocessing(predicted_output)

        except Exception as e:
            print(e)
            # return print({"status": "Error", "message": str(e)})

        return predicted_output

rf = RandomForest()
rf.compute_prediction(data = {
        "Short Desc": "Steel Conduit Hot-Dippedâ Galvanizedâ Steel (Inner Core); PVC (Liquidtightâ Jacket), 1/2 in.",
        "Long Desc": "1/2 in. PVC-coated galvanized steel type ATLA grey liquid-tight conduit. Conduit is 1000 ft.",
        })
# rf = RandomForest()
# rf.postprocessing(prediction = {
#             "Short Desc": "Metallic Liquidtight Conduit, Flexible, LA, Galvanized Steel (Inner Core), PVC (Jacket)",
#             "Long Desc": "_x005F_x000D_Type LA_x005F_x000D_",
#         })