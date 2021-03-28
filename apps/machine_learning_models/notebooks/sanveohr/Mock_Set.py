#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import string

import joblib
import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_rows", 3000)


# In[65]:


filename = input("Enter the file name you wish to train: ")
sheetname = input("Enter sheet name from file: ")
mock_data = pd.read_excel(filename, sheet_name=sheetname)
print("\nThe file has {} rows and {} columns" .format(mock_data.shape[0], mock_data.shape[1]))


# In[3]:


mock_data.head(2)


# In[4]:


mock_data = mock_data[["Short Desc", "Long Desc", "Size", "Length", "Material", "Type"]]
mock_data.head(2)


# In[5]:


# rename the columns 
mock_data.columns = mock_data.columns.str.lower().str.replace(" ", "_")
mock_data.columns


# ## Clean up the data

# In[6]:


# Check for missing values
mock_data.isnull().sum()


# In[7]:


# if 


# ## Preprocessing Data, Feature Selection and Model

# In[8]:


labelencoder = LabelEncoder()

for column in ["size", "length", "material", "type"]:
    mock_data[column] = mock_data[column].astype(str)
    mock_data[column + "_"] = labelencoder.fit_transform(mock_data[column])
    
mock_data.head(2)


# In[9]:


mock_data.sort_values(by="type_")[["type", "type_"]]


# In[10]:


type_ = str(dict(zip(mock_data["type_"], mock_data["type"]))).replace("\'", "\"")
material = str(dict(zip(mock_data["material_"], mock_data["material"]))).replace("\'", "\"")
size = str(dict(zip(mock_data["size_"], mock_data["size"]))).replace("\'", "\"")
length = str(dict(zip(mock_data["length_"], mock_data["length"]))).replace("\'", "\"")

filenames = ["type_", "material", "size", "length"]


for filename, content in zip(filenames, [type_, material, size, length]):
    with open("{}.txt".format(filename), "w") as f:
        f.write(content)


# In[11]:


mock_data[mock_data["short_desc"] == "Non-Metallic Gray Liquidtight Flexible Conduit, 1-1/4 in."]


# In[12]:


type_


# In[13]:


mock_data[mock_data["type_"] == 0]


# In[14]:


mock_data["type"].unique()


# In[15]:


mock_data["material"].unique()


# In[16]:


size


# In[17]:


stopwords = nltk.corpus.stopwords.words("english")
word_net = nltk.WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = str(text).replace("\n", " ")
    text = "".join(word for word in str(text) if word not in string.punctuation)
    tokens = re.split("\W+", text)
    lemmatized = [word_net.lemmatize(word) for word in tokens if word not in stopwords]
    return lemmatized


# In[18]:


count_vector = CountVectorizer(analyzer=clean_text)
count_vector_ = CountVectorizer(analyzer=clean_text)

vector = count_vector.fit_transform(mock_data["short_desc"])
vector_ = count_vector_.fit_transform(mock_data["long_desc"])


# In[19]:


short_desc_df = pd.DataFrame(vector.todense(), columns=count_vector.get_feature_names())
long_desc_df = pd.DataFrame(vector_.toarray(), columns=count_vector_.get_feature_names())


# In[20]:


short_desc_df.head(2)


# In[21]:


short_long_desc_df = pd.concat([short_desc_df, long_desc_df], axis=1)
short_long_desc_df = short_long_desc_df.groupby(short_long_desc_df.columns, axis=1).sum()

independent_variables = short_long_desc_df.columns


# In[22]:


material_df = pd.concat([short_long_desc_df, mock_data["material_"]], axis=1)
size_df = pd.concat([short_long_desc_df, mock_data["size_"]], axis=1)
length_df = pd.concat([short_long_desc_df, mock_data["length_"]], axis=1)
type_df = pd.concat([short_long_desc_df, mock_data["type_"]], axis=1)


# In[23]:


material_df.shape, size_df.shape, type_df.shape, length_df.shape


# In[24]:


X = material_df[independent_variables]
y = material_df["material_"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, )


# In[25]:


X_train = X_train.groupby(X_train.columns, axis=1).sum()
X_test = X_test.groupby(X_test.columns, axis=1).sum()


# In[26]:


X_train.shape, X_test.shape


# In[27]:


def mae(y_true, y_pred):
    # mean absolute error
    return np.mean(abs(y_true - y_pred))

def training_and_evaluate(model):
    model.fit(X_train, y_train)
    
    model_pred = model.predict(X_test)
    
    model_mae = mae(y_test, model_pred)
    
    return model_mae


# In[28]:


rf = RandomForestClassifier()
dt = DecisionTreeClassifier()


# In[60]:


print("==========================================\n")
print("          Mean Absolute Errors:")
print("    (lower error => better performance)")
print("\n==========================================")


# In[29]:


def modelling(df, target_variable, models):
    print(target_variable)
    X = df[independent_variables]
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    for model in models:
        
        model.fit(X_train, y_train)
        model_pred = model.predict(X_test)
        model_mae = np.mean(abs(y_test - model_pred))

        print("{} Mean Absolute Error: {}".format(model, model_mae))
    print("\n")


# In[30]:


for df, column in zip([material_df, size_df, length_df, type_df], ["material_", "size_", "length_", "type_"]):
    modelling(df, column, [rf, dt])


# In[58]:


print("==========================")
print("     Accuracy Scores:")
print("==========================")


# In[31]:


X = material_df[independent_variables]
X = X.groupby(X.columns, axis=1).sum()

y = material_df["material_"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[32]:


material_rf = rf.fit(X_train, y_train)
material_dt = dt.fit(X_train, y_train)


# In[33]:


df = pd.DataFrame(columns=X_train.columns)
df.to_excel("empty_dataframe.xlsx")


# In[34]:


material_rf_pred = material_rf.predict(X_test)
print("material: {}".format(accuracy_score(y_test, material_rf_pred)))


# In[35]:


joblib.dump(material_rf, "./material_random_forest.joblib", compress=True)
joblib.dump(material_dt, "./material_decision_trees.joblib", compress=True)


# In[36]:


X = length_df[independent_variables]
y = length_df["length_"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[37]:


length_rf = rf.fit(X_train, y_train)
length_dt = dt.fit(X_train, y_train)


# In[38]:


length_rf_pred = length_rf.predict(X_test)
print("length: {}".format(accuracy_score(y_test, length_rf_pred)))


# In[39]:


joblib.dump(length_rf, "./length_random_forest.joblib", compress=True)
joblib.dump(length_dt, "./length_decision_trees.joblib", compress=True)


# In[40]:


X = size_df[independent_variables]
y = size_df["size_"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[41]:


size_rf = rf.fit(X_train, y_train)
size_dt = dt.fit(X_train, y_train)


# In[42]:


size_rf_pred = size_rf.predict(X_test)
print("size: {}".format(accuracy_score(y_test, size_rf_pred)))


# In[43]:


joblib.dump(size_rf, "./size_random_forest.joblib", compress=True)
joblib.dump(size_dt, "./size_decision_trees.joblib", compress=True)


# In[44]:


X = type_df[independent_variables]
y = type_df["type_"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[45]:


type_rf = rf.fit(X_train, y_train)
type_dt = dt.fit(X_train, y_train)


# In[46]:


type_rf_pred = type_rf.predict(X_test)
print("type: {}".format(accuracy_score(y_test, type_rf_pred)))


# In[47]:


joblib.dump(type_rf, "./type_random_forest.joblib", compress=True)
joblib.dump(type_dt, "./type_decision_trees.joblib", compress=True)

