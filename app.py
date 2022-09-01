import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

def main():
    from PIL import Image
    #image_hospital = Image.open('Neuro1.png')
    #image_ban = Image.open('Neuro2.png')
    #st.image(image_ban, use_column_width=False)
    #st.sidebar.image(image_hospital)
if __name__ == '__main__':
    main()


st.write("""
# Random forest for predicting intracranial injury of pediatric traumatic brain injury (for unseen data)

""")
st.write ("Tunthanathip et al.")

#st.write("""
### Performances of various algorithms from the training dataset [Link](https://pedtbi-train.herokuapp.com/)
#""")

#st.write ("""
### Labels of input features
#1.GCSer (Glasgow Coma Scale score at ER): range 3-15

#2.Hypotension (History of hypotension episode): 0=no , 1=yes

#3.pupilBE (pupillary light refelx at ER): 0=fixed both eyes, 1= fixed one eye, 2=React both eyes

#4.SAH (Subarachnoid hemorrhage on CT of the brain): 0=no, 1=yes

#""")


st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/Thara-PSU/CT_pedTBI/blob/main/example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if  uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Agemo = st.sidebar.slider('Age(months)', 1, 180, 60)
        Sex = st.sidebar.slider('Sex (1=male, 2=female)', 1, 2, 1)
        Road_taff = st.sidebar.slider('Road_traffic_injury (0=no, 1=yes)', 0, 1, 0)
        LOC = st.sidebar.slider('Loss_of_consciousness (0=no, 1=yes)', 0, 1, 0)
        Amnesia = st.sidebar.slider('Amnesia (0=no, 1=yes)', 0, 1, 0)
        NV = st.sidebar.slider('Vomiting(0=no, 1=yes)', 0, 1, 0)
        Motorweak = st.sidebar.slider('Hemiparesis (0=no, 1=yes)', 0, 1, 0)
        Scalp = st.sidebar.slider('Scalp_injury (0=no, 1=yes)', 0, 1, 0)
        Basesign = st.sidebar.slider('Bleeding_per_nose/ear (0=no, 1=yes)', 0, 1, 0)
        Hypotension = st.sidebar.slider('Hypotension (0=no, 1=yes)', 0, 1, 0)
        Bradycardia = st.sidebar.slider('Bradycardia (0=no, 1=yes)', 0, 1, 0)
        Seizure = st.sidebar.slider('Seizure (0=no, 1=yes)', 0, 1, 0)
        GCSer = st.sidebar.slider('Glasgow_Coma_Scale_at_emergency_depratment', 3, 15, 15)
        pupilBE = st.sidebar.slider('Pupillary_light_reflex (0=fixed BE, 1=react one eye, 2=react both eyes)', 0, 2, 2)
        # Hypotension = st.sidebar.selectbox('Hypotension',( '0', '1'))
        # pupilBE = st.sidebar.selectbox('pupilBE', ('0', '1','2','3'))
        data = {'Agemo': Agemo,
                'Sex': Sex,
                'Road_taff': Road_taff,
                'LOC': LOC,
                'NV': NV,
                'Motorweak': Motorweak,
                'Scalp': Scalp,
                'Basesign': Basesign,
                'Hypotension': Hypotension,
                'Bradycardia': Bradycardia,
                'Seizure': Seizure,
                'Amnesia': Amnesia,
                'GCSer': GCSer,
                'pupilBE': pupilBE,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
GBM_raw = pd.read_csv('train.2020.csv')
GBM = GBM_raw.drop(columns=['CTpositive'])
df = pd.concat([input_df,GBM],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['pupilBE']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)


# Reads in saved classification model
load_clf = pickle.load(open('pedtbi_rf_clf.pkl', 'rb'))
 

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.write("""# Prediction Probability""")
#st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Class Labels and their corresponding index number')

label_name = np.array(['Negative','Positive'])
st.write(label_name)
# labels -dictionary
names ={0:'Negative',
1: 'Positive'}

st.write("""# Prediction""")
#st.subheader('Prediction')
two_year_survival = np.array(['Negative','Positive'])
st.write(two_year_survival[prediction])

st.write("""# Prediction is positive when probability of the class 1 is more than 0.5""")


st.write ("""
### Other Algorithms for Predicting Intracranial Injury of Pediatric Traumatic Brain Injury 

""")

st.markdown( "  [Random forest] (https://ct-pedtbi-test-rf.herokuapp.com/) ")
st.markdown( "  [Logistic Regression] (https://ct-pedtbi-test-ln.herokuapp.com/) ")
st.markdown( "  [Neural Network] (https://ct-pedtbi-test-nn.herokuapp.com/) ")
#st.markdown( "  [K-Nearest Neighbor (kNN)] (https://pedtbi-test-knn.herokuapp.com/) ")
st.markdown( "  [naive Bayes] (https://ct-pedtbi-test-nb.herokuapp.com/) ")
st.markdown( "  [Support Vector Machines] (https://ct-pedtbi-test-svm.herokuapp.com/) ")
#st.markdown( "  [Gradient Boosting Classifier] (https://pedtbi-test-gbc.herokuapp.com/) ")
st.markdown( "  [Nomogram] (https://psuneurosx.shinyapps.io/ct-pedtbi-nomogram/) ")

st.write ("""
### [Home](https://ct-pedtbi-home.herokuapp.com/)

""")