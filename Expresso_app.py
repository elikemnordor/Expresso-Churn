import streamlit as st
import pandas as pd
import pickle
import numpy as np

#Picture of logo
from PIL import Image
img = Image.open("logo2.jpg")
st.image(img)


def main():

    #Sidebar Widgets
    st.sidebar.header('User Input features')

    #HOme Page
    st.title("Customer Churn Predictor")
    st.write(""" 
    Expresso is an African telecommunications company that provides customers with airtime and mobile data bundles.
    The objective of this machine learning built app to predict the likelihood an Expresso customer “churning,” (inactive/not transactioning for 90 days.)
    Data obtained from the [Zindi Africa](https://zindi.africa/hackathons/umojahack-ghana-expresso-churn-prediction-challenge/data).
    """)

    #uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    #if uploaded_file is not None:
    #    input_df = pd.read_csv(uploaded_file)
    #else:

    st.write('  ')



    st.write('  ')
    #Sidebar Widgets
    def user_input_features():
        REGION = st.sidebar.selectbox('REGION', ('DAKAR','DIOURBEL','FATICK','KAFFRINE','KAOLACK','KEDOUGOU','KOLDA','LOUGA','MATAM','SAINT-LOUIS','SEDHIOU','TAMBACOUNDA','THIES','ZIGUINCHOR'))      
        TENURE = st.sidebar.selectbox('TENURE',('D 3-6 month',' E 6-9 month','F 9-12 month','G 12-15 month','H 15-18 month','I 18-21 month','J 21-24 month','K > 24 month'))
        MONTANT = st.sidebar.slider('MONTANT', 10, 470000, 50)
        FREQUENCE_RECH = st.sidebar.slider('FREQUENCE_RECH', 1, 133, 65)
        REVENUE = st.sidebar.slider('REVENUE', 1, 532177, 2000)
        ARPU_SEGMENT = st.sidebar.slider('ARPU_SEGMENT', 0, 177392, 52000)
        FREQUENCE = st.sidebar.slider('FREQUENCE', 1, 91, 30)
        DATA_VOLUME = st.sidebar.slider('DATA_VOLUME', 0, 1823866, 50000)
        ON_NET = st.sidebar.slider('ON_NET', 0, 50809, 670)
        ORANGE = st.sidebar.slider('ORANGE', 0, 21323, 500)
        TIGO = st.sidebar.slider('TIGO', 0, 4174, 1000)
        ZONE1 = st.sidebar.slider('ZONE1', 0, 4792, 1200)
        ZONE2 = st.sidebar.slider('ZONE2', 0, 3697, 290)
        REGULARITY = st.sidebar.slider('REGULARITY', 1, 62, 55)
        FREQ_TOP_PACK = st.sidebar.slider('FREQ_TOP_PACK', 1, 713, 13)
        data = {
                    'REGION': REGION,
                    'TENURE': TENURE,
                    'MONTANT': MONTANT,
                    'FREQUENCE_RECH': FREQUENCE_RECH,
                    
                    'REVENUE': REVENUE,
                    'ARPU_SEGMENT': ARPU_SEGMENT,
                    'FREQUENCE': FREQUENCE,
                    'DATA_VOLUME': DATA_VOLUME,
                    'ON_NET': ON_NET,
                    'ORANGE': ORANGE,
                    'TIGO': TIGO,
                    'ZONE1': ZONE1,
                    'ZONE2': ZONE2,
                    'REGULARITY': REGULARITY,
                    'FREQ_TOP_PACK': FREQ_TOP_PACK,
                    
                        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # This will be useful for the encoding phase
    expresso_raw = pd.read_csv('expresso_example2.csv')
    expresso = expresso_raw.drop(columns=['CHURN'])
    df = pd.concat([input_df,expresso],axis=0)

    # Encoding of ordinal features

    encode = ['REGION','TENURE']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df,dummy], axis=1)
        del df[col]
    df = df[:1] # Selects only the first row (the user input data)

    df







    #Loading MOdel

    load_clf = pickle.load(open('expresso2.pkl', 'rb'))
    #xgb_model = pickle.load(open(file_name, "rb"))
    #expresso_train = pd.read_csv('expresso_train.csv')
    #x = expresso_train.drop('CHURN',1)
    #y =expresso_train.CHURN




    # Apply model to make predictions
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)
    
    #prediction_proba = mod.predict_proba(features)

    good_proba = prediction_proba[:,0]*100
    if prediction == 0:
        st.success("This Customer will stay with Expresso! ")
    else:
        st.warning('Sorry,this Customer has Churned')


    st.subheader('Prediction')
    prediction
    st.subheader('Prediction Probalility')
    prediction_proba
    st.text('-Model Logloss: 0.25911')



      #----------------------------   





    st.write('                 ')
    #Reading coordinates csv
    plotData = pd.read_csv("coordinates2.csv")
    Data = pd.DataFrame()
    Data['lat'] = plotData['lat']
    Data['lon'] = plotData['lon']

    st.write('                 ')
    st.write('                 ') 
    # Visualization Section


    st.title('Available Regions in the Dataset')
    #PLotting MAP
    st.map(plotData, zoom = 6)

    st.write( 
       '''

    Expresso is an African telecommunications services company that provides telecommunication services in two African markets: Mauritania and Senegal. Expresso offers a wide range of products and services to meet the needs of customers.subheader


    '''
        )


if __name__ == '__main__':
    main()