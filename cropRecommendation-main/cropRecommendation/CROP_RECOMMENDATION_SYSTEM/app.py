import streamlit as st
import numpy as np
import io
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from lime import lime_tabular

def predict():
   

    label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/standard_scaler.pkl', 'rb'))
    model = pickle.load(open('models/classifier.pkl', 'rb'))


    st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:center;"> SmartCrop: Intelligent Crop Recommendation 🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    N = st.number_input("Nitrogen", 1, 10000)
    P = st.number_input("Phosporus", 1, 10000)
    K = st.number_input("Potassium", 1, 10000)
    temperature = st.number_input("Temperature", 0.0, 100000.0)
    humidity = st.number_input("Humidity in %", 0.0, 100000.0)
    ph = st.number_input("Ph", 0.0, 100000.0)
    rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)

    feature_list = [N, P, K, temperature, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, 7)

    df = pd.DataFrame(data=single_pred, columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

    scaled_df = standard_scaler.transform(df)

    ok = st.button('Predict')

    if ok:
        prediction = model.predict(scaled_df)[0]
        st.write(prediction)

    # Create Lime explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=standard_scaler.transform(scaled_df),
        feature_names=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        class_names=label_encoder.classes_,
        mode='classification'
    )

    exp = explainer.explain_instance(
        data_row=scaled_df[0],
        predict_fn=model.predict_proba
    )

    exp_figure = exp.as_pyplot_figure()

    # Increase the image size
    exp_figure.set_figwidth(13)  # Adjust the width as needed
    exp_figure.set_figheight(8)  # Adjust the height as needed

    exp_figure_bytes = io.BytesIO()
    exp_figure.savefig(exp_figure_bytes, format='png')
    exp_figure_bytes.seek(0)

    st.image(exp_figure_bytes, caption='LIME Interpretation', use_column_width=True)

predict()
