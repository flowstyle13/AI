#Library imports
import pickle
import streamlit as st
import numpy as np

CLASS_NAMES=["non diabitique"," diabitique"]

#Setting Title of App
st.title("Diabities predections")
p = st.text_input("Pregnancies")
a = st.text_input("Glucose")
b = st.text_input("BloodPressure")
c = st.text_input("SkinThickness")
d = st.text_input("Insulin")
e = st.text_input("BMI")
f = st.text_input("DiabetesPedigreeFunction")
j = st.text_input("Age")
submit = st.button('Predict')
loaded_model = pickle.load(open('knnpickle_file', 'rb'))
if submit:

        # Convert the file to an opencv image.
        
        Y_pred = loaded_model.predict([[float(p),float(a),float(b),float(c),float(d),float(e),float(f),float(j)]])


        st.title(str(CLASS_NAMES[np.argmax(Y_pred)]))
        


