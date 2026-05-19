import streamlit as st
import joblib as jb

# Load the model from the file
model = jb.load('iris_model.pkl')

st.title('Iris Flower Prediction')

# Create input fields for the user to enter the features
sepal_length = st.number_input('Sepal Length')
sepal_width = st.number_input('Sepal Width')
petal_length = st.number_input('Petal Length')
petal_width = st.number_input('Petal Width')
# Create a button to make the prediction

if st.button('Predict'):
    # Create a feature array from the user input
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    # Make the prediction using the loaded model
    prediction = model.predict(features)
    # Display the predicted class
    st.write(f'Predicted class: {prediction[0]}')

    # 4.9, 3.0, 1.4,0.2