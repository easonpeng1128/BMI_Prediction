#!/usr/bin/env python
# coding: utf-8

# In[ ]:


filename = '/home/appuser/venv/lib/python3.9/site-packages/keras_vggface/models.py' 
text = open(filename).read() 
open(filename, 'w+').write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))


# In[3]:


#!pip install streamlit


# In[11]:


#!pip install streamlit-webrtc


# In[2]:


import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import pandas as pd 
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


# In[3]:


import av
from turn import get_ice_servers
import threading



# In[4]:


@st.cache_resource(show_spinner=False)
def preprocessing_v2(img):
    
    sample = img.copy()
    sample = cv2.resize(sample, (224, 224))
    sample = np.array(sample).astype(np.float64)
    sample = np.expand_dims(sample, axis = 0)
    sample = preprocess_input(sample, version = 2)
    
    return sample


# In[5]:


# Load VGGFace Model
@st.cache_resource(show_spinner=False)
def load_vggface():
    vgg_senet = VGGFace(model='senet50')
    vgg_senet_avg_pool = Model(inputs=vgg_senet.input, outputs=vgg_senet.get_layer('avg_pool').output)
    return vgg_senet_avg_pool


# In[6]:


# Load the Regressor
@st.cache_resource(show_spinner=False)
def load_regressor():
    senet_rbf_model = joblib.load('senet_SVR_rbf_model.pkl')
    return senet_rbf_model


# In[7]:


# Initialize Models
vgg_senet_avg_pool = load_vggface()
senet_rbf_model = load_regressor()


# In[8]:


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


# In[9]:


font = cv2.FONT_HERSHEY_SIMPLEX


# In[10]:


def predict_bmi(frame):
    pred_bmi = []

    faces = faceCascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor = 1.15,
            minNeighbors = 5,
            minSize = (30,30),
            )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        processed_img = preprocessing_v2(frame[y:y+h, x:x+w])
        features = vgg_senet_avg_pool.predict(processed_img)
        # Flatten the features for SVR
        features_flatten = features.reshape(features.shape[0], -1)
        preds = senet_rbf_model.predict(features_flatten)
        pred_bmi.append(preds[0])
        cv2.putText(frame, f'BMI: {preds}', (x+5, y-5), font, 1, (255, 255, 255), 2)

    return pred_bmi, frame


# In[11]:


class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.out_image = None
        self.pred_bmi = []

    def recv(self, frame):
        frm = frame.to_ndarray(format='bgr24')
        pred_bmi, frame_with_bmi = predict_bmi(frm)
        with self.frame_lock:
            self.out_image = frame_with_bmi
            self.pred_bmi = pred_bmi

        return av.VideoFrame.from_ndarray(frame_with_bmi, format='bgr24') 


# In[12]:


def bmi_interpretation(bmi):
    if bmi<18.5:
        st.write('**Seems like you are a bit UNDERWEIGHT. Make sure to have balanced meals!🍽️**')
    elif 18.5<=bmi<=25:
        st.write('**Great job! Your BMI is in the healthy range! Keep it up!👍**')
    elif 25<bmi<30:
        st.write('**Looks like you are OVERWEIGHT. Try to balance your meals and exercise regularly.🏋️**')
    elif 30<=bmi<35:
        st.write('**You are in the MODERATELY OBESE category. It is important to eat healthily and move more.🚴**')
    elif 35<=bmi<=40:
        st.write('**It appears you are SEVERELY OBESE. It is crucial to focus on your diet and physical activity. 🧘**')
    elif bmi>40:
        st.write('**You are in the VERY SEVERELY OBESE category. It might be helpful to seek medical advice on your health.👨‍⚕️**')



# In[ ]:





# In[ ]:


# UI design
st.markdown("<h1 style='text-align: center; color: #800020;'>Predict Your BMI Live</h1>", unsafe_allow_html=True)
ctx = webrtc_streamer(key="example", video_transformer_factory=VideoProcessor, sendback_audio=False) # Add a title and a brief introduction.
st.title("Live BMI Prediction")
st.write("""
Welcome to the live BMI prediction app! Here, you can use your webcam to capture an image and the application will predict your Body Mass Index (BMI). 
The BMI is a measurement that uses your height and weight to work out if your weight is in a healthy range. It's a useful measure of overweight and obesity.
""")

# Add some details about the BMI.
st.header("Understanding Your BMI")
st.write("""
The Body Mass Index (BMI) is a measurement of a person's weight with respect to their height. It is calculated by dividing the person's weight by the square of their height.
""")

# Display a table with BMI ranges.
st.header("BMI Ranges")
bmi_ranges = pd.DataFrame({
  'Category': ['Underweight', 'Normal weight', 'Overweight', 'Obesity Class I', 'Obesity Class II', 'Obesity Class III'],
  'BMI Range': ['< 18.5', '18.5 - 24.9', '25 - 29.9', '30 - 34.9', '35 - 39.9', '>= 40']
})
st.table(bmi_ranges)

# Add more explanation about the predictions and how to interpret them.
st.header("How to Interpret Your Results")
st.write("""
Once you capture your image, the application will provide a predicted BMI. You can refer to the BMI ranges table above to understand what your result means.
Please note that while BMI is a useful measurement for most people, it does have some limitations. For example, it may overestimate body fat in athletes and others who have a muscular build.
""")

# Add a call-to-action encouraging users to start the application.
st.header("Get Started")
st.write("To get started, simply click on the 'Snapshot' button when you are ready.")

# Input fields for user to input their height and weight
weight = st.number_input("Enter your weight (in kg)", min_value=0.0, max_value=300.0, step=0.1)
height = st.number_input("Enter your height (in cm)", min_value=0.0, max_value=300.0, step=0.1)

# Calculate BMI button
if st.button('Calculate BMI'):
    # Calculate BMI
    height_m = height / 100.0  # Convert height from cm to meters
    bmi = weight / (height_m * height_m)
    
    # Display BMI and interpretation
    st.write(f'Your BMI is {bmi:.2f}')
    bmi_interpretation(bmi)


# In[ ]:





# In[ ]:




