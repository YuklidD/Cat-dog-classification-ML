# prompt: Write python code for make visual application use with the trained model

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

st.title('Dog or Cat Classifier')

model = tf.keras.models.load_model('dog_cat_classifier.h5')

class_names = ['Cat', 'Dog']

def predict(image):
    img = cv2.imread(image)
    resized_image = tf.image.resize(img, (128, 128))
    scaled_image = resized_image/255
    yhat = model.predict(np.expand_dims(scaled_image, 0))
    return yhat

def main():
    st.sidebar.title('Upload an Image')
    image = st.sidebar.file_uploader('Upload Images', type=['jpg', 'png'])
    if image is not None:
        st.image(image)
        yhat = predict(image)
        st.write(f'The image is a {class_names[np.argmax(yhat)]}')

if __name__ == '__main__':
    main()
