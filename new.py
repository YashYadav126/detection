import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Load Haarcascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces in an image
def detect_faces(image):
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    draw = ImageDraw.Draw(image)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

    return image

# Streamlit UI
st.title("📷 Real-Time Object Detection")

# Webcam input using Streamlit's built-in feature
st.subheader("Take a Photo")
uploaded_image = st.camera_input("Capture Image")

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Captured Image", use_container_width=True)

    # Detect objects in the image
    st.subheader("Detected Objects")
    processed_image = detect_faces(image)
    st.image(processed_image, caption="Processed Image", use_container_width=True)

