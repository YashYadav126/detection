import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Load OpenCV's built-in Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

# Function to detect objects in images
def detect_objects(image, detection_type):
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    draw = ImageDraw.Draw(image)

    if detection_type == "Faces, Eyes & Smiles":
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            roi_gray = gray[y:y + h, x:x + w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            for (ex, ey, ew, eh) in eyes:
                draw.rectangle([x + ex, y + ey, x + ex + ew, y + ey + eh], outline="blue", width=2)

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            for (sx, sy, sw, sh) in smiles:
                draw.rectangle([x + sx, y + sy, x + sx + sw, y + sy + sh], outline="green", width=2)

    elif detection_type == "Full Body":
        bodies = full_body_cascade.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in bodies:
            draw.rectangle([x, y, x + w, y + h], outline="purple", width=3)

    elif detection_type == "Upper Body":
        upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in upper_bodies:
            draw.rectangle([x, y, x + w, y + h], outline="orange", width=3)

    return image

# Function to process video
def detect_objects_in_video(detection_type):
    cap = cv2.VideoCapture(0)  # 0 for webcam
    
    stframe = st.empty()  # Placeholder for the video feed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if detection_type == "Faces, Eyes & Smiles":
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)

        elif detection_type == "Full Body":
            bodies = full_body_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 0, 128), 2)

        elif detection_type == "Upper Body":
            upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in upper_bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 140, 0), 2)

        # Convert frame to RGB and display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()

# Streamlit UI
st.title("🔍 Multi-Object Detection (Images & Live Video)")

# Select Image or Video Mode
mode = st.radio("Choose Mode", ["Image Upload", "Live Video"])

# Select Object Type to Detect (Removed Cars Option)
detection_type = st.selectbox("Select Object to Detect", ["Faces, Eyes & Smiles", "Full Body", "Upper Body"])

# Image Upload Mode
if mode == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button(f"Detect {detection_type}"):
            processed_image = detect_objects(image, detection_type)
            st.image(processed_image, caption=f"Detected {detection_type}", use_container_width=True)

# Live Video Mode
elif mode == "Live Video":
    if st.button(f"Start Video Detection for {detection_type}"):
        detect_objects_in_video(detection_type)
