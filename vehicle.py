import cv2
import numpy as np
import streamlit as st
import time

# Load Haar Cascade classifier from the specified path
vehicle_cascade_path = r"C:\Users\COMD\Desktop\Chatbot\cars.xml"
vehicle_cascade = cv2.CascadeClassifier(vehicle_cascade_path)

# Check if classifier was loaded properly
if vehicle_cascade.empty():
    st.error("Haar Cascade classifier for vehicle detection could not be loaded.")
else:
    st.success("Haar Cascade classifier loaded successfully.")

# Streamlit app
st.title("🚗 Vehicle Detection and Counting (Real-Time)")

# File uploader for image or video
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

def detect_vehicles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vehicles = vehicle_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, len(vehicles)

if uploaded_file is not None:
    if uploaded_file.type.startswith("video"):
        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Open the video
        cap = cv2.VideoCapture("temp_video.mp4")

        # Get the video frames
        stframe = st.empty()  # Placeholder for real-time video display
        total_vehicle_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame for vehicle detection
            processed_frame, vehicle_count = detect_vehicles(frame)
            total_vehicle_count += vehicle_count

            # Convert the frame from BGR (OpenCV format) to RGB (Streamlit format)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Display the processed frame in Streamlit
            stframe.image(processed_frame_rgb, caption="Vehicle Detection in Progress", use_column_width=True)

            # Add a small delay to simulate real-time video
            time.sleep(0.1)

        cap.release()

        # Display total vehicle count at the end of the video
        st.write(f"Total vehicles detected in the video: {total_vehicle_count}")

    else:
        st.write("Please upload a video file to start vehicle detection.")
