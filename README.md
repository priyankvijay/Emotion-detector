# Emotion-detector
Real-Time Emotion Detector

A Python application that uses "OpenCV" and "DeepFace" to track your face and detect your expressions live through your webcam.

Features

* Live Tracking: Uses OpenCV Haar Cascades to lock onto your face smoothly.
* AI Expression Analysis: Powered by DeepFace and TensorFlow to catch emotions like happy, sad, angry, and neutral.
* Selfie Mirror View: Flips the video horizontally so it feels like a natural front camera.
* Lag-Free Performance: Runs the heavy AI analysis in a background thread so the video feed stays smooth.

Installation
Run this command in your terminal to grab all the required dependencies:

pip install opencv-python deepface tensorflow tf-keras setuptools

How to Run: 

1. Make sure your webcam is plugged in and working.
2. Run the script:
   python emotion.py
3. Press 'q' on your keyboard anytime to close the window.
