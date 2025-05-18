#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
import os
import pickle
import time
import sys

print("Starting face recognition for attendance...")
curd = os.getcwd()

# Function to load the model
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to extract face features (same as in train.py)
def extract_face_features(image, face_location):
    """
    Extract simple features from a face region
    Returns a feature vector
    """
    # Extract face region
    top, right, bottom, left = face_location
    face_img = image[top:bottom, left:right]
    
    # Resize to a standard size
    face_img = cv2.resize(face_img, (50, 50))
    
    # Convert to grayscale
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Histogram of pixel values as a simple feature
    hist = cv2.calcHist([face_img], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Add some basic statistics as features
    mean = np.mean(face_img)
    std = np.std(face_img)
    
    # Combine all features
    features = np.concatenate((hist, [mean, std]))
    
    return features

# Function to detect faces using OpenCV
def detect_faces(image):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained face detector
    face_cascade_path = os.path.join(curd, "py", "haarcascade_frontalface_default.xml")
    if not os.path.exists(face_cascade_path):
        print(f"Error: Could not find cascade file at {face_cascade_path}")
        return []
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Convert to format (top, right, bottom, left)
    face_locations = []
    for (x, y, w, h) in faces:
        face_locations.append((y, x+w, y+h, x))
    
    return face_locations

# Function to recognize faces
def recognize_faces(image, model, distance_threshold=0.5):
    face_locations = detect_faces(image)
    
    if not face_locations:
        return []
    
    predictions = []
    for face_loc in face_locations:
        features = extract_face_features(image, face_loc)
        
        # Get prediction from model
        closest_distances = model.kneighbors([features], n_neighbors=1)
        is_match = closest_distances[0][0][0] <= distance_threshold
        
        if is_match:
            person_name = model.predict([features])[0]
        else:
            person_name = "unknown"
        
        predictions.append((person_name, face_loc))
    
    return predictions

# Start headless webcam capture for attendance
def take_attendance():
    model_path = os.path.join(curd, "assets", "models", "trained_knn_model.clf")
    model = load_model(model_path)
    
    if model is None:
        print("Could not load model. Please train the model first.")
        return
    
    detected_names = []
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    print("Starting webcam for attendance. Will capture for 30 seconds.")
    print("Keep your face in front of the camera.")
    
    # Capture for 30 seconds
    end_time = time.time() + 30
    frame_count = 0
    
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue
            
        # Recognize faces
        predictions = recognize_faces(frame, model)
        
        # Add names to attendance list
        for name, _ in predictions:
            if name != "unknown":
                detected_names.append(name)
                print(f"Detected: {name}")
        
        # Small pause
        time.sleep(0.2)
    
    # Cleanup
    cap.release()
    
    print(f"Attendance capture complete. Processed {frame_count} frames.")
    
    # Create attendance record
    if detected_names:
        # Count occurrences of each name
        name_counts = pd.Series(detected_names).value_counts()
        
        # Create attendance DataFrame
        attendance = pd.DataFrame(name_counts)
        attendance.columns = ['Count']
        
        # Mark as present if detected more than once
        attendance['Present'] = 0
        for i in range(len(attendance)):
            if attendance['Count'][i] > 1:
                attendance['Present'][i] = 1
        
        # Save to CSV
        attendance_final = attendance.drop(['Count'], axis=1)
        attendance_final.to_csv(os.path.join(curd, 'Attendance.csv'))
        
        print("\nAttendance Summary:")
        print(attendance)
        print("\nAttendance saved to Attendance.csv")
    else:
        print("No students detected for attendance.")

if __name__ == "__main__":
    take_attendance()
