import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import streamlit as st
import time

# Load the saved model from file
model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define the alphabet (1-9 and A-Z)
alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# Initialize variables
sentence = ""
last_detection_time = 0
detection_interval = 3  # Time interval in seconds to detect gestures
last_letter = None
consecutive_count = 0
required_consecutive = 2  # Number of consecutive detections required to add a letter

# Functions to calculate landmarks and normalize
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    return list(map(lambda n: n / max_value, temp_landmark_list))

# Check if both hands show the flat hand gesture (all fingers extended)
def is_flat_hand(landmark_list):
    # Define finger tips and bases
    finger_tips = [4, 8, 12, 16, 20]
    finger_bases = [2, 5, 9, 13, 17]
    
    # Check if all finger tips are above their bases (for a vertical hand)
    return all(landmark_list[tip][1] < landmark_list[base][1] for tip, base in zip(finger_tips, finger_bases))

# Streamlit application setup
st.title("Sign Language to Text Converter")
st.write("This app detects hand gestures and converts them into sentences.")

# Placeholder for real-time video display
video_placeholder = st.empty()
sentence_placeholder = st.empty()  # Placeholder for sentence updates

# For webcam input
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,  # Allow up to 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            st.warning("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Process the image
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_time = time.time()

        # Only process if hand landmarks are detected and it's time for a new detection
        if results.multi_hand_landmarks and (current_time - last_detection_time >= detection_interval):
            flat_hands_count = 0
            detected_alphabet = None

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Convert to DataFrame for model prediction
                df = pd.DataFrame(pre_processed_landmark_list).transpose()
                predictions = model.predict(df, verbose=0)
                predicted_class = np.argmax(predictions)
                detected_alphabet = alphabet[predicted_class]

                # Check for flat hand gesture
                if is_flat_hand(landmark_list):
                    flat_hands_count += 1

            # Update the sentence based on the detection
            if flat_hands_count == 2:
                if last_letter != ' ':
                    sentence += " "
                    last_letter = ' '
                    consecutive_count = 0
            elif detected_alphabet:
                if detected_alphabet == last_letter:
                    consecutive_count += 1
                    if consecutive_count >= required_consecutive:
                        sentence += detected_alphabet
                        consecutive_count = 0
                else:
                    last_letter = detected_alphabet
                    consecutive_count = 1

            # Update the last detection time
            last_detection_time = current_time

            # Display the updated sentence
            sentence_placeholder.write(f"Detected Sentence: {sentence}")

        # Display the video with annotations
        video_placeholder.image(image, channels="BGR")

        # Break the loop if 'ESC' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()