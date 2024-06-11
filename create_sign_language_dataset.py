# create_sign_language_dataset_manual.py

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Create DataFrame to store hand landmarks and labels
df = pd.DataFrame(columns=['landmarks', 'label'])

# Mapping of keys to gestures (change as needed)
key_mapping = {
    ord('a'): 'ghe ghalun',
    ord('b'): 'bts ke chode',
    ord('c'): 'Ram Ram bhaiyon',
    ord('d'): 'Oo maa goo turu lobe'
}

# Collect data for each sign gesture
current_gesture = None

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Convert landmarks to list
                landmarks_list = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
                landmarks_flat = np.array(landmarks_list).flatten()

                # Display gesture label instructions
                cv2.putText(frame, 'Press a, b, c, d to record gestures', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Check if a key is pressed and record gesture
                key = cv2.waitKey(1) & 0xFF
                if key in key_mapping:
                    current_gesture = key_mapping[key]
                    print(f"Recording gesture {current_gesture}")

                # If a gesture is being recorded, save landmarks
                if current_gesture:
                    # Add landmarks and label to DataFrame
                    df.loc[len(df)] = [landmarks_flat, current_gesture]
                    cv2.putText(frame, f"Recording gesture: {current_gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Sign Language Dataset Creator', frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Save DataFrame to CSV file
df.to_csv('sign_language_dataset_manual.csv', index=False)

print("Dataset creation completed. CSV file saved as 'sign_language_dataset_manual.csv'.")
