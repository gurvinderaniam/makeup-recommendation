import cv2
import mediapipe as mp
import numpy as np

def extract_facial_features(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize Mediapipe Face Detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
    # Process the image
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None  # No face detected
    
    # Extract key facial features
    facial_features = {}
    for face_landmarks in results.multi_face_landmarks:
        # Extracting skin tone (average of central face region)
        h, w, _ = image.shape
        skin_region = [152, 9, 10, 234, 454]  # Sample facial landmark indices
        skin_tones = [image_rgb[int(face_landmarks.landmark[i].y * h)][int(face_landmarks.landmark[i].x * w)] for i in skin_region]
        avg_skin_tone = np.mean(skin_tones, axis=0)
        facial_features['skin_tone'] = avg_skin_tone.tolist()
        
        # Extracting face shape (relative distance of key points)
        jaw_points = [face_landmarks.landmark[i] for i in [152, 234, 454]]
        jaw_width = abs(jaw_points[1].x - jaw_points[2].x)
        jaw_length = abs(jaw_points[0].y - jaw_points[1].y)
        facial_features['face_shape_ratio'] = jaw_width / jaw_length
        
    return facial_features

# Test the function
features = extract_facial_features(image_path)
print(features)
