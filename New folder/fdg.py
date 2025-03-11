import cv2
import dlib
import pandas as pd
import os

# Load Dlib’s face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/Pranshul Raghuvanshi/Desktop/terraform/shape_predictor_68_face_landmarks.dat")

# Folder where images are stored
image_folder = "ffhq_images"
output_data = []

# Function to extract face shape ratio
def get_face_shape(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        return None  # No face detected

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract specific landmarks for face width & height
        jaw_width = abs(landmarks.part(14).x - landmarks.part(2).x)
        face_height = abs(landmarks.part(27).y - landmarks.part(8).y)

        # Face ratio: width-to-height
        ratio = jaw_width / face_height

        # Debugging: Print jaw width, face height, and ratio
        print(f"📌 Image: {image_path}")
        print(f"Jaw Width: {jaw_width}, Face Height: {face_height}, Ratio: {ratio}")

        # Classify face shape with improved thresholds
        if ratio > 1.45:
            shape = "Oval"
        elif 1.25 <= ratio <= 1.45:
            shape = "Round"
        elif 1.05 <= ratio < 1.25:
            shape = "Heart"
        else:
            shape = "Square"

        return shape

# Process all images
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    face_shape = get_face_shape(image_path)
    
    if face_shape:
        output_data.append([image_file, face_shape])
        print(f"✅ Processed {image_file} - Face Shape: {face_shape}")

# Save results as CSV
df = pd.DataFrame(output_data, columns=["Image", "Face_Shape"])
df.to_csv("face_shape_data_fixed.csv", index=False)

print("✅ Face shape extraction complete! Data saved to face_shape_data_fixed.csv")
