import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv("final_dataset.csv")

# Encode categorical face shapes
df["Face_Shape"] = df["Face_Shape"].astype("category").cat.codes  
df["Recommended_Makeup"] = df["Recommended_Makeup"].astype("category").cat.codes  # Encode target labels

# Define input (X) and output (y)
X = df[["Skin_Tone_R", "Skin_Tone_G", "Skin_Tone_B", "Face_Shape"]]  # Features
y = df["Recommended_Makeup"]  # Labels (Makeup Recommendation)

# Split dataset into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Dataset Loaded: {df.shape[0]} samples")

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained with accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "makeup_recommendation_model.pkl")

print("✅ Model saved as makeup_recommendation_model.pkl")