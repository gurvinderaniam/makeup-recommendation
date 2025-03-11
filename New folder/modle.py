import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

# Load Dataset
print("📌 Loading dataset...")
df = pd.read_csv("final_dataset.csv")
label_encoder = LabelEncoder()
df["Face_Shape"] = label_encoder.fit_transform(df["Face_Shape"])
# Prepare Data
X = df[['Skin_Tone_R', 'Skin_Tone_G', 'Skin_Tone_B', 'Face_Shape']]
y = df['Recommended_Makeup']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter Tuning
print("📌 Running Hyperparameter Tuning...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("✅ Best Parameters:", grid_search.best_params_)

# Train Best Model
print("📌 Training best model...")
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Model Evaluation
y_pred = best_model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Cross-Validation
cv_scores = cross_val_score(best_model, X, y, cv=10)
print("✅ Cross-Validation Score:", np.mean(cv_scores))

# SHAP Model Interpretability
print("📌 Generating SHAP Explanations...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
plt.show()

# Deep Learning Feature Extraction
print("📌 Extracting Deep Features...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))

data_gen = ImageDataGenerator(rescale=1./255)
train_gen = data_gen.flow_from_directory(r'C:\Users\Pranshul Raghuvanshi\Desktop\terraform\flaskfolder\ffhq_images', target_size=(224, 224), batch_size=32, class_mode=None)

features = model.predict(train_gen)
np.save("deep_features.npy", features)
print("✅ Deep Features Extracted!")

# Save Model
joblib.dump(best_model, "makeup_recommendation_model_v2.pkl")
print("✅ Model saved as makeup_recommendation_model_v2.pkl")