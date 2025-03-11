from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("makeup_recommendation_model.pkl")

# Create Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract input values
        skin_tone_r = data["Skin_Tone_R"]
        skin_tone_g = data["Skin_Tone_G"]
        skin_tone_b = data["Skin_Tone_B"]
        face_shape = data["Face_Shape"]

        # Convert face shape to numeric encoding
        face_shape_map = {"Oval": 0, "Round": 1, "Heart": 2, "Square": 3}
        face_shape_encoded = face_shape_map.get(face_shape, -1)
        
        if face_shape_encoded == -1:
            return jsonify({"error": "Invalid face shape"}), 400

        # Make prediction
        features = pd.DataFrame([[skin_tone_r, skin_tone_g, skin_tone_b, face_shape_encoded]],
                        columns=["Skin_Tone_R", "Skin_Tone_G", "Skin_Tone_B", "Face_Shape"])
        prediction = model.predict(features)[0]

        # Define makeup recommendations based on prediction
        makeup_map = {
            0: "Neutral & Warm Shades",
            1: "Bright Colors & Contouring",
            2: "Soft Pinks & Mauves",
            3: "Bold & Defined Colors"
        }
        recommendation = makeup_map.get(prediction, "No Recommendation Found")

        return jsonify({"Recommended_Makeup": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
