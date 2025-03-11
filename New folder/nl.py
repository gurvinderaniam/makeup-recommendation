import pickle

# Load the model
with open("makeup_recommendation_model_v2.pkl", "rb") as file:
    model = pickle.load(file)

# Print model details
print(model)
