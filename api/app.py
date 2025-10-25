from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'heart_model.h5')
model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))
    prediction = model.predict(img_array)[0][0]
    return jsonify({'risk_score': float(prediction)})
