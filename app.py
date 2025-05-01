from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model('sentiment_bilstm_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 150
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['review'].strip()
    if not text:
        return render_template('index.html', error="Please enter a review.")

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    probs = model.predict(padded)[0]
    label_index = np.argmax(probs)
    sentiment = label_map[label_index]
    confidence = float(probs[label_index])

    return render_template(
        'index.html',
        review=text,
        sentiment=sentiment,
        score=round(confidence, 4)
    )

if __name__ == '__main__':
    app.run(debug=True)