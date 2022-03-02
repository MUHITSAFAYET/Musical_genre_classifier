import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import math
import einops
import librosa
from flask import Flask, request, jsonify
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

saved_model = keras.models.load_model("h5file/genre_classifier.h5")
#PREDICTION_PATH = "/media/mrifat/MRIFAT/MUSICAL_GENRE_CLASSIFIER/test/hiphop.00000.wav"
SAMPLE_RATE = 22050
def collect_mfcc(audio, sr):

    TRACK_DURATION = 30  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

    samples_per_segment = int(SAMPLES_PER_TRACK / 10)

    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / 512)

    for d in range(10):
        start = samples_per_segment * d
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=SAMPLE_RATE, n_mfcc=13, n_fft=2048,
                                                hop_length=512)

    mfcc = mfcc.T

    mfcc=mfcc.reshape(mfcc.shape[0], mfcc.shape[1],1)

    mfcc = einops.rearrange(mfcc, 'h w c -> c h w')

    return mfcc

def predict(X):
    predictions=saved_model.predict(X)
    predictions = tf.nn.softmax(predictions)
    pred0=predictions[0]
    label0=np.argmax(pred0)
    return label0


#collected=collect_mfcc(audio=PREDICTION_PATH)

#predicted=predict(X=collected)

#print(predicted)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            audio_bytes = file.read()
            signal, sample_rate = librosa.load(PREDICTION_PATH, sr=SAMPLE_RATE)
            mfcc = collect_mfcc(audio=signal, sr=SAMPLE_RATE)
            predicted = predict(X=mfcc)
            data = {"prediction": int(predicted)}
            return jsonify(data)

        except Exception as e:
            return jsonify({"error": str(e)})


    return "OK"

if __name__ == "main":
    app.run(debug=True)