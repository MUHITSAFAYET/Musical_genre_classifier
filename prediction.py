import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import math
import einops
import librosa

"""prediction step for genre classifier"""

#provide test sample parameters
PREDICTION_PATH = "/media/mrifat/MRIFAT/MUSICAL_GENRE_CLASSIFIER/test/hiphop.00000.wav"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

#load the saved h5 model
saved_model = keras.models.load_model("h5file/genre_classifier.h5")

#load the test-sample
signal, sample_rate = librosa.load(PREDICTION_PATH , sr=SAMPLE_RATE)

#prepares the test-sample for testing
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

#predict probability values for all classes
predictions=saved_model.predict(mfcc)

#select the one having highest probability
predictions = tf.nn.softmax(predictions)
pred0=predictions[0]

#show prediction
label0=np.argmax(pred0)
print(label0)