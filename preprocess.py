import json
import os
import math
import librosa

DATASET_PATH = "/media/mrifat/MRIFAT/MUSICAL_GENRE_CLASSIFIER/genres"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
        #"spectral_centroid": [],
        #"chromagram": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    #print(samples_per_segment)
    #print(SAMPLES_PER_TRACK)
    #print(num_segments)
    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T
                    #print(mfcc.shape)

                    #extract spectral centroid
                    #spectral_centroid = librosa.feature.spectral_centroid(y=signal[start:finish], sr=SAMPLE_RATE, n_fft=n_fft,
                                                #hop_length=hop_length)
                    #spectral_centroid = spectral_centroid.T

                    #extract chroma features
                    #chromagram = librosa.feature.chroma_stft(y=signal[start:finish], sr=SAMPLE_RATE, n_fft=n_fft,
                                                #hop_length=hop_length)
                    #chromagram = chromagram.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        #print(data["mfcc"])
                        #print(data["labels"])
                        #print(i)
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        #data["spectral_centroid"].append(spectral_centroid.tolist())
                        #data["chromagram"].append(chromagram.tolist())
                        #print("{}, segment:{}".format(file_path, d+1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)