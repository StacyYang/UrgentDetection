import librosa
import numpy as np
import os 
import pandas as pd 
import pickle
import torch

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import ErnieConfig, ErnieForSequenceClassification
from transformers import pipeline

from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_EXTENSIONS'] = ['.wav']
app.config['UPLOAD_PATH'] = 'uploads'

# Bert 
bert_tokenizer = BertTokenizer.from_pretrained("./bert_base_model")
bert_model = BertForSequenceClassification.from_pretrained("./bert_base_model")
# Distil Bert Base 
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_base_model")
distilbert_model = DistilBertForSequenceClassification.from_pretrained("./distilbert_base_model")
# Distil Bert Amazon Reviews
dbert2_tokenizer = DistilBertTokenizer.from_pretrained("./reviews_sentiment_distilbert_model")
dbert2_model = DistilBertForSequenceClassification.from_pretrained("./reviews_sentiment_distilbert_model")
# Ernie 2.0 EN
ernie_tokenizer = BertTokenizer.from_pretrained("./ernie_2_base_model")
ernie_model = ErnieForSequenceClassification.from_pretrained("./ernie_2_base_model")

def extract_features(signal, sr):
    # Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.
    rms = librosa.feature.rms(signal + 0.0001)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # mel frequnecy cepstral coefficients
    # mfcc = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T, axis=0)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.
    stft = np.abs(librosa.stft(signal))

    pitches, magnitudes = librosa.piptrack(signal, sr=sr, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])
    # print('pitch: ', pitch)
    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    flatness = np.mean(librosa.feature.spectral_flatness(y=signal))

    cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
    centroid = cent / np.sum(cent)
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    ext_features = np.array([
        rms_mean, rms_std, pitch_tuning_offset, pitch_mean, pitch_std,
        flatness, centroid_mean, centroid_std])

    extract_features = np.concatenate((ext_features, mfccs, delta_mfccs, delta2_mfccs))

    return extract_features


@app.route('/')
def home():
    return render_template('home.html')

# Accepting File Submissions & Securing file uploads
@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    selected_model = request.form['model']
    classifier = None
    
    if selected_model == "BERT":
        classifier = pipeline(task="text-classification", 
                            model=bert_model,
                            tokenizer=bert_tokenizer)
    elif selected_model == "DISTILBERT":
        classifier = pipeline(task="text-classification",
                            model=distilbert_model,
                            tokenizer=distilbert_tokenizer)
    elif selected_model == "DISTILBERT-REVIEWS":
        classifier = pipeline(task="text-classification",
                            model=dbert2_model,
                            tokenizer=dbert2_tokenizer)    
    else: #if selected_model == "ERNIE"
    	classifier = pipeline(task="text-classification",
    		                model=ernie_model,
    		                tokenizer=ernie_tokenizer)

    output = classifier(message)
    label = output[0]["label"]

    res = render_template('crisis_notify.html', prediction=label, 
                        message=message, model=selected_model)
    return res

@app.route('/reveal', methods=['GET', 'POST'])
def reveal():
    columns = ['rms_mean', 'rms_std', 'pitch_tuning_offset', 'pitch_mean', 'pitch_std', 'flatness', 'centroid_mean', 'centroid_std']
    columns_plus = columns + [(lambda x: "mfcc_" + str(x))(x) for x in range(1, 14)] + [(lambda x: "delta_mfccs_" + str(x))(x) for x in range(1, 14)] + [(lambda x: "delta2_mfccs_" + str(x))(x) for x in range(1, 14)]
    df= pd.DataFrame(columns = columns_plus)

    audio_file = request.files['file']
    signal, sr = librosa.load(audio_file)
    features_all = list(extract_features(signal, sr))
    df = df.append(pd.DataFrame(features_all, index=columns_plus).transpose(), ignore_index=True)

    model_type = request.form['model']

    if model_type == "LOGISTIC REGRESSION":
        classifier = pickle.load(open("./saved_models/logistic_regression.pkl", "rb"))
    elif model_type == "MULTINOMIAL NB":
        classifier = pickle.load(open("./saved_models/MultinomialNB.pkl", "rb"))
    else:
        classifier = pickle.load(open("./saved_models/randomforest.pkl", "rb"))

    speech_prediction = classifier.predict(df)[0]

    
    return render_template('emotion_display.html', audiofile=audio_file, 
                    model=model_type, prediction=speech_prediction, )

if __name__ == '__main__':
	app.run(debug=True)