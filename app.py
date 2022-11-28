import pandas as pd 
import torch

import transformers
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import ErnieConfig, ErnieForSequenceClassification
from transformers import pipeline

from flask import Flask, render_template, request


app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('home.html')

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

    res = render_template('result.html', prediction=label, 
                        message=message, model=selected_model)
    return res


if __name__ == '__main__':
	app.run(debug=True)