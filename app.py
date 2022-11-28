import pandas as pd 
import torch

import transformers
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline

from flask import Flask, render_template, request


app = Flask(__name__)


bert_tokenizer = BertTokenizer.from_pretrained("./bert_base_model")
bert_model = BertForSequenceClassification.from_pretrained("./bert_base_model")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_base_model")
distilbert_model = DistilBertForSequenceClassification.from_pretrained("./distilbert_base_model")


@app.route('/', methods=['POST','GET'])
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

    output = classifier(message)
    label = output[0]["label"]

    res = render_template('result.html', prediction=label, 
                        message=message, model=selected_model)
    return res


if __name__ == '__main__':
	app.run(debug=True)