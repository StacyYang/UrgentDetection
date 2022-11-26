import pandas as pd 
import torch

import transformers
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import pipeline

from flask import Flask,render_template,request


app = Flask(__name__)
bert_tokenizer = BertTokenizer.from_pretrained("./bert_base_model")
bert_model = BertForSequenceClassification.from_pretrained("./bert_base_model")


def get_pred(sentences):
	encoded_input = tokenizer([sentences], padding=True, truncation=True, max_length=128, return_tensors='pt')

	with torch.no_grad():
		labels = torch.tensor([1]).unsqueeze(0)
		model_output = model(**encoded_input)

		logits = model_output[1]
		_, preds = torch.max(model_output[1], dim=1)
		predictions = predictions.extend(preds)
		predictions = torch.stack(predictions)
		print("This is the prediction: ", predictions)
	return predictions

	

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	message = request.form['message']
	bert_classifier = pipeline(task="text-classification", 
								model=bert_model,
								tokenizer=bert_tokenizer)	
	output = bert_classifier(message)
	label = output[0]["label"]

	#if label == "LABEL_1":
	#	outcome = "Urgent Message"
	#else:
	#	outcome = "Non-urgent Message"
	
	res = render_template('result.html', prediction=label)
	
	return res


if __name__ == '__main__':
	app.run(debug=True)