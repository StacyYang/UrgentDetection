import pandas as pd 
import torch

import transformers
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from flask import Flask,render_template,request


app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./urgent_bert")


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
	pred = get_pred(message)
	vect = 0
	#prediction = model.predict(vect)
	#output = prediction[0]
	if vect == 0:
		my_prediction = 0
	else:
		my_prediction = 1
	
	res = render_template('result.html', prediction=my_prediction)
	
	return res



if __name__ == '__main__':
	app.run(debug=True)