from flask import Flask, render_template, url_for,request
import pickle

#loading the model from disk
model = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("transform.pkl","rb"))

#create a flask object/ starting point
app = Flask(__name__)

#first page loading 
@app.route('/')
def home():
	return render_template("home.html")

#routing to /predict page to get the model and predict the output
@app.route('/predict',methods=["POST"])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data)
		my_pred = model.predict(vect)
	return render_template("result.html",prediction =my_pred)

if __name__ == '__main__':
	app.run(debug=True)