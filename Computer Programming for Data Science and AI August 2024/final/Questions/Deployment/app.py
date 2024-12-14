from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import pickle
import joblib
import numpy
import random

app = Flask(__name__, template_folder='templates')


model = joblib.load('model.pkl')


@ app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

@ app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        features = [x for x in request.form.values()]
    print(features)
    # labels = model.predict([features])
    # species = labels[0]
    species = random.randint(0, 1)
    if species == 0:
        s = "Diabet is No - 0"
    else:
        s = "Diabet is Yes - 1"
    return s

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

