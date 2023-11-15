from flask import Flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    print(output)
    if output == 0:
        sobrevive = "NO"
    else:
        sobrevive = "SI" 
    return render_template('formulario.html', prediction_text='El usuario {} va a sobrevivir'.format(sobrevive))

if __name__ == "__main__":
    app.run(debug=True)