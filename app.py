from flask import Flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open("models/titanic_model.pkl", "rb"))
sc = joblib.load(open("models/titanic_scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("formulario.html")


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]

    pclass = np.zeros(3, dtype=int)
    pclass[int_features[0] - 1] = 1

    sex = np.zeros(2, dtype=int)
    sex[int_features[1] - 1] = 1

    final_features = [np.array([*int_features[2:], *sex, *pclass])]
    
    sclaed_features = sc.transform(final_features)
    prediction = model.predict(sclaed_features)

    output = round(prediction[0], 2)
    print(output)
    if output == 0:
        sobrevive = "NO"
    else:
        sobrevive = "SI"
    return render_template(
        "formulario.html",
        prediction_text="El usuario {} va a sobrevivir".format(sobrevive),
    )


if __name__ == "__main__":
    app.run(debug=True)

# @app.route("/")
# def hello_world():
# return "<p>Hello, World! cargo modelo</p>"
