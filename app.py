from flask import Flask, render_template, request, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("iris_model.pkl")  

# Mapping of numeric labels to flower names
label_map = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

# Map flower names to images
# Mapping flower names to image files in static folder
image_map = {
    "Iris-setosa": "setosa.png",
    "Iris-versicolor": "Iris-versicolor.png",
    "Iris-virginica": "Iris-virginica.png"
}


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]

        # Convert numeric prediction to flower name
        flower_name = label_map[int(prediction)]
        flower_image = url_for('static', filename=image_map[flower_name])  # ✅ correct image path

        return render_template("index.html", 
                               result=f"Predicted Flower: {flower_name}",
                               image_url=flower_image)

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
