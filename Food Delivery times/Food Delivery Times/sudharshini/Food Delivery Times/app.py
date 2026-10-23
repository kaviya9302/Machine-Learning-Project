from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("food_delivery_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form
        distance = float(request.form["distance"])
        preparation_time = float(
            request.form["preparation_time"]
        )
        courier_experience = float(
            request.form["courier_experience"]
        )

        # Basic input validation
        if distance <= 0:
            return render_template(
                "index.html",
                error="Distance must be greater than zero."
            )

        if preparation_time < 0 or courier_experience < 0:
            return render_template(
                "index.html",
                error="Preparation time and experience cannot be negative."
            )

        # Create input DataFrame with correct feature names
        input_data = pd.DataFrame(
            [[distance, preparation_time, courier_experience]],
            columns=[
                "Distance_km",
                "Preparation_Time_min",
                "Courier_Experience_yrs"
            ]
        )

        # Predict delivery time
        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction=round(float(prediction), 2),
            distance=distance,
            preparation_time=preparation_time,
            courier_experience=courier_experience
        )

    except (ValueError, TypeError):
        return render_template(
            "index.html",
            error="Please enter valid numeric values."
        )


if __name__ == "__main__":
    app.run(debug=True)