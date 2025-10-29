from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            total_bill = float(request.form["total_bill"])
            size = int(request.form["size"])
            smoker = request.form["smoker"]

            is_smoker = 1 if smoker.lower() == "yes" else 0
            features = np.array([[total_bill, size, is_smoker]])
            prediction = model.predict(features)[0]

            # Store results temporarily in query parameters
            return redirect(
                url_for(
                    "home",
                    prediction=f"💰 Predicted Tip: Rs. {prediction:.2f}",
                    total_bill=total_bill,
                    size=size,
                    smoker=smoker,
                )
            )
        except Exception as e:
            return redirect(url_for("home", prediction=f"Error: {str(e)}"))

    # Handle GET request (page load)
    prediction_text = request.args.get("prediction", "")
    total_bill = request.args.get("total_bill", "")
    size = request.args.get("size", "")
    smoker = request.args.get("smoker", "")

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        total_bill=total_bill,
        size=size,
        smoker=smoker,
    )

if __name__ == "__main__":
    app.run(debug=True)
