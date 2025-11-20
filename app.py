from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

# Absolute path fix
base = os.path.dirname(os.path.abspath(__file__))

tfidf = pickle.load(open(os.path.join(base, "model/tfidf.pkl"), "rb"))
model = pickle.load(open(os.path.join(base, "model/cyber_model.pkl"), "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_text = request.form["text"]

        vector = tfidf.transform([user_text])
        prediction = model.predict(vector)[0]

        return render_template("result.html", text=user_text, prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
