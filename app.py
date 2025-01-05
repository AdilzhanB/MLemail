from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

with open("models/cv_new.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/clf_new.pkl", "wb") as f:
    pickle.dump(model, f)

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        email_text = request.form.get('email-content')
    tokenized_text = tokenizer.transform([email_text])
    prediction = model.predict(tokenized_text)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction_response=prediction, email_text=email_text)
if __name__ == "__main__":
    app.run(debug=True)