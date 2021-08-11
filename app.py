from flask import Flask, render_template, request
import os
import pandas as pd
import pickle

app = Flask(__name__)
@app.route('/')
def welcome():
    return render_template("index.html")

@app.route('/verdict', methods=['GET', 'POST'])
def classifyArticle():
    # Read Test Data into a Dataframe
    articleInfo = {
        'title':request.form['title'],
        'text':request.form['text']
    }
    article_df = pd.DataFrame(articleInfo, index=[0])
    # Load ML Dependencies
    root = os.path.dirname(os.path.abspath(__file__)) 
    cv_file_path = os.path.join(root,'static/machineLearning/cv.sav')
    nb_file_path = os.path.join(root,'static/machineLearning/nb_model.sav')
    countVectorizer = pickle.load(open(cv_file_path,'rb'))
    naiveBayes = pickle.load(open(nb_file_path,'rb'))
    # Predict Outcome
    cv_text = countVectorizer.transform(article_df['text'])
    outcome = naiveBayes.predict(cv_text)[0]
    # Return Result
    if (outcome == 0):
        classification = 'fact'
    elif (outcome == 1):
        classification = 'fake'
    return render_template("results.html",results=classification)

if __name__ == "__main__":
    app.run(debug=True)