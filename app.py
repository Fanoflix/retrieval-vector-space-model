from flask import Flask, render_template, url_for, redirect, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
import os, json
from nltk.stem import WordNetLemmatizer
import math

stopWords = ['"', '.', 'a', 'is', 'the', 'of', 'all', 'and', 'to', 'can', 'be', 'as', 'once', 'for', 'at', 'am', 'are', 'has', 'have', 'had', 'up', 'his', 'her', 'in', 'on', 'no', 'we', 'do']
qtfidf = {}
qtf = {}

#========= Loading Required Data ================================\\
with open('idf.json', 'r') as f: #                               ||
    idf = json.load(f) #                                         ||
#                                                                ||
with open('postingsList.json', 'r') as f: #                      ||
    postingsList = json.load(f) #                                ||
#                                                                ||
with open('magnitudes.json', 'r') as f: #                        ||
    magnitudes = json.load(f) #                                  ||
#                                                                ||
with open('tfidf.json', 'r') as f: #                             ||
    tfidf = json.load(f) #                                       ||
#================================================================//



def queryProcessing(query):
    global stopWords, qtfidf, qtf, idf, postingsList, magnitudes, tfidf
    lemmatizer = WordNetLemmatizer()
    queryWords = []
    answer = []

    # ============================== PROCESSING QUERY ==============================
    # ==========================================================================================

    query = query.split()
    
    # Lemmatizing the Query
    for word in query:
        queryWords.append(lemmatizer.lemmatize(word))

    # TF of query
    for word in queryWords:
        if word not in stopWords:
            if word in qtf:
                qtf[word] += 1
            else:
                qtf[word] = 1


    CalculateQtfIdf(qtf) # Stored in {qtfidf}
    
    # Writing Query tf-idf to query-tfidf.json (JUST FOR ASSIGNMENT PURPOSE, SO TEACHER CAN SEE, DONT INCLUDE THIS IN PROGRAM COMPLEXITY)
    # with open('./query-tfidf.json', 'w') as f:
    #     json.dump(qtfidf, fp=f, sort_keys=True)

    queryMagnitude = CalculateQueryMagnitute(qtfidf)

    # ============================== SEARCHING CORPUS ==============================
    # ==========================================================================================
    alpha = 0.005
    similarities = {}
    
    CalculateCosSims(queryMagnitude, similarities) # Stored in {similarities}

    # Writing the Cosine Similarities in similarities.json (JUST FOR ASSIGNMENT PURPOSE, SO TEACHER CAN SEE, DONT INCLUDE THIS IN PROGRAM COMPLEXITY)
    # with open('./similarities.json', 'w') as f:
    #     json.dump(similarities, fp=f, sort_keys=True)
    
    for item in similarities:
        if similarities[item] > alpha:
            answer.append(item)


    if answer is None:
        return "Word does not exist in the corpus"
    else:
        answer.sort(key = int)
        return answer


def CalculateQtfIdf(qtf):
    for word in postingsList:
        qtfidf[word] = 0
        if word in qtf:
            qtfidf[word] = (1 + math.log10(qtf.get(word))) * idf[word]


def CalculateQueryMagnitute(qtfidf):
    mag = 0
    for word in qtfidf:
        mag += qtfidf[word]**2
    queryMagnitude = math.sqrt(mag)
    return queryMagnitude


def CalculateCosSims(queryMagnitude, similarities):
    mul = 0
    for i in range(1,51):
        for word in tfidf:
            mul += tfidf[word][str(i)] * qtfidf[word]
        similarity = mul   /  (magnitudes[str(i)] * queryMagnitude)
        similarities[str(i)] = similarity
        mul = 0
        similarity = 0



class SearchForm(FlaskForm):
    query = StringField('Type query here')
    submit = SubmitField('Search')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
basedir = os.path.abspath(os.path.dirname(__file__))
app.debug = True

@app.route('/', methods=["GET", "POST"])
def searchPage():

    form = SearchForm()

    if form.submit.data:
        query = form.query.data.lower()
        
        # Processing Query
        answer = queryProcessing(query)


        return redirect(url_for('resultPage', answer=answer))
    return render_template('home.html', form=form)

@app.route('/result/<answer>', methods=['GET', 'POST'])
def resultPage(answer):
    return render_template('result.html', answer=answer)

if __name__ == '__main__':
    app.run()
