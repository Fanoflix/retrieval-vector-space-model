from os import listdir
from nltk.stem import WordNetLemmatizer
import json, re
import math
lemmatizer = WordNetLemmatizer()

# Postings List Schema

# postingsList = {
# "word1" : {
#       "docID1" : tf,
#       "docID2" : tf,
#       "docID3" : tf,
#       "docID4" : tf
#  },
# "word2" : {
#       "docID1" : tf,
#       "docID2" : tf,
#       "docID3" : tf
#  },
# ...
# ...

docFreq = {}
idf = {}
docLength = {}
tfidf = {}


stopWords = ['"', '.', 'a', 'is', 'the', 'of', 'all', 'and', 'to', 'can', 'be', 'as', 'once', 'for', 'at', 'am', 'are', 'has', 'have', 'had', 'up', 'his', 'her', 'in', 'on', 'no', 'we', 'do']

files = listdir('ShortStories') # Names of all files
postingsList = {}
for file in files:
    with open(f'./ShortStories/{file}', 'r', encoding='utf8') as f:
        # File as a string and splitting each substring on spaces and special characters e.g: line break, encoded quotations etc.
        string = re.split(' |-|\n|\u00e3|\u2019|\u201c|\u201d|\u2014|\u2018|\u00a9|\u00af|\u00aa|\u00b4|\u00a7|\u00a8', f.read())

    
        for word in string:
            word = word.lower()
            # Stripping word and removing " ' : ; - _ # + @ ( ) / ? ~ ` [ ] { } =
            word = word.strip(',|!|.|"|;|:|-|_|#|+|@|)|(|/|?|~|`|[|]|{|}|=|\u00e3') 

            if word not in stopWords:

                # Lemmatizing
                word = lemmatizer.lemmatize(word)

                # Finding Doc Lenghts and storing in {docLength}
                if file.split('.')[0] in docLength:
                    docLength[file.split('.')[0]] += 1
                else:
                    docLength[file.split('.')[0]] = 1

                # Finding Doc Frequency and storing in {docFreq}
                if word not in docFreq:
                    docFreq[word] = 0

                # Making Posting List (Refer to Postings list schema at the top)
                if word in postingsList:
                    if file.split('.')[0] in postingsList[word]:
                        postingsList[word][file.split('.')[0]] += 1
                    else:
                        postingsList[word][file.split('.')[0]] = 1
                    
                else:
                    postingsList[word] = {file.split('.')[0] : 1}



# Calculating DFs and storing in dictionary
for word in docFreq:
    docFreq[word] = len(postingsList[word])

# Caluclating IDFs and storing in dictionary
for word in docFreq:
    idf[word] = math.log10(50/docFreq[word])

# Popping artefacts
docFreq.pop("")
postingsList.pop("")
idf.pop("")


# tfidf schema
# {
#   word1: {
#           doc1: tf1-idf1,
#           doc2: tf1-idf2
#           ...,
#           doc50: tf1-idf50;
#         },
#   word2: {
#           doc1: tf2-idf1,
#           doc2: tf2-idf2
#           ...,
#           doc50: tf2-idf50;
#         },
# }

# Calculating tf-idf = (1+log(tf)) * (idf[word])
for word in postingsList:
    tfidf[word] = {str(key) : 0 for key in range(1,51)}
    for key in postingsList[word]:
        tfidf[word][key] = (1 + math.log10(postingsList[word][key])) * idf[word]


magnitudes = {}

i = 1
mag = 0

# Finding Magnitude of Each Vector
for i in range(1,51):
    for word in tfidf:
        mag += tfidf[word][str(i)]**2
    mag = math.sqrt(mag)
    magnitudes[str(i)] = mag
    mag = 0
    print(magnitudes[str(i)])

# Writing to postingsList.json
with open('./postingsList.json', 'w') as f:
    json.dump(postingsList, fp=f, sort_keys=True)

# Writing DF to docFreq.json
with open('./docFreq.json', 'w') as f:
    json.dump(docFreq, fp=f, sort_keys=True)

# Writing DF to idf.json
with open('./idf.json', 'w') as f:
    json.dump(idf, fp=f, sort_keys=True)

# Writing tfidf to tfidf.json
with open('./tfidf.json', 'w') as f:
    json.dump(tfidf, fp=f, sort_keys=True)

# Writing magnitudes to magnitudes.json
with open('./magnitudes.json', 'w') as f:
    json.dump(magnitudes, fp=f, sort_keys=True)