from flask import Flask, render_template, request
import spacy
from heapq import nlargest
import nltk
nltk.download('punkt')

app = Flask(__name__)

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

def summarize_text(text, select_len=2):
    # Tokenization and Sentence Scoring
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    word_frequencies = {}
    for word in doc:
        if word.is_stop == False and word.is_punct == False:
            if word.text.lower() not in word_frequencies:
                word_frequencies[word.text.lower()] = 1
            else:
                word_frequencies[word.text.lower()] += 1

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    sent_scores = {}
    for sent in sentences:
        score = 0
        for word in nlp(sent):
            if word.text.lower() in word_frequencies.keys():
                score += word_frequencies[word.text.lower()]
        sent_scores[sent] = score

    # Get the summary by selecting top sentences
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)
    return ' '.join(summary)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':
        text = request.form['text']
        select_len = int(request.form['select_len']) if 'select_len' in request.form else 2
        summary = summarize_text(text, select_len)
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
