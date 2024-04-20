from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import pickle
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def nltk_to_wordnet_pos(nltk_pos_tag):
    if nltk_pos_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif nltk_pos_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif nltk_pos_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif nltk_pos_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None

def lemmatize_with_pos(tokens_with_pos):
    lemmatized_tokens = []
    for token, nltk_pos_tag in tokens_with_pos:
        wordnet_pos_tag = nltk_to_wordnet_pos(nltk_pos_tag)
        if wordnet_pos_tag is not None:
            lemmatized_token = WordNetLemmatizer().lemmatize(token, pos=wordnet_pos_tag)
        else:
            lemmatized_token = token
        lemmatized_tokens.append(lemmatized_token)
    return lemmatized_tokens

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in set(stopwords.words('english'))]
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = lemmatize_with_pos(pos_tags)
    return ' '.join(lemmatized_tokens)

# Load TF-IDF vectorizer
with open('./models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load sentiment classifier
with open('./models/LR_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if 'comment' in data:
        comment = data['comment']
        processed_comment = preprocess_text(comment)
        tfidf_comment = tfidf_vectorizer.transform([processed_comment])
        prediction = classifier.predict(tfidf_comment)[0]
        return jsonify({'sentiment': prediction})
    else:
        return jsonify({'error': 'Comment field missing'})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        df = pd.read_csv(file)

        # Preprocess text
        df['clean_reviewText'] = df['reviewText'].apply(preprocess_text)

        # Perform sentiment analysis
        tfidf_matrix = tfidf_vectorizer.transform(df['clean_reviewText'])
        df['predicted_sentiment'] = classifier.predict(tfidf_matrix)

        # Create sentiment distribution plot
        sentiment_counts = df['predicted_sentiment'].value_counts()
        sentiment_fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
                               labels={'x': 'Sentiment', 'y': 'Count'}, title='Sentiment Analysis Results')

        # Generate category-wise sentiment counts
        category_sentiment_counts = df.groupby('category')['predicted_sentiment'].value_counts().unstack(fill_value=0)
        category_sentiment_fig = px.bar(category_sentiment_counts, barmode='group',
                                         labels={'x': 'Category', 'y': 'Count'}, title='Category-wise Sentiment Analysis')

        # Convert plots to JSON strings
        sentiment_graph_json = sentiment_fig.to_json()
        category_sentiment_graph_json = category_sentiment_fig.to_json()

    return render_template('dashboard.html', sentiment_graph_json=sentiment_graph_json,
                           category_sentiment_graph_json=category_sentiment_graph_json)


if __name__ == '__main__':
    app.run(debug=True)
