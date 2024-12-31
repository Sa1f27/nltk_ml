# NLP Techniques and Algorithms - Complete Reference Notebook

# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import gensim
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample Text
text = "Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence. It helps machines understand human language."

# 1. Tokenization
words = word_tokenize(text)
sentences = sent_tokenize(text)
print("Tokenized Words:", words)
print("Tokenized Sentences:", sentences)

# 2. Stopwords Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print("Filtered Words (No Stopwords):", filtered_words)

# 3. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
print("Stemmed Words:", stemmed_words)

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("Lemmatized Words:", lemmatized_words)

# 5. Bag of Words
corpus = ["This is the first document.",
          "This document is the second document.",
          "And this is the third one.",
          "Is this the first document?"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print("Bag of Words:\n", X.toarray())
print("Feature Names:", vectorizer.get_feature_names_out())

# 6. TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
print("TF-IDF:\n", X_tfidf.toarray())
print("Feature Names:", tfidf_vectorizer.get_feature_names_out())

# 7. Word Embeddings (Word2Vec)
sentences = [["natural", "language", "processing"],
             ["artificial", "intelligence"],
             ["machine", "learning", "deep", "learning"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
word_vectors = model.wv
print("Word Vector for 'natural':", word_vectors['natural'])

# 8. Sentiment Analysis (Naive Bayes)
data = pd.DataFrame({
    'text': ["I love this movie", "This movie is terrible", "What a great film", "I hated this movie"],
    'label': [1, 0, 1, 0]
})
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 9. Named Entity Recognition (NER) - Using spaCy (Optional)
# Uncomment the following lines if you have spaCy installed
# import spacy
# nlp = spacy.load("en_core_web_sm")
# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
# for ent in doc.ents:
#     print(ent.text, ent.label_)

# 10. Topic Modeling (LDA) - Using gensim (Optional)
# Uncomment the following lines if you want to explore topic modeling
# from gensim import corpora
# texts = [["natural", "language", "processing"], ["artificial", "intelligence"], ["machine", "learning"]]
# dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]
# import gensim
# lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)
# print(lda_model.print_topics())

# 11. Text Summarization - Using sumy (Optional)
# Uncomment the following lines if you have sumy installed
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# parser = PlaintextParser.from_string(" ".join(sentences), Tokenizer("english"))
# summarizer = LsaSummarizer()
# summary = summarizer(parser.document, 2)
# for sentence in summary:
#     print(sentence)

# 12. Machine Translation - Using transformers (Optional)
# Uncomment the following lines if you have transformers installed
# from transformers import pipeline
# translator = pipeline("translation_en_to_fr")
# print(translator("Natural Language Processing is amazing."))

# 13. Question Answering - Using transformers (Optional)
# Uncomment the following lines if you have transformers installed
# from transformers import pipeline
# qa_pipeline = pipeline("question-answering")
# result = qa_pipeline(question="What is NLP?", context="Natural Language Processing (NLP) is a field of AI.")
# print(result)

# 14. Chatbots - Using transformers (Optional)
# Uncomment the following lines if you have transformers installed
# from transformers import pipeline
# chatbot = pipeline("conversational")
# print(chatbot("Hi, how are you?"))

# 15. Advanced Techniques - BERT (Optional)
# Uncomment the following lines if you have transformers installed
# from transformers import BertTokenizer, BertForSequenceClassification
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# inputs = tokenizer("Hello, how are you?", return_tensors="pt")
# outputs = model(**inputs)
# print(outputs)
