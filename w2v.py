import re
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# Preprocess the input text
def preprocess_text(paragraph):
    text = re.sub(r'\[[0-9]*\]', ' ', paragraph)  # Remove references like [1]
    text = re.sub(r'\s+', ' ', text)             # Remove extra whitespace
    text = text.lower()                          # Convert to lowercase
    text = re.sub(r'\d', ' ', text)              # Remove digits
    text = re.sub(r'\s+', ' ', text)             # Remove extra spaces
    return text

# Tokenize and remove stopwords
def tokenize_and_filter(text):
    sentences = nltk.sent_tokenize(text)  # Sentence tokenization
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]  # Word tokenization
    filtered_sentences = [
        [word for word in sentence if word not in stopwords.words('english')]
        for sentence in sentences
    ]
    return filtered_sentences

# Train Word2Vec model
def train_word2vec(sentences, vector_size=100, window=5, sg=1, min_count=1):
    return Word2Vec(sentences, vector_size=vector_size, window=window, sg=sg, min_count=min_count)

# Display similar words
def display_similar_words(model, word, topn=5):
    try:
        similar = model.wv.most_similar(word, topn=topn)
        print(f"Words similar to '{word}': {similar} \n")
    except KeyError:
        print(f"Word '{word}' not found in the vocabulary.")

# Example usage
if __name__ == "__main__":
    # Input paragraph
    paragraph = """
    Word embeddings are a type of word representation that allows words with similar meaning 
    to have a similar representation. They are a distributed representation for text that is 
    perhaps one of the key breakthroughs for the impressive performance of deep learning methods 
    on challenging natural language processing problems.
    """
    
    # Preprocess text and prepare sentences
    processed_text = preprocess_text(paragraph)
    sentences = tokenize_and_filter(processed_text)

    # Train Word2Vec model (Skip-gram)
    model = train_word2vec(sentences, vector_size=100, window=5, sg=1)

    # Display word vector and similar words
    word_to_explore = 'representation'
    try:
        vector = model.wv[word_to_explore]
        print(f"Vector for '{word_to_explore}':\n{vector} \n")
    except KeyError:
        print(f"Word '{word_to_explore}' not found in the vocabulary.")

    display_similar_words(model, 'representation', topn=3)
    display_similar_words(model, 'text', topn=3)

    # CBOW vs Skip-gram example
    sample_sentences = [
        ["I", "love", "natural", "language", "processing"],
        ["Word2Vec", "is", "a", "great", "tool"],
        ["Machine", "learning", "is", "fun"],
        ["Natural", "language", "processing", "is", "awesome"]
    ]

    # CBOW Model
    cbow_model = train_word2vec(sample_sentences, vector_size=100, window=2, sg=0)
    cbow_similar = cbow_model.wv.most_similar('Machine', topn=2)
    print(f"CBOW - Words similar to 'Machine': {cbow_similar} \n")

    # Skip-gram Model
    skipgram_model = train_word2vec(sample_sentences, vector_size=100, window=2, sg=1)
    skipgram_similar = skipgram_model.wv.most_similar('Machine', topn=3)
    print(f"Skip-gram - Words similar to 'Machine': {skipgram_similar} \n")
