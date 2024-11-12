# preprocessing/text_processing.py
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder

def preprocess_text(text):
    """
    Preprocess text by removing special characters, tokenizing, removing stopwords,
    and lemmatizing.
    """
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text.lower())

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def apply_text_processing(data, text_column):
    """
    Apply the text preprocessing function to the given text column in the DataFrame.
    """
    data[text_column] = data[text_column].apply(preprocess_text)
    return data


def encode_labels(data, label_column):
    """
    Encode text labels into numeric labels.
    """
    encoder = LabelEncoder()
    data[label_column] = encoder.fit_transform(data[label_column])
    return data, encoder



# preprocessing/feature_extraction.py
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(data, text_column):
    """
    Convert text into numerical features using TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    features = vectorizer.fit_transform(data[text_column]).toarray()
    return features, vectorizer


# preprocessing/split_data.py
from sklearn.model_selection import train_test_split

def split_data(features, labels, test_size=0.2):
    """
    Split the features and labels into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test