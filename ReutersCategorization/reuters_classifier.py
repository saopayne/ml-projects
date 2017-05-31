import nltk
from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# download the reuters corpus
nltk.download('reuters')
nltk.download('punkt')


# Collect stats about the reuters documents
def collection_stats():
    # list of documents
    documents_stat = reuters.fileids()
    print(str(len(documents)) + " documents")

    train_docs_stat = list(filter(lambda doc: doc.startswith("train"), documents_stat))
    print(str(len(train_docs_stat)) + " total training documents")

    test_docs_stat = list(filter(lambda doc: doc.startswith("test"), documents_stat))
    print(str(len(test_docs_stat) + " total test documents"))

    # list of categories
    categories = reuters.categories()
    print(str(len(categories)) + " categories")

    # get the documents in a category
    category_docs = reuters.fileids("acq")

    # words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0])
    print(document_words)

    # print the raw document
    print(reuters.raw(document_id))

cachedStopWords = stopwords.words("english")


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens


def represent(documents):
    train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

    # Tokenization
    vectorizer = TfidfVectorizer(tokenizer=tokenize)

    # Learn and transform train documents
    vectorised_train_documents = vectorizer.fit_transform(train_docs)
    vectorised_test_documents = vectorizer.transform(test_docs)

    # Transform multilabel labels
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
    test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])

    return vectorised_train_documents, train_labels, vectorised_test_documents, test_labels


def train_classifier(train_docs, train_labels):
    classifier = OneVsRestClassifier(LinearSVC(random_state=42))
    classifier.fit(train_docs, train_labels)
    return classifier


def evaluate(test_labels, predictions):
    precision = precision_score(test_labels, predictions, average='micro')
    recall = recall_score(test_labels, predictions, average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')

    print("Macro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))


documents = reuters.fileids()
train_docs, train_labels, test_docs, test_labels = represent(documents)
model = train_classifier(train_docs, train_labels)
predictions = model.predict(test_docs)
evaluate(test_labels, predictions)
