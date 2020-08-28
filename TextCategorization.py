import os
os.chdir('E:/Freelancer/22Aug2020/')
import pandas as pd
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB



PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))



STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


cnt = Counter()

FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])


n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])


stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()
# def lemmatize_words(text):
#     return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# from nltk.corpus import wordnet
# from nltk.stem import WordNetLemmatizer
# import nltk

# lemmatizer = WordNetLemmatizer()
# wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
# def lemmatize_words(text):
#     pos_tagged_text = nltk.pos_tag(text.split())
#     return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


def preprocessing(dataframe):   
    dataframe["SearchTerm"] = dataframe["SearchTerm"].str.lower()
    dataframe["SearchTerm"] = dataframe["SearchTerm"].apply(lambda text: remove_punctuation(text))
    dataframe["SearchTerm"] = dataframe["SearchTerm"].apply(lambda text: remove_stopwords(text))
    dataframe["SearchTerm"] = dataframe["SearchTerm"].apply(lambda text: remove_freqwords(text))
    dataframe["SearchTerm"] = dataframe["SearchTerm"].apply(lambda text: remove_rarewords(text))
    dataframe["SearchTerm"] = dataframe["SearchTerm"].apply(lambda text: stem_words(text))
    # dataframe["SearchTerm"] = dataframe["SearchTerm"].apply(lambda text: lemmatize_words(text))
    # dataframe["SearchTerm"] = dataframe["SearchTerm"].apply(lambda text: lemmatize_words(text))
    return dataframe


def train():
	print('Loading the dataset')
	df = pd.read_csv("trainSet.csv",header=None,nrows=5000)
	df.columns= ['SearchTerm','Category']

	print('Shape of the dataset is :',df.shape)

	df.head()

	df = preprocessing(df)

	tfidf = TfidfVectorizer( max_features=1000)
	X_train, X_test, y_train, y_test = train_test_split(df.SearchTerm, df.Category, test_size=0.33, random_state=42)

	print('Creating word Embedding')
	X_train_mat = tfidf.fit_transform(X_train)
	X_test_mat = tfidf.transform(X_test)

	print('Model Building initilized')
	NBclf = MultinomialNB()
	NBclf.fit(X_train_mat,y_train)

	y_pred_nb = NBclf.predict(X_test_mat)

	print( 'Accuracy of model is ',accuracy_score(y_test,y_pred_nb))

	print(classification_report(y_test,y_pred_nb))


def test():
	testSet = pd.read_csv('candidateTestSet.txt',header=None)
	testSet.columns = ['SearchTerm']

	testSet = preprocessing(testSet)
	testSetMat = tfidf.transform(testSet.SearchTerm)
	testSet['Category'] = NBclf.predict(testSetMat)
	testSet.to_csv('output.csv',index=False)
