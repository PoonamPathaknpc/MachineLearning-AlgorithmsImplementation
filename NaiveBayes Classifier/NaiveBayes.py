from __future__ import division

import math
import operator
import os
import io
import pickle
from collections import defaultdict
import PerformanceParams

class NaiveBayes:

    'initialization of bayes with the text,stopwords and categories set '
    def __init__(bayes):
        #'reload(sys)'  # had to deal with 'unicode' issues :/
        #'sys.setdefaultencoding('utf8')' not required in python3
        bayes.bagofwords = set()
        bayes.categories = {}  # count of 'categories given classes'
        bayes.word_counts = defaultdict(dict)  # used to compute likelihood of a 'word' given a class ;
                                              # {cateogry: (word, count) }
        bayes.perf_params = PerformanceParams.PerformanceParams()
        bayes.d_path = './data/'
        bayes.m_path = './Results/'


    def load_data(bayes,fname):
        f = open(fname, 'r',encoding = "utf8",errors='ignore')
        doc = f.read()

        f.close()
        return doc

    def StopWordlist(bayes,fname):
        f = open(fname + 'stopwords.txt','r')
        stopwords = [line.rstrip('\n').rstrip('\r') for line  in f]
        return stopwords

    def training(bayes, f, category,label):
        'split the words'

        lines = f.readlines()
        for line in lines:
            words = line.split(" ")

        #words = nltk.word_tokenize(text)
        # 'find the unique words and then add it to the bag of words'

        if label=='SSwords':
         #'filter the terms which are stop words'
         StopWordlist = bayes.StopWordlist(bayes.d_path)         
         words = [word.encode('utf-8') for word in words if word not in StopWordlist]
         #print(words)
        else:
         # do not filter the words with stopwords..
         words = [word.encode('utf-8') for word in words]

        unique_word = set(words)
        #print(unique_word)
        bayes.bagofwords = bayes.bagofwords.union(unique_word)

        #'check if the category is new , '
        #'if new add it as new key in categories or increment the value of the already present key'
        if category not in bayes.categories.keys():
            bayes.categories[category] = 1
        else:
            bayes.categories[category] += 1

        #'for every  word in unique words check if the term already exists in the word counts '
        #'if yes, then add the occurrences of the the term '
        #'else assign it as new term '
        for word in unique_word:
            if word in bayes.word_counts[category]:
                bayes.word_counts[category][word] = bayes.word_counts[category][word] + words.count(word)
            else:
                bayes.word_counts[category][word] = words.count(word)

    def classify_posterior(bayes,f,accur):

        lines = f.readlines()
        for line in lines:
                words = line.split(" ")

        #words = nltk.word_tokenize(text, language='english')
        # 'filter the terms which are stop words'
        if accur=='SSwords':
         StopWordlist = bayes.StopWordlist(bayes.d_path)
         words = [word.encode('utf-8') for word in words if word not in StopWordlist]
         
        else:
         words = [word.encode('utf-8') for word in words]
        posterior = {}

        total_categories = 0.
        for cat in bayes.categories.keys():
            total_categories += bayes.categories[cat]

        for cat in bayes.categories.keys():
            probLikelihood = 0.  # probability of a word given category, i.e likelihood
            # finding likelyhood with Laplace Smoothening
            for word in words:
                if word in bayes.word_counts[cat]:
                    probLikelihood += math.log((bayes.word_counts[cat][word] + 1) /
                                              (len(bayes.word_counts[cat]) +
                                               len(bayes.bagofwords) + 1), 2)
                # Laplace smoothening for the word that doesntt exist in word_counts[cat]
                else:
                    probLikelihood += math.log(1. / (len(bayes.word_counts[cat]) +
                                                    len(bayes.bagofwords) + 1), 2)
            #'calculating posterior = likelyhood*prob(cat) .. log(likelyhood) + log(prob(cat)'
            prior_cat = math.log(bayes.categories[cat] / total_categories, 2)
            posterior[cat] = probLikelihood + prior_cat

        # print(posterior)
        return posterior

    # Store the state of 'dictionary, priors & word_counts' , helpful for large datasets
    def writeToFile(bayes,accur):
        if accur=='SSwords':
             f = open(bayes.m_path + 'bagofwords.pkl', 'wb')
             pickle.dump(bayes.bagofwords, f)
             f.close()
             f = open(bayes.m_path + 'categories.pkl', 'wb')
             pickle.dump(bayes.categories, f)
             f.close()
             f = open(bayes.m_path + 'word_counts.pkl', 'wb')
             pickle.dump(bayes.word_counts, f)
             f.close()
        else:
            f = open(bayes.m_path + 'bagofwordswithss.pkl', 'wb')
            pickle.dump(bayes.bagofwords, f)
            f.close()
            f = open(bayes.m_path + 'categorieswithss.pkl', 'wb')
            pickle.dump(bayes.categories, f)
            f.close()
            f = open(bayes.m_path + 'word_countswithss.pkl', 'wb')
            pickle.dump(bayes.word_counts, f)
            f.close()

    # To load the persisted 'dictionary, priors & word_counts'
    def ReadDSFromFile(bayes,accur):
         if accur == 'SSwords':
           f = open(bayes.m_path + 'bagofwords.pkl', 'rb')
           bayes.bagofwords = pickle.load(f)
           f.close()
           f = open(bayes.m_path + 'categories.pkl', 'rb')
           bayes.categories = pickle.load(f)
           f.close()
           f = open(bayes.m_path + 'word_counts.pkl', 'rb')
           bayes.word_counts = pickle.load(f)
           f.close()
         else:
            f = open(bayes.m_path + 'bagofwordswithss.pkl', 'rb')
            bayes.bagofwords = pickle.load(f)
            f.close()
            f = open(bayes.m_path + 'categorieshsswit.pkl', 'rb')
            bayes.categories = pickle.load(f)
            f.close()
            f = open(bayes.m_path + 'word_countswithss.pkl', 'rb')
            bayes.word_counts = pickle.load(f)
            f.close()
    # Initialize the data of a specific category
    def initialize_data(bayes, path, label,accur):
        data = {}
        # Read all files in a directory
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith('.txt'):
                        #data[f] = bayes.load_data(os.path.abspath(os.path.join(root, f)))
                        fi = io.open(os.path.abspath(os.path.join(root, f)), 'r',errors='ignore')
                        bayes.training(fi, label, accur)

        else:
            print(path, ' is invalid !')
            return

        # Train the data with the given 'label'
        #for d in data.keys():
        #  for text in data[d]:
         #     bayes.training(data[d], label,accur)


    # For single sentence
    def classify_SentbySent(bayes, f,accur):
        predclass = bayes.classify_posterior(f,accur)
        #'take the category with maximum posterier value'
        prediction = max(predclass.items(), key=operator.itemgetter(1))[0]
        return prediction

    def Accuracy_sentence_score(bayes, f,  tradclass,accur):
        prediction = bayes.classify_SentbySent(f,accur)
        # print 'True Label: ', label, ' -- Prediction: ', prediction

        if prediction == tradclass:
            bayes.perf_params.true += 1
        else:
            bayes.perf_params.false += 1

    # For a directory ...
    # check if prediction = label in test data
    def classify_text(bayes, path, label,accur):
        data = {}
        # Read the test data
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith('.txt'):
                        #data[f] = bayes.load_data(os.path.abspath(os.path.join(root, f)))
                        fi = io.open(os.path.abspath(os.path.join(root, f)), 'r', errors='ignore')
                        bayes.Accuracy_sentence_score(fi, label, accur)
        else:
            print (path, ' is invalid !')

        #print( '# of ', label, ' files: ', len(data))


        #for d in data:
         #  bayes.Accuracy_sentence_score(data[d], label,accur)


    def initialize_training_data(bayes,accur):
        ham_path = bayes.d_path + 'hw2_train/train/ham'
        spam_path = bayes.d_path + 'hw2_train/train/spam'

        #'train the data'
        print ('*** Training Set ***')
        bayes.initialize_data(ham_path, 'ham',accur)
        bayes.initialize_data(spam_path, 'spam',accur)



    def NBClassification_test(bayes,accur):
        ham_test_path = bayes.d_path + 'hw2_test/test/ham'
        spam_test_path = bayes.d_path + 'hw2_test/test/spam'

        print('*** Test Set ***')
        bayes.classify_text(ham_test_path, 'ham',accur)
        bayes.classify_text(spam_test_path, 'spam',accur)



    def test_naive_bayes(bayes,accur, load_persistence=False):
        print('... test_NaiveBayes() ...')
        if load_persistence:
            bayes.ReadDSFromFile(accur)
        else:
            bayes.initialize_training_data(accur)
            bayes.writeToFile(accur)

        bayes.NBClassification_test(accur)
        bayes.perf_params.get_accuracy()
        if accur == 'SSwords':
            print('accuracy for the Naive Bayes Text Classification is with Stop words is:')
        else:
            print('accuracy for the Naive Bayes Text Classification without Stop words is:')
        print( bayes.perf_params.accuracy)

if __name__ == '__main__':
    NaiveBayes().test_naive_bayes('SSwords', load_persistence=False)
    NaiveBayes().test_naive_bayes('tSSwords', load_persistence=False)

