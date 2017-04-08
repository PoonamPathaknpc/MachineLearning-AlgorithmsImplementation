import math
import os
import io
import PerformanceParams
import pickle
import sys

class LogisticRegression:
   def __init__(LR):
     LR.bagofwords = set()
     LR.weight = []
     LR.words = []
     LR.bagofwords_ham = set()
     LR.bagofwords_spam = set()
     LR.perf_params = PerformanceParams.PerformanceParams()
     LR.d_path = './data/'
     LR.m_path = './Results/LR/'

   def StopWordlist(bayes, fname):
       f = open(fname + 'stopwords.txt', 'r')
       stopwords = [line.rstrip('\n') for line in f]
       return stopwords

   def sigmoid(LR,x):
    #''Compute the sigmoid function '''
    den = 1.0 + math.e ** (-1.0 * x)
    sig = 1.0 / den
    return sig

   # assembling the feartures for the logistic regression equation
   def initialize_data(LR, label,accur):
        StopWordlist = LR.StopWordlist(LR.d_path)
        if label == 'ham':
            train_path = LR.d_path + 'hw2_train/train/ham'
        else:
            train_path = LR.d_path + 'hw2_train/train/spam'

        UniqueWords = set()
        # Read all files in a directory
        if os.path.isdir(train_path):
            for root, dirs, files in os.walk(train_path):
                for f in files:
                    if f.endswith('.txt'):
                        fi = io.open(os.path.abspath(os.path.join(root, f)), 'r',errors='ignore')
                        lines = fi.readlines()
                        #print(lines)
                        for line in lines:
                            terms = line.split(" ")
                            if accur == 'withoutSSwords':
                              words = [word.rstrip('\n') for word in terms]
                              words = [word for word in words if word not in StopWordlist]
                              #print(words)

                            else:
                              words = [word.rstrip('\n') for word in terms]
                              #print(words)


                        UniqueWords = set(words)
                        if accur == 'withoutSSwords':
                          UniqueWords = [word for word in UniqueWords if word not in LR.bagofwords]


                        #print(UniqueWords)
                        LR.bagofwords= LR.bagofwords.union(UniqueWords)


        else:
            print(train_path, ' is invalid !')
            return

         # Store the state of 'dictionary, priors & word_counts' , helpful for large datasets
   def writeToFile(LR, accur):
          if accur == 'withoutSSwords':
                   f = open(LR.m_path + 'bagofwords.pkl', 'wb')
                   pickle.dump(LR.bagofwords, f)
                   f.close()
          else:
                  f = open(LR.m_path + 'bagofwordswithss.pkl', 'wb')
                  pickle.dump(LR.bagofwords, f)
                  f.close()


        # To load the persisted 'dictionary, priors & word_counts'
   def ReadDSFromFile(LR, accur):
           if accur == 'withoutSSwords':
                    f = open(LR.m_path + 'bagofwords.pkl', 'rb')
                    LR.bagofwords = pickle.load(f)
                    f.close()

           else:
                    f = open(LR.m_path + 'bagofwordswithss.pkl', 'rb')
                    LR.bagofwords = pickle.load(f)
                    f.close()


   # Cost regression function
   def costReg(LR, file, theta, label, learningRate,accur,y=0):
    StopWordlist = LR.StopWordlist(LR.d_path)
    sum = 0
    uniquewords = set()
    m=len(LR.bagofwords)
    #print(m)
    if label=='ham':
     y=1

    lines = file.readlines()
    for line in lines:
      words = line.split(" ")
      if accur == 'withoutSSwords':
          terms = [word for word in words if word not in StopWordlist]
      else:
          terms = [word for word in words if word]
      uniquewords = uniquewords.union(terms)

      for i in range(len(LR.words)):
         if LR.words[i] in words:
             sum += 1*LR.weight[i]
    sum = sum + 4
    delta = LR.sigmoid(sum*theta) - y
    grad = (1.0 / m) * delta  + ((learningRate / 2*m)*(theta*theta))

    for i in range(len(LR.weight)):
            LR.weight[i]+=grad

   def Training_Data(LR, accur,lr):
       LR.initialize_data('ham', accur)
       LR.initialize_data('spam', accur)

       n=0
       train_path_ham = LR.d_path + 'hw2_train/train/ham'
       train_path_spam = LR.d_path + 'hw2_train/train/spam'
       theta = 0.001
       learningRate = lr

       for i in range(len(LR.bagofwords)):
           LR.weight.append(0.5)

       i=0
       for word in LR.bagofwords:
           LR.words.append(word)
           i+=1

       if os.path.isdir(train_path_ham):
           for root, dirs, files in os.walk(train_path_ham):
               for f in files:
                   if f.endswith('.txt')& n<10:
                       fi = io.open(os.path.abspath(os.path.join(root, f)), 'r', errors='ignore')
                       LR.costReg(fi,theta,'ham', learningRate,accur)
                       n+=1
       else:
           print(train_path_ham, ' is invalid !')

       n=0
       if os.path.isdir(train_path_spam):
           for root, dirs, files in os.walk(train_path_spam):
               for f in files:
                   if f.endswith('.txt')& n<10:
                       fi = io.open(os.path.abspath(os.path.join(root, f)), 'r', errors='ignore')
                       LR.costReg(fi,theta,'spam', learningRate, accur)
                       n+=1
       else:
            print(train_path_spam, ' is invalid !')


   # predict the class baed on the LR equation converged
   def predict(LR,file,accur):
    sum = 0
    theta = 0.01
    uniquewords = set()
    StopWordlist = LR.StopWordlist(LR.d_path)
    lines = file.readlines()
    for line in lines:
        words = line.split(" ")
        if accur == 'withoutSSwords':
            terms = [word for word in words if word not in StopWordlist]
        else:
            terms = [word for word in words]
        uniquewords = uniquewords.union(terms)

    for i in range(len(LR.words)):
            if LR.words[i] in uniquewords:
                sum += 1 * LR.weight[i]

    h = LR.sigmoid(sum * theta)
    if h > 0.5:
        return 'ham'
    else:
        return 'spam'

   def Classify(LR,path,accur,label):
        print('*** Test Set ***')
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith('.txt'):
                        fi = io.open(os.path.abspath(os.path.join(root, f)), 'r', errors='ignore')
                        prediction = LR.predict(fi, accur)
                        if prediction==label:
                            LR.perf_params.true += 1
                        else:
                            LR.perf_params.false += 1

   def LRTest(LR,accur,lr,load_persistence=False):
       ham_test_path = LR.d_path + 'hw2_test/test/ham'
       spam_test_path = LR.d_path + 'hw2_test/test/spam'
       if load_persistence:
           LR.ReadDSFromFile(accur)
       else:
           LR.Training_Data(accur,lr)
           LR.writeToFile(accur)

       LR.Classify(ham_test_path,accur,'ham')
       LR.Classify(spam_test_path, accur, 'spam')
       LR.perf_params.get_accuracy()
       if accur=='withSSwords':
        print('accuracy for the Logistic Regression with theta=0.001 and learning rate =' +  sys.argv[1] +  ' with Stop words is:')
       else:
        print('accuracy for the Logistic Regression with theta=0.001 and learning rate =' +  sys.argv[1] +  ' without Stop words is:')
       print(LR.perf_params.accuracy)


# set X and y (remember from above that we moved the label to column 0)
if __name__ == '__main__':
    lr= sys.argv[1]
    LogisticRegression().LRTest('withSSwords',False)
    LogisticRegression().LRTest('withoutSSwords',False)
