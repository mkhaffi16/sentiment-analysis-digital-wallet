import nltk
import emoji
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm 
import datetime as dtm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay

nltk.download('stopwords')
nltk.download('punkt')
sns.set_style("whitegrid")
warnings.simplefilter(action='ignore', category=FutureWarning)



class LoadData:
  def loadData(self, loc):
    data = pd.read_csv(loc)
    return data

  def savedata(self, loc, data, file_name):
    data.to_csv(loc+file_name, index=False)
    return loc+file_name

# Kelas Preprocess
class Preprocess:
  kamus_normal = pd.read_csv("app/static/file/asset/normalisasi.csv",encoding='latin-1',header=None,names=["non-standard word","standard word"])
  meaningless = pd.read_csv("app/static/file/asset/new_stopword_review.csv",header=None,names=['stopword'])

  def preproses_data(self, data):
    #Cleaning
    data['Preprocess'] = data['content'].str.replace("&am", " ")
    data['Preprocess'] = data['Preprocess'].str.replace("&gt", " ")
    data['Preprocess'] = data['Preprocess'].str.replace("\\\\r", " ")
    data['Preprocess'] = data['Preprocess'].str.replace("    ", " ")
    data['Preprocess'] = data['Preprocess'].str.replace('"', '')
    data['Preprocess'] = data['Preprocess'].str.replace('@[\w]+','')
    data['Preprocess'] = data['Preprocess'].str.replace("\\\\n", " ")
    data['Preprocess'] = data['Preprocess'].str.replace("\n", " ")
    data['Preprocess'] = data['Preprocess'].str.replace("\r", " ")
    #hapus web   
    data['Preprocess'] = data['Preprocess'].str.replace(r'''(?i)\b((?:https|http?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ",regex=True)
    data['Preprocess'] = data['Preprocess'].str.replace('"', '')
    data['Preprocess'] = data['Preprocess'].str.replace("\\\\x[a-zA-z0-9][a-zA-z0-9]","",regex=True)
    #CaseFolding
    data['Preprocess'] = data['Preprocess'].str.lower()
    #PUNCTUATION REMOVAL
    data['Preprocess'] = data['Preprocess'].str.replace('(?<=\w)[^\s\w](?![^\s\w])',' ',regex=True)
    #normalization
    def normalize_text(text, stdword_, nonstdword_):
      text = text.split(" ")
      for i in range(len(text)):
        if text[i] in nonstdword_:
          index = nonstdword_.index(text[i])
          text[i] = stdword_[index]
      return ' '.join(map(str, text))

    nonstdword = self.kamus_normal['non-standard word'].values.tolist()
    stdword = self.kamus_normal['standard word'].values.tolist()
    meaningless = self.meaningless['stopword'].tolist()

    data['Preprocess'] = data['Preprocess'].map(lambda com : normalize_text(com,stdword,nonstdword))
    data['Preprocess'] = data['Preprocess'].str.replace('[^a-zA-Z_]+',' ',regex=True)
    for word in meaningless:
      regex_meaningless = r"\b" + word + r"\b"
      data['Preprocess'] = data['Preprocess'].str.replace(regex_meaningless, '')

    # STOPWORD REMOVAL
    stop_words = list(stopwords.words('indonesian'))
    stop_words = stop_words+["user","rt","retweet","url","banget"]

    stop_words.remove('tidak')
    stop_words.remove('baik')
    stop_words.remove('lama')
    stop_words.remove('jangan')
    stop_words.remove('benar')
    stop_words.remove('kurang')

    for stop_word in stop_words:
      regex_stopword = r"\b" + stop_word + r"\b"
      data['Preprocess'] = data['Preprocess'].str.replace(regex_stopword, '')

    # STEMMING
    factory = StemmerFactory()
    stemmerID = factory.create_stemmer()

    def stemming(text, stemmer_id):
      text_split = text.split(" ")
      stemmed_list = []
      for i in text_split:
          if '_' not in i:
              stem_text1 = stemmer_id.stem(i)
              stemmed_list.append(stem_text1)
          else:
            stemmed_list.append(i)
      stemmed = ' '.join(map(str,stemmed_list))
      return stemmed  

    data['Preprocess'] = data['Preprocess'].map(lambda com : stemming(com,stemmerID))

    #TOKENIZING
    data['review_tokenized'] = data['Preprocess'].apply(word_tokenize)
    return data 


  def Word2Vec(self,data):
     Y_data = data[['CustomerService','FiturAplikasi','UserExperience','Verifikasi']]
     word2vec = pickle.load(open("app/static/file/asset/Word2VecModel.pkl", 'rb'))

     sentences = [word_tokenize(sentence) for sentence in data['Preprocess']]
     sentence_vectors = []
     
     for sentence in sentences:
        valid_words = [word for word in sentence if word in word2vec.wv.key_to_index]
        if valid_words:
           vector_sum = sum(word2vec.wv[word] for word in valid_words)
           average_vector = vector_sum / len(valid_words)
           sentence_vectors.append(average_vector)
        else:
          sentence_vectors.append(None)

     return sentence_vectors,Y_data

    
  
class XGBoostAlgorithm:
  def train(self, x, y):
    x = np.array(x)  # Convert x to a numpy array
    y = np.array(y)  # Convert y to a numpy array
    classifier = MultiOutputClassifier(XGBClassifier(learning_rate=0.75,max_depth=50,n_estimators = 100, random_state=0))
    model = Pipeline([('classify', classifier)])
    model.fit(x,y)
    return model

  def test(self, x, model):
    predictions = model.predict(x)
    return predictions

  def savemodel(self, model):
    filename = 'app/static/file/model/finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return "savefile"

  def loadmodel(self, loc):
    filename = 'app/static/file/model/'+loc
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


class Evaluasi:
  def report(self, label_test, predictions):
    return classification_report(label_test, predictions, output_dict=True, digits=4)

  def cm(self, y, pred, name, judul):
      cf_matrix = confusion_matrix(y, pred)
      labels = [-1,0,1]
      # disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix , display_labels=labels)

      ax = sns.heatmap(cf_matrix, annot=True, xticklabels=labels,
                       yticklabels=labels, cmap="YlGnBu", fmt='d')
      plt.title(judul)
      plt.xlabel('Prediction')
      plt.ylabel('Actual')
      self.t = dtm.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
      self.nm = "HeatMap_" + name + self.t + '.png'
      plt.savefig('app/static/img/grafik/'+self.nm)
      # plt.show()
      plt.close()
      return self.nm


