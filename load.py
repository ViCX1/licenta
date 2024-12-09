import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords 
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import ssl
import nltk
from fer import FER
import sys

string.punctuation

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[^\w]', ' ', text)
    return text

 

#img = Image.open("sample.jpg")
img = Image.open(sys.argv[1])


text = pytesseract.image_to_string(img)

print(text)

bad_chars = ['@', '#', '$', "^"]
 
 
# using replace() to
# remove bad_chars
for i in bad_chars :
    text = text.replace(i, '')
#ntext = remove_punct(text)

      ###  Write to Text File ######


import csv
row_list = [
             [str(text), "positive"],]

with open('test.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)
    file.close()


is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


#img = plt.imread("sample.jpg")
img = plt.imread(sys.argv[1])
detector = FER(mtcnn=True)
#print(detector.detect_emotions(img))
#plt.imshow(img)
emotion, score = detector.top_emotion(img)

#index update
with open("data/test/index.txt", "r+") as file:

    first_line = file.readline()
    findex = int(first_line) + 1
    file.seek(0)
    file.write(str(findex))
    file.truncate()
    file.close()

test_csv = 'test.csv'
tdb = pd.read_csv(test_csv)
tdb.head()

base_csv = 'data/train/train.csv'
df = pd.read_csv(base_csv)
df.head()

X,y = df['data'].values,df['sentiment'].values
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)

def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s


def tockenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(stopwords.words('english')) 
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
  
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
    
    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                    if preprocess_string(word) in onehot_dict.keys()])
            
    encoded_train = [1 if label =='positive' else 0 for label in y_train]  
    encoded_test = [1 if label =='positive' else 0 for label in y_val] 
    return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test), np.array(encoded_test),onehot_dict

x_train,y_train,x_test,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, data in enumerate(sentences):
        if len(data) != 0:
            features[ii, -len(data):] = np.array(data)[:seq_len]
    return features


model = torch.load('data/model/state_dict.pt')

def predict_text(text):
        word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
                         if preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(padding_(word_seq,500))
        inputs = pad.to(device)
        batch_size = 1
        h = model.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
        return(output.item())





#index = 3      #49755  #30
index = findex
print(tdb['data'][index])
print('='*70)
print(f'Actual sentiment is  : {tdb["sentiment"][index]}')
print('='*70)
pro = predict_text(tdb['data'][index])
status = "positive" if pro > 0.5 else "negative"
pro = (1 - pro) if status == "negative" else pro
print(f'Predicted sentiment is {status} with a probability of {pro}')
print('='*70)

print('Face emotion:',emotion,score)
print('='*70)
if emotion == 'angry' and status=="negative":
    overall = pro + 0.10
    print("Overall sentiment is: negative ",overall)
elif emotion == 'disgust' and status=="negative":
    overall = pro + 0.10
    print("Overall sentiment is: negative ",overall)
elif emotion == 'sad' and status=="negative":
    overall = pro + 0.10
    print("Overall sentiment is: negative ",overall)
elif emotion == 'fear' and status=="negative":
    overall = pro + 0.10
    print("Overall sentiment is: negative ",overall)
elif emotion == 'happy' and status=="positive":
    overall = pro + 0.10
    print("Overall sentiment is: positive ",overall)
elif emotion == 'surprise':
    print("Overall sentiment is: ",status)
elif emotion == 'neutral':
    print("Overall sentiment is: ",status)
else:
    print("Overall sentiment is: ",status)




f = open("log.txt", "a")
f.write('\n')
f.write('='*70)
f.write('\n')
f.write(tdb['data'][index])
f.write('\n')
f.write('='*70)
f.write('\n')
f.write(f'Actual sentiment is  : {tdb["sentiment"][index]}')
f.write('\n')
f.write('='*70)
f.write('\n')
f.write(f'Predicted sentiment is {status} with a probability of {pro}')
f.write('\n')
f.write('='*70)
f.write('\n')
f.write(f'Face emotion: {emotion} {score}')
f.write('\n')
f.write('='*70)
f.write('\n')
if emotion == 'angry' and status=="negative":
    #overall = pro + 0.10
    f.write(f'Overall sentiment is: negative {overall}')
    f.write('\n')
    f.write('='*70)
elif emotion == 'disgust' and status=="negative":
    #overall = pro + 0.10
    f.write(f'Overall sentiment is: negative {overall}')
    f.write('\n')
    f.write('='*70)
elif emotion == 'sad' and status=="negative":
    #overall = pro + 0.10
    f.write(f'Overall sentiment is: negative {overall}')
    f.write('\n')
    f.write('='*70)
elif emotion == 'fear' and status=="negative":
    #overall = pro + 0.10
    f.write(f'Overall sentiment is: negative {overall}')
    f.write('\n')
    f.write('='*70)
elif emotion == 'happy' and status=="positive":
    #overall = pro + 0.10
    f.write(f'Overall sentiment is: positive {overall}')
    f.write('\n')
    f.write('='*70)
elif emotion == 'surprise':
    f.write(f'Overall sentiment is: {status}')
    f.write('\n')
    f.write('='*70)
elif emotion == 'neutral':
    f.write(f'Overall sentiment is: {status}')
    f.write('\n')
    f.write('='*70)
else:
    f.write(f'Overall sentiment is: {status}')
    f.write('\n')
    f.write('='*70)
f.write('\n')
f.write('\n')
f.write('\n')
f.write('\n')
f.close()