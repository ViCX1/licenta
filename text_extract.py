from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import ssl
import nltk


string.punctuation

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


img = Image.open("sample.jpg")


text = pytesseract.image_to_string(img)


remove_punct(text)

print(text)
      ###  Write to Text File ######


import csv
row_list = [
             [text, "positive"],]
#with open('data/train/IMDB Dataset.csv', 'a', newline='') as file:
with open('test.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

    #49735 -35(+2) incolo
    #49755