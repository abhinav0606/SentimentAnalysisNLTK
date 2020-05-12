import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter
text=open("hello.txt",encoding="UTF-8").read()
text=text.lower().replace("\n","").replace("’","").replace("”","")
text=text.translate(str.maketrans("","",string.punctuation))
def sentiments(text):
    return SentimentIntensityAnalyzer().polarity_scores(text)
token_words=word_tokenize(text)
clean_words=[]
for i in token_words:
    if i not in stopwords.words("english"):
        clean_words.append(i)
d=[]
with open("emotion.txt","r") as f:
    for i in f:
        clean=i.replace("\n","").strip().replace("'","").replace(" ","").replace(",","")
        t,emotion=clean.split(":")
        if t in clean_words:
            d.append(emotion)
y=Counter(d)
fig1,ax1=plt.subplots()
ax1.bar(y.keys(),y.values())
fig1.autofmt_xdate()
plt.show()
w=sentiments(text)
fig,ax=plt.subplots()
ax.bar(w.keys(),w.values())
fig.autofmt_xdate()
plt.show()
