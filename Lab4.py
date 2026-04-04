#1
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
print("\n Bài 1:")
df = pd.read_csv("ITA105_Lab_4_Hotel_reviews.csv")
print(df)
df = df.dropna()
missing = df.isnull().sum()
print(missing)
le = LabelEncoder()
cols_encoder = ['hotel_name', 'customer_type']
for col in cols_encoder:
   df[col] = le.fit_transform(df[col].astype(str))
print(df)
stop_words = ["là", "và", "có", "rất", "thì", "một", "những", "các", "ở"]
def clean_text(text):
    text = text.lower()                          
    text = re.sub(r"[^\w\s]", "", text)          
    words = text.split()                        
    words = [w for w in words if w not in stop_words]  
    return words                               
df['words'] = df['review'].apply(clean_text)
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['review'].astype(str))
print(X_tfidf.shape)
model = Word2Vec(
    sentences=df['words'],
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)
print("\n Từ gần nghĩa với'sạch sẽ':", model.wv.most_similar("sạch sẽ", topn = 5))
print("\n --> Dùng TF-IDF khi cần đếm từ quan trọng," \
" dùng Word2Vec khi cần hiểu nghĩa của từ.")
#2
print("\n Bài 2:")
df = pd.read_csv("ITA105_Lab_4_Match_comments.csv")
print(df)
df = df.dropna()
le = LabelEncoder()
cols_encoder = ['team']
for col in cols_encoder:
   df[col] = le.fit_transform(df[col].astype(str))
print(df)
stop_words = ["là", "và", "có", "rất", "thì", "một", "những", "các", "ở"]
def clean_text(text):
   text = text.lower()
   text = re.sub(r"[^\w\s]", "", text)
   words = text.split()
   words = [w for w in words if w not in stop_words]
   return words
df['words'] = df['comment'].apply(clean_text)
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['comment'].astype(str))
print(X_tfidf.shape)
model = Word2Vec(
    sentences=df['words'],
    vector_size=50,
    min_count=1,
    workers=4
)
print("\n Từ gần nghĩa với 'xuất sắc':", model.wv.most_similar("xuất sắc", topn = 5))
print("\n --> TF-IDF chỉ đếm từ, Word2Vec hiểu nghĩa tốt hơn.")
#3
print("\n Bài 3:")
df = pd.read_csv("ITA105_Lab_4_Payer_feedback.csv")
print(df)
df = df.dropna()
le = LabelEncoder()
cols_encoder = ['player_type', 'device']
for col in cols_encoder:
   df[col] = le.fit_transform(df[col].astype(str))
print(df)
stop_words = ["là", "và", "có", "rất", "thì", "một", "những", "các", "ở"]
def clean_text(text):
   text = text.lower()
   text = re.sub(r"[^\w\s]", "", text)
   words = text.split()
   words = [w for w in words if w not in stop_words]
   return words
df['words'] = df['feedback_text'].apply(clean_text)
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['feedback_text'].astype(str))
print(X_tfidf.shape)
model = Word2Vec(
    sentences=df['words'],
    vector_size=50,
    min_count=1,
    workers=4
)
print("\n Từ gần nghĩa với 'đẹp':", model.wv.most_similar("đẹp", topn = 5))
print("\n --> Để phân loại cảm xúc nên chọn Word2Vec vì nó hiểu nghĩa tốt hơn.")
#4
print("\n Bài 4:")
df = pd.read_csv("ITA105_Lab_4_Album_reviews.csv")
print(df)
df = df.dropna()
le = LabelEncoder()
cols_encoder = ['genre', 'platform']
for col in cols_encoder:
   df[col] = le.fit_transform(df[col].astype(str))
print(df)
stop_words = ["là", "và", "có", "rất", "thì", "một", "những", "các", "ở"]
def clean_text(text):
   text = text.lower()
   text = re.sub(r"[^\w\s]", "", text)
   words = text.split()
   words = [w for w in words if w not in stop_words]
   return words
df['words'] = df['review_text'].apply(clean_text)
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['review_text'].astype(str))
print(X_tfidf.shape)
model = Word2Vec(
    sentences=df['words'],
    vector_size=50,
    min_count=1,
    workers=4
)
print("\n Từ gần nghĩa với 'sáng tạo':", model.wv.most_similar("sáng tạo", topn = 5))
print(""""\n --> Tổng kết: 
+ TF-IDF: Giống như đếm từ, từ xuất hiện nhiều = quan trọng.
+ Word2Vec: Hiểu nghĩa của từ, dùng để phân loại cảm xúc.
+ Chỉ cần đếm dùng TF-IDF, cần hiểu nghĩa hay phân loại cảm xúc dùng Word2Vec.""")
