import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#1
df = pd.read_csv("house_data.csv")
print(df)
print("Thong ke mo ta:", df.describe())
missing = df.isnull().sum()
print(missing)
duplicate = df.duplicated().sum()
print("Duplicate:", duplicate)
df.hist(figsize=(10, 8))
plt.show()
sns.boxplot(data=df['price'])
plt.show()

#2
import pandas as pd
def clean_data(df):
    # fill missing
    df['price'].fillna(df['price'].median(), inplace=True)
    df['area'].fillna(df['area'].mean(), inplace=True)
    df = df[df['price'] > 0]
    df = df[df['rooms'] > 0]
    df = df.drop_duplicates()
    return df

#3
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

#4
from sklearn.preprocessing import MinMaxScaler
def scale_data(df):
    scaler = MinMaxScaler()
    df[['price', 'area']] = scaler.fit_transform(df[['price', 'area']])
    return df
df = pd.get_dummies(df, columns=['location'])

#5
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def find_duplicates(df):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(df['description'])
    sim_matrix = cosine_similarity(vectors)
    return sim_matrix
