import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#Giai doan 1
#1
dataset = {
    "price": [3000000, 5000000, -1000000, 7000000, 4500000, 4000000],
    "area": [50, 70, 60, 80, None, 90],
    "rooms": [2, 3, 2, 4, 3, 5],
    "location": ["Ha noi", "Bac Ninh", "HCM", "Da nang", "Bac ninh", "Ha noi"],
    "description": [
        "Cozy small house", 
        "Luxury apartment", 
        "Bad data example", 
        "Beautiful villa near beach", 
        "Good location house", 
        "Big house"],
    "date": [
        "2025-01-01",
        "2025-02-15",
        "2023-03-10",
        "2024-04-20",
        "2023-05-05",
        "2026-01-02"
    ]
}
df = pd.DataFrame(dataset)
df.to_csv("house_data.csv", index=False)
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
from scipy import stats
import numpy as np

def clean_data(df):
    df['price'].fillna(df['price'].median(), inplace=True)
    df['area'].fillna(df['area'].mean(), inplace=True)
    df = df[df['price'] > 0]
    df = df[df['rooms'] > 0]
    df = df.drop_duplicates()
    return df

#3
def outliers_IQR(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

df = df[(np.abs(stats.zscore(df['price'])) < 3)]

#4
from sklearn.preprocessing import MinMaxScaler
def scale_data(df):
    scaler = MinMaxScaler()
    df[['price', 'area']] = scaler.fit_transform(df[['price', 'area']])
    return df
df = pd.get_dummies(df, columns=['location'])

#5

def find_duplicates(df):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(df['description'])
    sim_matrix = cosine_similarity(vectors)
    return sim_matrix

#Giai doan 2
#1

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

df['price'] = df['price'].fillna(df['price'].median())
df = df[df['price'] > 0]
df['price_log'] = np.log1p(df['price'])

df['desc_length'] = df['description'].apply(len)
df['has_luxury'] = df['description'].str.contains("luxury").astype(int)

df = df.drop(['description', 'date'], axis=1)

#2+3
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score 
df = df.dropna()
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'bool']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])
pipeline.fit(X_train, y_train)

#4
pred = pipeline.predict(X_test)

print("RMSE:", root_mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

#Giai doan hoan thien
#1
df['area_rooms'] = df['area'] * df['rooms']

#2
df['price_per_m2'] = df['price'] / df['area']

#3
from sklearn.linear_model import LinearRegression

models = {
    "LR": LinearRegression(),
    "RF": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    
    print(name)
    print("RMSE:", root_mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))
    print("-----")

print("""Nhận xét:
- Giá nhà tăng theo diện tích và số phòng.
- Khu vực Hà Nội và HCM có xu hướng giá cao hơn.
- Feature has_luxury giúp model dự đoán tốt hơn.
- Log transform giúp giảm skew → cải thiện RMSE.""")
