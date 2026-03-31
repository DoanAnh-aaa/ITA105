#1
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#2
dataset = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Score": [2,3,4,5,6,7,8,9]
}
df = pd.DataFrame(dataset)
print(df)
#3
X = df["Hours"]
Y = df["Score"]
#4
model = LinearRegression()
#5
model.fit(X,Y)
#6
new_hours = pd.DataFrame([[6]], columns = ["Hours"])
predicted_score = model.predict(new_hours)
print("Predicted score:",predicted_score)
#7
new_data = pd.DataFrame([[3],[4],[6],[8],[9]], columns = ["Hours"])
predictions = model.predict(new_data)
print(predictions)
#8
plt.scatter(X,Y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("Hours vs Score")
plt.show()
#9
from sklearn.metrics import r2_score
y_read = model.predict(X)
score = r2_score(Y,y_read)
print("r2 score:", score)
