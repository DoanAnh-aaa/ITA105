import pandas as pd
import matplotlib.pyplot as plt
#1
df = pd.read_csv("ITA105_Lab_5_Supermarket.csv")
print(df)
df ['date'] = pd.to_datetime(df['date'], format = 'mixed', dayfirst = True, errors = 'coerce') 
df = df.set_index("date")
print("Sau khi chuan hoa :", df)
missing = df.ffill()
df_missing = pd.DataFrame({
    "original" : df['revenue'],
    "ffill" : missing['revenue']
})
print(df_missing)

df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

monthly = df['revenue'].resample('ME').sum()
monthly.plot()
plt.title("Doanh thu theo tháng")
plt.xlabel("Thời gian")
plt.ylabel("Doanh thu")
plt.grid()
plt.show()

df['rolling_mean'] = df['revenue'].rolling(7).mean()
df[['revenue', 'rolling_mean']].plot()
plt.show()

#2
df = pd.read_csv("ITA105_Lab_5_Web_traffic.csv")
print(df)
df ['datetime'] = pd.to_datetime(df['datetime'], format = 'mixed', dayfirst = True, errors = 'coerce') 
df = df.set_index("datetime")
print("Sau khi chuan hoa:", df)

df = df.resample('h').mean()

df = df.interpolate(method='linear')

df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

hourly = df.groupby('hour')['visits'].mean()
hourly.plot()
plt.title("Lưu lượng theo giờ")
plt.xlabel("Giờ")
plt.ylabel("Lượt truy cập")
plt.grid()
plt.show()
weekly = df.groupby('dayofweek')['visits'].mean()
weekly.plot()
plt.title("Lưu lượng theo ngày trong tuần")
plt.xlabel("Ngày (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun)")
plt.ylabel("Lượt truy cập")
plt.grid()
plt.show() 

print("""\n Nhận xét:
      - Lưu lượng truy cập thay đổi theo từng giờ trong ngày, các khung giờ cao điểm thường rơi vào buổi tối (18h-21h).
      - Lưu lượng truy cập có sự khác biệt giữa các ngày trong tuần, các ngày thứ 2 và thứ 4 có lượng truy cập cao hơn các ngày.
""")

#3
df = pd.read_csv("ITA105_Lab_5_Stock.csv")
print(df)

df ['date'] = pd.to_datetime(df['date'], format = 'mixed', dayfirst = True, errors = 'coerce') 
df = df.set_index("date")
print("Sau khi chuan hoa :", df)
missing = df.ffill()
df_missing = pd.DataFrame({
    "original" : df['close_price'],
    "ffill" : missing['close_price']
})
print(df_missing)

df['close_price'].plot()
plt.title("Giá đóng cửa")
plt.grid()
plt.show()

df['MA7'] = df['close_price'].rolling(7).mean()
df['MA30'] = df['close_price'].rolling(30).mean()
df[['close_price','MA7','MA30']].plot()
plt.grid()
plt.show()

monthly = df.groupby(df.index.month)['close_price'].mean()
monthly.plot()
plt.title("Seasonality theo tháng")
plt.grid()
plt.show()

#4
df = pd.read_csv("ITA105_Lab_5_Production.csv")
print(df)
df ['week_start'] = pd.to_datetime(df['week_start'], format = 'mixed', dayfirst = True, errors = 'coerce') 
df = df.set_index("week_start")
print("Sau khi chuan hoa :", df)
missing = df.ffill()
df_missing = pd.DataFrame({
    "original" : df['production'],
    "ffill" : missing['production']
})
print(df_missing)

df['year'] = df.index.year
df['quarter'] = df.index.quarter
df['week'] = df.index.isocalendar().week

df['rolling_mean'] = df['production'].rolling(7).mean()
df[['production','rolling_mean']].plot()
plt.grid()
plt.show()

quarter = df.groupby('quarter')['production'].mean()
quarter.plot()
plt.title("Seasonality theo quý")
plt.grid()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
df['production'] = df['production'].ffill().bfill()
result = seasonal_decompose(df['production'], model='additive', period=12)
result.plot()
plt.grid()
plt.show()
