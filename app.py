import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
df = pd.read_csv('C:\\Hari\\PROJECT\\Car Price  Project\\CAR DETAILS FROM CAR DEKHO.csv')

print("Step 1: Data Loaded! Shape:", df.shape)
df['Age'] = 2024 - df['year']

df.drop(['year', 'name'], axis=1, inplace=True)


df = pd.get_dummies(df, drop_first=True)


X = df.drop(['selling_price'], axis=1) 
y = df['selling_price']              

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Step 2: Training the Random Forest Model... Please wait.")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = metrics.r2_score(y_test, predictions)


print("\n--- RESULTS ---")
print(f"Model Accuracy (R2 Score): {accuracy * 100:.2f}%")
print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, predictions):.2f} Rupees")
print("\nModel is ready for use!")
