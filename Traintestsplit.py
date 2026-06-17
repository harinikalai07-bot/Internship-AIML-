import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

df = df.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1)

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

print("Selected Features:")
print(X.head())

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

print("\nNormalized Data:")
print(X_normalized[:5])

standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)

print("\nStandardized Data:")
print(X_standardized[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X_standardized,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTarget Training Shape:")
print(y_train.shape)

print("\nTarget Testing Shape:")
print(y_test.shape)

print("\nProcess Completed Successfully!")