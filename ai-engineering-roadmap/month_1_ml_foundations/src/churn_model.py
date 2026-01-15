import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocessing
data = data.drop(['customerID'], axis=1)
for col in data.select_dtypes(include='object'):
    if col != 'Churn':
        data[col] = LabelEncoder().fit_transform(data[col])
data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})

# Train/test split
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
