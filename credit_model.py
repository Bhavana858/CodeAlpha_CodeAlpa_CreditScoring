import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'income': [50000, 60000, 20000, 80000, 30000],
    'debt': [10000, 20000, 5000, 30000, 10000],
    'payment_history': [1, 1, 0, 1, 0],
    'creditworthy': [1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Input and output
X = df[['income', 'debt', 'payment_history']]
y = df['creditworthy']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# Test new customer
new_customer = pd.DataFrame([[40000, 15000, 1]], 
                            columns=['income', 'debt', 'payment_history'])

print("New Customer Prediction:", model.predict(new_customer))