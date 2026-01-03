from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load the wine dataset
data = load_wine()

# Create the train, val split at an 80/20 split
X_train, X_test, y_train, y_test =train_test_split(data['data'], data['target'], test_size=0.2)

# Create model
model = xgb.XGBClassifier(n_estimators=4, max_depth=2, learning_rate=1, objective='multi:softprob')

# Train model
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Print model accuracy
accuracy = accuracy_score(y_test, preds)
print(f"Model Accuracy: {accuracy}")

# Load saved model
# model = xgb.XGBClassifier()
# model.load_model("models/wine_xgboost_model.json")

