import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/data_raw.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train model
model = LogisticRegression(random_state=42)
model.fit(X, y)
preds = model.predict(X)

# Save metrics
acc = accuracy_score(y, preds)
with open('metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f, indent=4)

# Generate and save confusion matrix plot
cm = confusion_matrix(y, preds, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
