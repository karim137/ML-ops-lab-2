import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/data_raw.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = SVC(random_state=42, max_iter=1000)  # increase max_iter if needed
model.fit(X_scaled, y)
preds = model.predict(X_scaled)

# Save metrics
acc = accuracy_score(y, preds)
with open('metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f, indent=4)

# Generate and save confusion matrix plot
cm = confusion_matrix(y, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
