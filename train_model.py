import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#Load dataset
data  =  pd.read_csv("diabetes.csv")

#Features and target
X = data.drop("Outcome",axis=1)
y = data["Outcome"]

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Predict
y_pred =  model.predict(X_test)
y_proba  =model.predict_proba(X_test)[:, 1]

#Evaluate model
acc =  accuracy_score(y_test,y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy: {acc:.2f}")
print(f"AUC: {auc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Save model
joblib.dump(model, "diabetes_model.pkl")
print("Model saved as diabetes_model.pkl")

#plot accuracy bar
plt.figure(figsize=(10, 6))
sns.barplot(x=['Accuracy','ROC AUC'], y=[acc, auc])
plt.title('Model Metrics')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("metrics_plot_accuracy.png")
plt.close()

#plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize= (10, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("metrics_plot_roc.png")
plt.close()