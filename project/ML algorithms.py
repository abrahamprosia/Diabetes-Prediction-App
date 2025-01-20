import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target
from scipy.stats import norm
import joblib




# Load dataset
df = pd.read_csv("D:\\Datasets\\diabetes_prediction_dataset.csv")

# Label encoding for smoking history
categ_mapping = {
    "No Info": 0,
    "never": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5
}

#Make a function to extract and can use later
def get_mapped_value(category, mapping=categ_mapping):
    return mapping.get(category, None)
df["smoking_history"] = df["smoking_history"].map(categ_mapping)

# Label encoding for gender
df["gender"] = df["gender"].map({"Male": 0, "Female": 1, "Other": 2})

# Split features and target
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build Models

# logistic regression model
logreg = LogisticRegression()
logreg_model = logreg.fit(X_train, y_train)

#KNN model
KNN_model = KNeighborsClassifier(n_neighbors= 9, weights= 'uniform')
KNN_model.fit(X_train, y_train)

#SVM model
svm_model = SVC(kernel='linear', C =  0.5)
svm_model.fit(X_train, y_train)

#Random Forest Model
rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)



# Make predictions
logreg_model_pred = logreg_model.predict(X_test)
KNN_model_pred = KNN_model.predict(X_test)
svm_model_pred = svm_model.predict(X_test)
rf_model_pred = rf_model.predict(X_test)

#I will save the models and scaler
joblib.dump(logreg_model, "logistic_regression_model.pkl")
joblib.dump(KNN_model, "KNN_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Evaluate model performance
logreg_acc = accuracy_score(y_test, logreg_model_pred)
KNN_acc = accuracy_score(y_test, KNN_model_pred)
svm_acc = accuracy_score(y_test, svm_model_pred)
rf_acc = accuracy_score(y_test, rf_model_pred)

#calculation of confidence and margin of error
def calculate_confidence(predictions, probabilities, confidence_level = 0.99):
    z_score = norm.ppf((1 + confidence_level)/2)
    margin_of_error = z_score * np.sqrt((probabilities*(1-probabilities))/len(probabilities))
    return margin_of_error

models = {
    "Logistic Regression" : logreg_model,
    "KNN": KNN_model,
    "SVM": svm_model,
    "Random Forest": rf_model,
}

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:,1]
        preds = model.predict(X_test)
        margin_of_error = calculate_confidence(preds, probs)
        print(f"{name}")
        print(f" margin of Error (99%): {margin_of_error.mean():.4f}")
        print(f" Confidence Level (99%): ±{margin_of_error.mean():.4f}")
    else:
        print(f"{name}: Model does not support probability-based confidence intervals.")



print("Models and scaler saved successfully.")
print(f"Logistic Regression Model Accuracy: {logreg_acc:.2f}")
print(f"KNN Model Model Accuracy: {KNN_acc:.2f}")
print(f"SVM Model Model Accuracy: {svm_acc:.2f}")
print(f"Random Forest Model Model Accuracy: {rf_acc:.2f}")



