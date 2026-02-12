import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

st.title("🚢 Titanic Survival Prediction App")
st.write("Train and compare multiple ML models on Titanic dataset")

# Load dataset
df = sns.load_dataset("titanic")

# Preprocessing
df.drop(["deck","embark_town","class","who","adult_male","alive"], axis=1, inplace=True)

df['age'].fillna(df['age'].mean(), inplace=True)
df.dropna(subset=['embarked'], inplace=True)

df = pd.get_dummies(df, columns=['embarked'])

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])

df['age'] = df['age'].round(2)
df = df.astype(int)

# Feature / Target split
X = df.drop("survived", axis=1)
y = df["survived"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Sidebar model selection
model_option = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "KNN", "Naive Bayes", "Decision Tree", "SVM"]
)

# Train model based on selection
if model_option == "Logistic Regression":
    model = LogisticRegression()

elif model_option == "KNN":
    k = st.sidebar.slider("Select K value", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)

elif model_option == "Naive Bayes":
    model = GaussianNB()

elif model_option == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)

elif model_option == "SVM":
    model = SVC(kernel="rbf")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.4f}")

with col2:
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    st.metric("Cross Validation Accuracy (Avg)", f"{cv_scores.mean():.4f}")

st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(report)

st.subheader("Dataset Preview")
st.dataframe(df.head())
