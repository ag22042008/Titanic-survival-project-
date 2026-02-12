import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="Titanic ML Dashboard", layout="wide")

st.title("🚢 Titanic Survival Prediction – ML Dashboard")
st.markdown("End-to-End Machine Learning Model Comparison & Deployment Interface")

# ================= LOAD DATA =================

@st.cache_data
def load_data():
    df = sns.load_dataset("titanic")
    df.drop(["deck","embark_town","class","who","adult_male","alive"], axis=1, inplace=True)
    df['age'].fillna(df['age'].mean(), inplace=True)
    df.dropna(subset=['embarked'], inplace=True)
    df = pd.get_dummies(df, columns=['embarked'])
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['age'] = df['age'].round(2)
    df = df.astype(int)
    return df

df = load_data()

X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODEL SELECTION =================

model_option = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "KNN", "Naive Bayes", "Decision Tree", "SVM"]
)

if model_option == "Logistic Regression":
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])

elif model_option == "KNN":
    k = st.sidebar.slider("Select K value", 1, 15, 5)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=k))
    ])

elif model_option == "Naive Bayes":
    model = GaussianNB()

elif model_option == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)

elif model_option == "SVM":
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ================= METRICS =================

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.4f}")

with col2:
    cv_scores = cross_val_score(model, X, y, cv=5)
    st.metric("Cross Validation Accuracy", f"{cv_scores.mean():.4f}")

# ================= CONFUSION MATRIX =================

st.subheader("Confusion Matrix")
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
st.pyplot(fig1)

# ================= ROC CURVE =================

if hasattr(model, "predict_proba"):
    st.subheader("ROC Curve")
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr)
    ax2.plot([0,1], [0,1])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")
    st.pyplot(fig2)

# ================= FEATURE IMPORTANCE =================

if model_option == "Decision Tree":
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig3, ax3 = plt.subplots()
    ax3.barh(feature_df["Feature"], feature_df["Importance"])
    ax3.invert_yaxis()
    st.pyplot(fig3)

# ================= MANUAL PREDICTION =================

st.subheader("Manual Passenger Survival Prediction")

pclass = st.selectbox("Passenger Class", [1,2,3])
sex_input = st.selectbox("Sex", ["Male","Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.slider("Siblings/Spouse", 0, 5, 0)
parch = st.slider("Parents/Children", 0, 5, 0)
fare = st.slider("Fare", 0, 500, 50)
embarked = st.selectbox("Embarked", ["C","Q","S"])

sex = 1 if sex_input == "Male" else 0
emb_C = 1 if embarked == "C" else 0
emb_Q = 1 if embarked == "Q" else 0
emb_S = 1 if embarked == "S" else 0

if st.button("Predict Survival"):

    input_df = pd.DataFrame({
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "embarked_C": [emb_C],
        "embarked_Q": [emb_Q],
        "embarked_S": [emb_S],
    })

    try:
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.success("Passenger is likely to SURVIVE")
        else:
            st.error("Passenger is likely to NOT SURVIVE")

    except Exception as e:
        st.error("Prediction failed. Please check input structure.")

# ================= REPORT =================

st.subheader("Classification Report")
st.text(report)

st.markdown("---")
st.markdown("Developed by Aditya Gupta – End-to-End ML Deployment Project")
