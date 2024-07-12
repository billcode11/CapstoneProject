import streamlit as st
import pandas as pd
import itertools
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_and_process_data():
    dir = 'hungarian.data'
    with open(dir, encoding='Latin1') as file:
        lines = [line.strip() for line in file]

    data = itertools.takewhile(
        lambda x: len(x) == 76,
        (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
    )

    df = pd.DataFrame.from_records(data)
    df = df.iloc[:,:-1]
    df = df.drop(df.columns[0], axis=1)
    df = df.astype(float)
    df.replace(-9.0, np.nan, inplace=True)
    
    pilih_kolom = [
        2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57
    ]

    df_objek = df[pilih_kolom]
    df_objek.columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    df_objek = df_objek.astype(float)
    df_objek.drop(columns=['slope','ca','thal'], inplace=True)

    imputer = SimpleImputer(strategy='mean')
    df_objek = pd.DataFrame(imputer.fit_transform(df_objek), columns=df_objek.columns)

    X = df_objek.drop('target', axis=1)
    y = df_objek['target']

    oversample = SMOTE()
    Xs, ys = oversample.fit_resample(X, y)

    return Xs, ys

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree_parameter = {'max_depth': [None, 10, 20, 30, 40]}
    tree = GridSearchCV(DecisionTreeClassifier(), tree_parameter, cv=5)
    tree.fit(X_train, y_train)

    return tree

X, y = load_and_process_data()

model_tree = train_model(X, y)

# STREAMLIT
st.title("Prediksi Penyakit Jantung")
st.sidebar.header("Masukkan Data Pasien")

age = st.sidebar.number_input(
    "Usia", min_value=0, max_value=120, value=25)
sex = st.sidebar.selectbox(
    "Jenis Kelamin", ['Laki-laki', 'Perempuan'])
cp = st.sidebar.selectbox(
    "Tipe Nyeri Dada", ['Gejala biasa', 'Tidak bisa diprediksi', 'Gejala diluar penyakit jantung', 'Tanpa gejala'])
trestbps = st.sidebar.number_input(
    "Tekanan Darah Saat Istirahat", min_value=0, max_value=300, value=120)
chol = st.sidebar.number_input(
    "Kolesterol", min_value=0, max_value=600, value=200)
fbs = st.sidebar.selectbox(
    "Gula Darah Puasa > 120 mg/dl", ['>120 mg/dl', '<120 mg/dl'])
restecg = st.sidebar.selectbox(
    "Hasil EKG Istirahat", ['Normal', 'Kelainan ST-T', 'Kemungkinan Hypermetropi Vertikal Kiri'])
thalach = st.sidebar.number_input(
    "Detak Jantung Maksimum", min_value=0, max_value=220, value=150)
exang = st.sidebar.selectbox(
    "Angina yang Terinduksi oleh Latihan", ['Ya', 'Tidak'])
oldpeak = st.sidebar.number_input(
    "Depresi ST Dihasilkan oleh Latihan", min_value=0.0, max_value=10.0, value=1.0)

sex_map = {
    'Laki-laki': 1, 'Perempuan': 0}
cp_map = {
    'Gejala biasa': 0, 'Tidak bisa diprediksi': 1, 'Gejala diluar penyakit jantung': 2, 'Tanpa gejala': 3}
fbs_map = {
    '>120 mg/dl': 1, '<120 mg/dl': 0}
restecg_map = {
    'Normal': 0, 'Kelainan ST-T': 1, 'Kemungkinan Hypermetropi Vertikal Kiri': 2}
exang_map = {
    'Ya': 1, 'Tidak': 0}

sex = sex_map[sex]
cp = cp_map[cp]
fbs = fbs_map[fbs]
restecg = restecg_map[restecg]
exang = exang_map[exang]

# Prediksi
inputan_user = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])
prediksi = model_tree.predict(inputan_user)

st.subheader("Hasil Prediksi")
st.write(f"Prediksi Penyakit Jantung: {'Ada' if prediksi[0] == 1 else 'Tidak Ada'}")
