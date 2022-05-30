import streamlit as st
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split



sc = StandardScaler()

st.title('DIABETES PREDICTION SYSTEM')
age = st.slider("Age in Years", 1, 150, 25, 1)
preg = st.slider("Number of Pregnancies", 0, 20, 0, 1)
gl = st.slider("Glucose Level", 0, 200, 25, 1)
skt = st.slider("Skin Thickness", 0, 99, 20, 1)
bp = st.slider('Blood Pressure', 0, 130, 69, 1)
ins = st.slider("Insulin", 0, 900, 79, 1)
bmi = st.slider("BMI", 0.000, 70.00, 31.00, 0.1)
dpf = st.slider("Diabetics Pedigree Function", 0.000, 3.00, 0.471, 0.001)
inp = (preg, gl, bp, skt, ins, bmi, dpf, age)

df = pd.read_csv('diabetes.csv')
#st.write(df.head())

# x_test1 = pickle.load(open('diabetes_test.pkl', 'rb'))
# x_train1 = pickle.load(open('diabetes_train.pkl', 'rb'))
# y_test1 = pickle.load(open('diabetes_y_test.pkl', 'rb'))
# y_train1 = pickle.load(open('diabetes_y_train.pkl', 'rb'))
# x_train = pd.DataFrame(x_train1)
# x_test = pd.DataFrame(x_test1)
# y_train = pd.DataFrame(y_train1)
# y_test = pd.DataFrame([y_test1])
# y_train =
# y_train = np.array(list(y_train1.items()))
# x_train = np.array(list(x_train1.items()))
x = df.iloc[:, :8]
y = df.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model_svm = SVC(kernel='linear', degree=3, gamma=1000)

model_svm.fit(x_train, y_train)
input_data = sc.transform(np.asarray(inp).reshape(1, -1))

result = model_svm.predict(input_data)

# st.write('sorry! you have diabetes')
if st.button('PREDICT'):

    if result[0] == 0:
        st.write("congo! you don't have diabetes")
    else:
        st.write('sorry! you have diabetes')
