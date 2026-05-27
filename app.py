import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@st.cache_resource
def load_model():
    df = pd.read_csv('data/weather_type_classification.csv')
    le_cloud = LabelEncoder()
    le_season = LabelEncoder()
    le_location = LabelEncoder()
    le_target = LabelEncoder()
    df['Cloud Cover'] = le_cloud.fit_transform(df['Cloud Cover'])
    df['Season'] = le_season.fit_transform(df['Season'])
    df['Location'] = le_location.fit_transform(df['Location'])
    df['target'] = le_target.fit_transform(df['Weather Type'])
    X = df.drop(['Weather Type', 'target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, scaler, le_target

model, scaler, le_target = load_model()

st.title("Классификация погодных явлений")
st.write("Введите параметры погоды:")

temp = st.slider("Температура (C)", -25, 50, 20)
humidity = st.slider("Влажность (%)", 20, 109, 60)
wind = st.slider("Скорость ветра (км/ч)", 0, 50, 10)
precip = st.slider("Осадки (%)", 0, 109, 50)
pressure = st.slider("Давление (гПа)", 800, 1200, 1010)
uv = st.slider("UV индекс", 0, 14, 4)
visibility = st.slider("Видимость (км)", 0, 20, 5)
cloud = st.selectbox("Облачность", ['clear', 'overcast', 'partly cloudy', 'cloudy'])
season = st.selectbox("Сезон", ['Winter', 'Spring', 'Summer', 'Autumn'])
location = st.selectbox("Местность", ['inland', 'mountain', 'coastal'])

cloud_map = {'clear': 0, 'cloudy': 1, 'overcast': 2, 'partly cloudy': 3}
season_map = {'Autumn': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
location_map = {'coastal': 0, 'inland': 1, 'mountain': 2}

if st.button("Определить погоду"):
    X = np.array([[temp, humidity, wind, precip, cloud_map[cloud], pressure, uv, season_map[season], visibility, location_map[location]]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    label = le_target.inverse_transform(pred)[0]
    icons = {'Rainy': 'Дождь', 'Sunny': 'Солнечно', 'Cloudy': 'Облачно', 'Snowy': 'Снег'}
    st.success(f"Тип погоды: {icons.get(label, label)}")