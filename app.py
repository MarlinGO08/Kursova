import streamlit as st
import pandas as pd
from train import train_model
from predict import predict_price
from preprocessing import preprocess_ukr_dataset

# Налаштування сторінки
st.set_page_config(page_title="Прогноз нерухомості", layout="wide")
st.title("🏠 Інтелектуальна система прогнозування вартості нерухомості")

# Завантаження та попередня обробка даних
df = pd.read_csv("dataset.csv")
df_clean = preprocess_ukr_dataset(df)
st.caption(f"🔎 Розмір очищеного набору: {df_clean.shape[0]} рядків")

with st.expander("📋 Переглянути очищені дані"):
    st.dataframe(df_clean.head(10))

# Кнопка запуску навчання
if st.button("🔧 Навчити модель"):
    with st.spinner("Модель навчається, зачекайте..."):
        mae, mse, mae_fig, scatter_fig, pie_fig, X_train, X_test = train_model()

    # Зберігаємо в сесію
    st.session_state.mae = mae
    st.session_state.mse = mse
    st.session_state.mae_fig = mae_fig
    st.session_state.scatter_fig = scatter_fig
    st.session_state.pie_fig = pie_fig
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test

# Відображаємо результати навчання, якщо вони є в сесії
if "mae" in st.session_state:
    st.success(f"✅ Модель навчено: MAE = {st.session_state.mae:,.0f}, MSE = {st.session_state.mse:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(st.session_state.mae_fig, use_container_width=True)
    with col2:
        st.plotly_chart(st.session_state.scatter_fig, use_container_width=True)

    st.plotly_chart(st.session_state.pie_fig, use_container_width=True)

    st.subheader("📑 Частина тренувального набору")
    st.dataframe(st.session_state.X_train.head())

    st.subheader("📑 Частина тестового набору")
    st.dataframe(st.session_state.X_test.head())

st.markdown("---")

# 🔮 Прогноз вартості
st.header("🔮 Прогноз вартості нового об'єкта")

with st.form("predict_form"):
    area_m2 = st.number_input("Площа (в м²)", min_value=1.0, step=1.0)
    rooms = st.number_input("Кількість кімнат", min_value=0, step=1)
    floor = st.number_input("Поверх", min_value=0, step=1)
    city = st.text_input("Місто (наприклад, 'Київ')")

    metro = st.selectbox("Близько до метро", ["Так", "Ні"])
    renovated = st.selectbox("З ремонтом", ["Так", "Ні"])
    auction = st.selectbox("Торг можливий", ["Так", "Ні"])
    quiet = st.selectbox("Тиха зона", ["Так", "Ні"])
    furniture = st.selectbox("Меблі", ["Так", "Ні"])
    center = st.selectbox("Центр міста", ["Так", "Ні"])
    parking = st.selectbox("Паркінг", ["Так", "Ні"])

    submitted = st.form_submit_button("Спрогнозувати")

    if submitted and area_m2 and city:
        try:
            result = predict_price(
                area_m2=area_m2,
                rooms=rooms,
                floor=floor,
                city=city,
                metro=metro,
                renovated=renovated,
                auction=auction,
                quiet=quiet,
                furniture=furniture,
                center=center,
                parking=parking
            )
            st.success(f"💰 Орієнтовна вартість: {int(result):,} грн")
        except Exception as e:
            st.error(f"❌ Помилка прогнозу: {e}")
