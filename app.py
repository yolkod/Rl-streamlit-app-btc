import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

st.set_page_config(layout="centered")
st.title("Previsão BTC +1% nas próximas 3h (Otimizado)")

# Cache para acelerar o download de dados
@st.cache_data(ttl=3600)
def carregar_dados():
    df = yf.download("BTC-USD", period="60d", interval="1h")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

# Cache do modelo e dados processados
@st.cache_resource
def treinar_modelo(df):
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['EMA'] = EMAIndicator(df['Close']).ema_indicator()
    df['MACD'] = MACD(df['Close']).macd()
    df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    df['Return'] = df['Close'].pct_change()
    df['Vol'] = df['Return'].rolling(window=5).std()
    df['Future_Close'] = df['Close'].shift(-3)
    df['Target'] = (df['Future_Close'] > df['Close'] * 1.01).astype(int)
    df.dropna(inplace=True)

    features = ['Close', 'RSI', 'EMA', 'MACD', 'ADX', 'Return', 'Vol', 'Volume']
    X = df[features]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

    model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, df, features, acc

# Rodar tudo
with st.spinner("Carregando e treinando modelo..."):
    df = carregar_dados()
    model, scaler, df_proc, features, acc = treinar_modelo(df)

# Exibir acurácia e gráfico
st.write("Acurácia do modelo:", f"{acc * 100:.2f}%")
st.line_chart(df_proc['Close'])

# Previsão sob demanda
if st.button("Prever próximo movimento"):
    latest = scaler.transform([df_proc[features].iloc[-1]])
    pred = model.predict(latest)[0]
    if pred == 1:
        st.success("O modelo prevê: **ALTA acima de 1% nas próximas 3 horas**")
    else:
        st.error("O modelo prevê: **BAIXA ou variação inferior a 1%**")
