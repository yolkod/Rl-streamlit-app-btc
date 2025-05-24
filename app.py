import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.title("Previsão de Alta ou Baixa - BTC (Machine Learning)")

# Coleta de dados
df = yf.download("BTC-USD", period="90d", interval="1h")

# Corrigir MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if df.empty or 'Close' not in df.columns or df['Close'].notna().sum() < 20:
    st.error("Erro ao obter dados do BTC.")
    st.stop()

# Cálculo de indicadores técnicos
df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
df['EMA'] = EMAIndicator(df['Close'], window=14).ema_indicator()
df['MACD'] = MACD(df['Close']).macd()
df['Return'] = df['Close'].pct_change()
df.dropna(inplace=True)

# Target: prever se o próximo candle será de alta
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# Features e normalização
features = ['Close', 'RSI', 'EMA', 'MACD', 'Return']
X = df[features]
y = df['Target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

# Treinamento
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Interface com botão
st.write("Acurácia do modelo:", f"{acc * 100:.2f}%")
if st.button("Prever próximo movimento"):
    latest = scaler.transform([df[features].iloc[-1]])
    future_pred = model.predict(latest)[0]
    msg = "O modelo prevê: **ALTA**" if future_pred == 1 else "O modelo prevê: **BAIXA**"
    st.write(msg)

# Gráfico
st.line_chart(df['Close'])
