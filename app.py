import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Previsão de Alta ou Baixa - BTC (Machine Learning)")

# Coleta de dados
df = yf.download("BTC-USD", period="90d", interval="1h")

# Corrigir MultiIndex nas colunas
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Verificar se dados são válidos
if df.empty or 'Close' not in df.columns or df['Close'].notna().sum() < 15:
    st.error("Erro ao obter dados válidos do BTC. Tente novamente mais tarde.")
    st.stop()

# Cálculo dos indicadores técnicos
rsi = RSIIndicator(close=df['Close'], window=14)
df['RSI'] = rsi.rsi()
df['EMA'] = EMAIndicator(df['Close'], window=14).ema_indicator()

# Variável alvo (target)
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# Features e target
features = ['Close', 'RSI', 'EMA']
X = df[features]
y = df['Target']

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Treinamento do modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Acurácia
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Previsão futura
latest = df[features].iloc[[-1]]
future_pred = model.predict(latest)[0]
msg = "O modelo prevê: **ALTA**" if future_pred == 1 else "O modelo prevê: **BAIXA**"

# Exibição
st.write("Acurácia do modelo:", f"{acc * 100:.2f}%")
st.write(msg)
st.line_chart(df['Close'])
