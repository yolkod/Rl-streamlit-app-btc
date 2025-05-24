import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

st.title("Previsão de Alta +1% nas Próximas 3h (BTC)")

# Coleta de dados
df = yf.download("BTC-USD", period="90d", interval="1h")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
if df.empty or 'Close' not in df.columns or df['Close'].isna().sum() > 0:
    st.error("Erro ao obter dados do BTC.")
    st.stop()

# Indicadores técnicos
df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
df['EMA'] = EMAIndicator(df['Close'], window=14).ema_indicator()
df['MACD'] = MACD(df['Close']).macd()
df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()
df['Return'] = df['Close'].pct_change()
df['Volatilidade'] = df['Return'].rolling(window=5).std()
df['Volume'] = df['Volume']

# Novo alvo: prever se vai subir +1% nas próximas 3 horas
df['Future_Close'] = df['Close'].shift(-3)
df['Target'] = (df['Future_Close'] > df['Close'] * 1.01).astype(int)
df.dropna(inplace=True)

# Features
features = ['Close', 'RSI', 'EMA', 'MACD', 'ADX', 'Return', 'Volatilidade', 'Volume']
X = df[features]
y = df['Target']

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

# Modelo XGBoost
model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Interface
st.write("Acurácia do modelo:", f"{acc * 100:.2f}%")
if st.button("Prever próximo movimento"):
    latest = scaler.transform([df[features].iloc[-1]])
    pred = model.predict(latest)[0]
    msg = "O modelo prevê: **ALTA acima de 1% nas próximas 3 horas**" if pred == 1 else "O modelo prevê: **BAIXA ou variação < 1%**"
    st.write(msg)

# Gráfico
st.line_chart(df['Close'])
