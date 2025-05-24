import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np  # <-- Importa numpy para usar np.nan
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Previsão de Alta ou Baixa - BTC (Machine Learning)")

# Coleta de dados com verificação
df = yf.download("BTC-USD", period="90d", interval="1h")

if df.empty or len(df) < 50:
    st.error("Erro ao obter dados do BTC. Tente novamente mais tarde.")
    st.stop()

# Verifica se a coluna 'Close' tem dados válidos
if 'Close' in df.columns and df['Close'].notna().sum() > 14:
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
else:
    df['RSI'] = np.nan  # precisa do import numpy

# Calcula EMA
df['EMA'] = EMAIndicator(df['Close'], window=14).ema_indicator()

# Preparar variáveis para ML
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)  # remove linhas com NaN

# Define features e target
features = ['Close', 'RSI', 'EMA']
X = df[features]
y = df['Target']

# Divide dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Treina modelo RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsão e acurácia
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Previsão do próximo movimento
latest = df[features].iloc[[-1]]
future_pred = model.predict(latest)[0]
msg = "O modelo prevê: **ALTA**" if future_pred == 1 else "O modelo prevê: **BAIXA**"

# Exibir resultados no Streamlit
st.write("Acurácia do modelo:", f"{acc * 100:.2f}%")
st.write(msg)
st.line_chart(df['Close'])
