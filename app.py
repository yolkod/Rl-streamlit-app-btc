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

if df.empty or len(df) < 50:
    st.error("Erro ao obter dados do BTC. Tente novamente mais tarde.")
    st.stop()

# Debug - mostrar colunas e tipos para entender melhor o dataframe
st.write(f"Colunas do df: {df.columns}")
st.write(f"Tipo de df['Close']: {type(df['Close'])}")
st.write(f"Total de valores não nulos em df['Close']: {df['Close'].notna().sum()}")

# Verificar existência e quantidade de dados válidos na coluna 'Close'
if 'Close' in df.columns:
    count_notna = df['Close'].notna().sum()
    if count_notna > 14:
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi()
    else:
        df['RSI'] = np.nan
else:
    df['RSI'] = np.nan

# Calcular EMA
df['EMA'] = EMAIndicator(df['Close'], window=14).ema_indicator()

# Preparar variável target para ML
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)  # Remove linhas com NaN

# Selecionar features e target
features = ['Close', 'RSI', 'EMA']
X = df[features]
y = df['Target']

# Dividir dados treino/teste (sem embaralhar)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Treinar modelo RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliar acurácia
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Previsão do próximo movimento
latest = df[features].iloc[[-1]]
future_pred = model.predict(latest)[0]
msg = "O modelo prevê: **ALTA**" if future_pred == 1 else "O modelo prevê: **BAIXA**"

# Mostrar resultados no Streamlit
st.write("Acurácia do modelo:", f"{acc * 100:.2f}%")
st.write(msg)
st.line_chart(df['Close'])
