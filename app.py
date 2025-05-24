
import streamlit as st
import yfinance as yf
import pandas as pd
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

df['RSI'] = RSIIndicator(df['Close']).rsi()
df['EMA'] = EMAIndicator(df['Close'], window=14).ema_indicator()

# Preparar variáveis
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# Dividir dados
features = ['Close', 'RSI', 'EMA']
X = df[features]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Previsão do próximo movimento
latest = df[features].iloc[[-1]]
future_pred = model.predict(latest)[0]
msg = "O modelo prevê: **ALTA**" if future_pred == 1 else "O modelo prevê: **BAIXA**"

# Exibir resultados
st.write("Acurácia do modelo:", f"{acc * 100:.2f}%")
st.write(msg)
st.line_chart(df['Close'])
