

import streamlit as st import yfinance as yf import pandas as pd import numpy as np

from ta.momentum import RSIIndicator, StochasticOscillator from ta.trend import EMAIndicator, MACD, CCIIndicator from ta.volatility import BollingerBands from ta.volume import OnBalanceVolumeIndicator

from sklearn.preprocessing import StandardScaler from sklearn.model_selection import train_test_split from sklearn.metrics import classification_report, accuracy_score

from xgboost import XGBClassifier

st.set_page_config(layout="centered") st.title("Previsão de Alta, Estabilidade ou Baixa do BTC (Modelo Aprimorado)")

@st.cache_data(ttl=3600) def carregar_dados(): df = yf.download("BTC-USD", period="90d", interval="1h") if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0) return df.dropna()

def criar_variaveis(df): df['RSI'] = RSIIndicator(df['Close']).rsi() df['EMA'] = EMAIndicator(df['Close']).ema_indicator() df['MACD'] = MACD(df['Close']).macd() df['CCI'] = CCIIndicator(df['High'], df['Low'], df['Close']).cci() df['BB_high'] = BollingerBands(df['Close']).bollinger_hband() df['BB_low'] = BollingerBands(df['Close']).bollinger_lband() df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume() df['STOCH_RSI'] = StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()

df['Return'] = df['Close'].pct_change()
df['Vol'] = df['Return'].rolling(window=5).std()

df['Close>EMA'] = (df['Close'] > df['EMA']).astype(int)
df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
return df

def gerar_target(df, horas_futuro): df['Future_Close'] = df['Close'].shift(-horas_futuro) df['Pct_Future'] = (df['Future_Close'] - df['Close']) / df['Close'] df['Target'] = df['Pct_Future'].apply(lambda x: 2 if x > 0.01 else (1 if x < -0.01 else 0)) return df

def treinar_modelo(df, horas_futuro): df = criar_variaveis(df) df = gerar_target(df, horas_futuro) df.dropna(inplace=True)

features = ['Close', 'RSI', 'EMA', 'MACD', 'CCI', 'BB_high', 'BB_low',
            'OBV', 'STOCH_RSI', 'Return', 'Vol', 'Close>EMA',
            'RSI_overbought', 'RSI_oversold', 'Volume']

X = df[features]
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.02,
                      subsample=0.85, colsample_bytree=0.85,
                      use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

return model, scaler, df, features, acc, classification_report(y_test, y_pred, output_dict=True)

Interface de seleção de horizonte

horas_opcao = st.selectbox("Selecione o horizonte da previsão:", [3, 6, 12, 24], index=0, format_func=lambda x: f"{x} horas")

with st.spinner("Carregando e treinando modelo melhorado..."): df = carregar_dados() model, scaler, df_proc, features, acc, relatorio = treinar_modelo(df, horas_opcao)

st.write("Acurácia do modelo:", f"{acc * 100:.2f}%") st.write("Distribuição de classes previstas:") st.json({ "Baixa": relatorio['0']['precision'], "Estável": relatorio['1']['precision'], "Alta": relatorio['2']['precision'] }) st.line_chart(df_proc['Close'])

if st.button("Prever próximo movimento"): latest = scaler.transform([df_proc[features].iloc[-1]]) pred = model.predict(latest)[0] if pred == 2: st.success(f"Previsão para {horas_opcao}h: ALTA > 1%") elif pred == 0: st.error(f"Previsão para {horas_opcao}h: BAIXA > 1%") else: st.info(f"Previsão para {horas_opcao}h: MERCADO ESTÁVEL")

