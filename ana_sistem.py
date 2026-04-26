import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import smtplib
import warnings
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
warnings.filterwarnings('ignore')

# ─── Ayarlar ───────────────────────────────────────────
GMAIL_ADRES  = "tahir.aytekin72@gmail.com"
GMAIL_SIFRE  = "esky lnix cnil sfke"
ALICI_ADRES  = "tahir.aytekin72@gmail.com"

HISSELER = [
    "AKBNK.IS", "GARAN.IS", "YKBNK.IS", "EKGYO.IS",
    "PGSUS.IS", "TCELL.IS", "SISE.IS",  "FROTO.IS",
    "THYAO.IS", "EREGL.IS", "ASELS.IS", "TUPRS.IS"
]
PERIYOT           = "5y"
BASLANGIC_SERMAYE = 100000
KOMISYON          = 0.001
STOP_LOSS         = 0.07
# ───────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════
# MODÜL 1 — PİYASA REJİMİ
# ══════════════════════════════════════════════════════
def piyasa_rejimi_tespit():
    print("\n[1/5] Piyasa rejimi analiz ediliyor...")
    try:
        bist   = yf.Ticker("XU100.IS").history(period="6mo", interval="1d")
        usdtry = yf.Ticker("USDTRY=X").history(period="6mo", interval="1d")
        bist.index   = bist.index.tz_localize(None)
        usdtry.index = usdtry.index.tz_localize(None)

        if len(bist) < 25 or len(usdtry) < 5:
            raise ValueError("Yetersiz veri")

        bist['MA20']       = ta.sma(bist['Close'], length=20)
        bist['MA50']       = ta.sma(bist['Close'], length=50)
        bist['RSI']        = ta.rsi(bist['Close'], length=14)
        bist['Getiri_1ay'] = bist['Close'].pct_change(20)
        bist_temiz         = bist.dropna()
        son                = bist_temiz.iloc[-1]

        idx       = min(20, len(usdtry) - 1)
        kur_son   = float(usdtry['Close'].iloc[-1])
        kur_eski  = float(usdtry['Close'].iloc[-idx])
        kur_deg   = (kur_son - kur_eski) / kur_eski * 100

        puanlar = {}
        puanlar['trend']    = 2 if son['Close'] > son['MA20'] > son['MA50'] else (
                              1 if son['Close'] > son['MA50'] else -1)
        puanlar['rsi']      = 1 if son['RSI'] > 60 else (-1 if son['RSI'] < 40 else 0)
        puanlar['momentum'] = 2 if son['Getiri_1ay'] > 0.05 else (
                              1 if son['Getiri_1ay'] > 0 else -1)
        puanlar['kur']      = -2 if kur_deg > 5 else (-1 if kur_deg > 2 else 0)

        toplam = sum(puanlar.values())

        if toplam >= 4:
            rejim, emoji, carpani = "GUCLU_BOGA", "🚀", 1.3
        elif toplam >= 2:
            rejim, emoji, carpani = "BOGA",       "🟢", 1.1
        elif toplam >= -1:
            rejim, emoji, carpani = "YATAY",      "🟡", 0.9
        elif toplam >= -3:
            rejim, emoji, carpani = "AYI",        "🔴", 0.7
        else:
            rejim, emoji, carpani = "GUCLU_AYI",  "❌", 0.5

        print(f"    {emoji} {rejim} (Puan: {toplam:+d}) | "
              f"BIST: {son['Close']:,.0f} | "
              f"RSI: {son['RSI']:.1f} | "
              f"USD/TRY: {kur_son:.2f}")

        return {'rejim': rejim, 'emoji': emoji, 'carpani': carpani,
                'toplam': toplam, 'bist_son': float(son['Close']),
                'bist_rsi': float(son['RSI']), 'usdtry': kur_son,
                'getiri_1ay': float(son['Getiri_1ay'] * 100)}

    except Exception as e:
        print(f"    Hata: {e} — YATAY varsayıldı")
        return {'rejim': 'YATAY', 'emoji': '🟡', 'carpani': 0.9,
                'toplam': 0, 'bist_son': 0, 'bist_rsi': 50,
                'usdtry': 0, 'getiri_1ay': 0}

# ══════════════════════════════════════════════════════
# MODÜL 2 — TEMEL ANALİZ FİLTRESİ
# ══════════════════════════════════════════════════════
def temel_skor_hesapla(sembol):
    try:
        info = yf.Ticker(sembol).info
        def al(k):
            v = info.get(k)
            return float(v) if v and v != 'N/A' else None

        puan = 0
        fk   = al('trailingPE')
        if fk:
            puan += 2 if fk < 8 else (1 if fk < 15 else (0 if fk < 25 else -1))

        pddd = al('priceToBook')
        if pddd:
            puan += 2 if pddd < 1 else (1 if pddd < 2 else (0 if pddd < 4 else -1))

        roe = al('returnOnEquity')
        if roe:
            puan += 2 if roe > 0.30 else (1 if roe > 0.15 else (0 if roe > 0.05 else -1))

        kar = al('profitMargins')
        if kar:
            puan += 2 if kar > 0.20 else (1 if kar > 0.10 else (0 if kar > 0 else -1))

        buy = al('revenueGrowth')
        if buy:
            puan += 2 if buy > 0.30 else (1 if buy > 0.10 else 0)

        return puan
    except:
        return 0

def temel_filtre(hisseler):
    print("\n[2/5] Temel analiz filtresi uygulanıyor...")
    secilen = []
    for s in hisseler:
        skor  = temel_skor_hesapla(s)
        durum = "✅" if skor >= 0 else "❌"
        print(f"    {s:<12} {durum} Skor: {skor:+d}")
        if skor >= 0:
            secilen.append((s, skor))
    secilen = sorted(secilen, key=lambda x: x[1], reverse=True)
    print(f"    → {len(secilen)} hisse ML analizine gönderiliyor")
    return [s for s, _ in secilen]

# ══════════════════════════════════════════════════════
# MODÜL 3 — ML SİNYAL (RF + XGBoost + LightGBM)
# ══════════════════════════════════════════════════════
def ozellikler_ekle_ml(df):
    df = df.copy()
    df['RSI']         = ta.rsi(df['Close'], length=14)
    df['RSI_fast']    = ta.rsi(df['Close'], length=7)
    macd              = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_sinyal'] = macd['MACDs_12_26_9']
    df['MACD_hist']   = macd['MACDh_12_26_9']
    df['MA5']         = ta.sma(df['Close'], length=5)
    df['MA20']        = ta.sma(df['Close'], length=20)
    df['MA50']        = ta.sma(df['Close'], length=50)
    df['MA200']       = ta.sma(df['Close'], length=200)
    df['EMA20']       = ta.ema(df['Close'], length=20)
    bb                = ta.bbands(df['Close'], length=20, std=2)
    bb_s              = bb.columns.tolist()
    df['BB_ust']      = bb[bb_s[2]]
    df['BB_alt']      = bb[bb_s[0]]
    df['BB_genislik'] = (df['BB_ust'] - bb[bb_s[1]]) / bb[bb_s[1]]
    df['ATR']         = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Volatilite']  = df['Close'].pct_change().rolling(20).std()
    df['Hacim_MA20']  = ta.sma(df['Volume'], length=20)
    df['Hacim_Oran']  = df['Volume'] / df['Hacim_MA20']
    df['Hacim_Trend'] = df['Hacim_Oran'].rolling(5).mean()
    df['RSI_Norm']    = (df['RSI'] - 50) / 50
    df['RSI_f_Norm']  = (df['RSI_fast'] - 50) / 50
    df['BB_Konum']    = (df['Close'] - df['BB_alt']) / (df['BB_ust'] - df['BB_alt'])
    df['MA5_Fark']    = (df['Close'] - df['MA5'])   / df['MA5']
    df['MA20_Fark']   = (df['Close'] - df['MA20'])  / df['MA20']
    df['MA50_Fark']   = (df['Close'] - df['MA50'])  / df['MA50']
    df['MA200_Fark']  = (df['Close'] - df['MA200']) / df['MA200']
    df['MACD_Norm']   = df['MACD'] - df['MACD_sinyal']
    df['Trend_Guc']   = (df['MA5'] - df['MA50'])   / df['MA50']
    df['EMA_MA_Fark'] = (df['EMA20'] - df['MA20']) / df['MA20']
    for g in [1, 3, 5, 10, 20]:
        df[f'Getiri_{g}g'] = df['Close'].pct_change(g)
    df['Kanat']       = (df['High'] - df['Low']) / df['Close']
    df['Govde']       = abs(df['Close'] - df['Open']) / df['Close']
    df['Yon']         = np.where(df['Close'] > df['Open'], 1.0, -1.0)
    df['52H_Yuzde']   = df['Close'] / df['Close'].rolling(252).max()
    df['52L_Yuzde']   = df['Close'] / df['Close'].rolling(252).min()
    df['RSI_Trend']   = df['RSI'] - df['RSI'].shift(5)
    df['Hacim_Fiyat'] = df['Getiri_1g'] * df['Hacim_Oran']
    return df.dropna()

ML_OZELLIKLER = [
    'RSI_Norm', 'RSI_f_Norm', 'MACD_Norm', 'MACD_hist',
    'BB_Konum', 'BB_genislik', 'MA5_Fark', 'MA20_Fark',
    'MA50_Fark', 'MA200_Fark', 'Trend_Guc', 'EMA_MA_Fark',
    'Getiri_1g', 'Getiri_3g', 'Getiri_5g', 'Getiri_10g', 'Getiri_20g',
    'Hacim_Oran', 'Hacim_Trend', 'ATR', 'Volatilite',
    'Kanat', 'Govde', 'Yon', '52H_Yuzde', '52L_Yuzde',
    'RSI_Trend', 'Hacim_Fiyat'
]

def ml_model_egit(sembol):
    df = yf.Ticker(sembol).history(period=PERIYOT, interval="1d")
    df = df[['Open','High','Low','Close','Volume']]
    df.index = df.index.tz_localize(None)
    df = ozellikler_ekle_ml(df)
    df['Gelecek'] = df['Close'].shift(-3) / df['Close'] - 1
    df['Hedef']   = df['Gelecek'].apply(
        lambda g: 2 if g >= 0.015 else (0 if g <= -0.015 else 1))
    df = df.dropna()

    X      = df[ML_OZELLIKLER].values
    y      = df['Hedef'].values
    bolme  = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_e    = scaler.fit_transform(X[:bolme])

    model = VotingClassifier(estimators=[
        ('rf',   RandomForestClassifier(n_estimators=200, max_depth=6,
                  min_samples_leaf=5, random_state=42, class_weight='balanced')),
        ('xgb',  XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                  random_state=42, eval_metric='mlogloss', verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                  random_state=42, class_weight='balanced', verbose=-1))
    ], voting='soft')
    model.fit(X_e, y[:bolme])
    return model, scaler, df

# ══════════════════════════════════════════════════════
# MODÜL 4 — LSTM OYLAMA (Yardımcı Katman)
# ══════════════════════════════════════════════════════
class LSTMv2(nn.Module):
    def __init__(self, giris, gizli=64, cikis=3):
        super().__init__()
        self.lstm       = nn.LSTM(giris, gizli, 1, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(gizli)
        self.dropout1   = nn.Dropout(0.4)
        self.fc1        = nn.Linear(gizli, 32)
        self.dropout2   = nn.Dropout(0.3)
        self.fc2        = nn.Linear(32, cikis)
        self.relu       = nn.ReLU()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.batch_norm(hn[-1])
        out = self.dropout1(out)
        out = self.relu(self.fc1(out))
        out = self.dropout2(out)
        return self.fc2(out)

LSTM_OZELLIKLER = [
    'RSI_Norm', 'RSI_f_Norm', 'MACD_Norm', 'MACD_hist',
    'BB_Konum', 'MA5_Fark', 'MA20_Fark', 'MA50_Fark', 'MA200_Fark', 'Trend_Guc',
    'Getiri_1g', 'Getiri_3g', 'Getiri_5g', 'Getiri_10g',
    'Hacim_Oran', 'ATR', 'Volatilite', 'Kanat', 'Yon', '52H_Yuzde'
]
PENCERE = 20

def lstm_sinyal_yukle(sembol):
    """Kaydedilmiş LSTM modelini yükle"""
    try:
        dosya = f"{sembol.replace('.IS','')}_lstm_v2.pt"
        kayit = torch.load(dosya, map_location='cpu', weights_only=False)
        model = LSTMv2(giris=len(LSTM_OZELLIKLER))
        model.load_state_dict(kayit['model_state'])
        model.eval()
        return model, kayit['scaler']
    except:
        return None, None

def lstm_tahmin_yap(model, scaler, df):
    """LSTM ile son günün tahmini"""
    try:
        df_ozl = ozellikler_ekle_ml(df)
        if len(df_ozl) < PENCERE:
            return None, 0

        pencere = df_ozl[LSTM_OZELLIKLER].values[-PENCERE:]
        scaled  = scaler.transform(pencere)
        X       = torch.FloatTensor(scaled).unsqueeze(0)

        with torch.no_grad():
            cikis   = model(X)
            olas    = torch.softmax(cikis, dim=1).numpy()[0]
            tahmin  = int(np.argmax(olas))

        karar = {2: "AL", 0: "SAT", 1: "BEKLE"}[tahmin]
        guven = float(olas[tahmin])
        return karar, guven
    except:
        return None, 0

# ══════════════════════════════════════════════════════
# MODÜL 5 — SİNYAL ÜRET & OYLAMA
# ══════════════════════════════════════════════════════
def sinyal_uret_tam(sembol, carpani):
    """
    RF/XGBoost + LSTM oylama sistemi.
    İki model aynı yönde → güven %20 artar.
    """
    df_ham = yf.Ticker(sembol).history(period=PERIYOT, interval="1d")
    df_ham = df_ham[['Open','High','Low','Close','Volume']]
    df_ham.index = df_ham.index.tz_localize(None)

    # ML modeli
    ml_model, ml_scaler, df_ml = ml_model_egit(sembol)
    son_X        = ml_scaler.transform([df_ml[ML_OZELLIKLER].iloc[-1].values])
    ml_tahmin    = ml_model.predict(son_X)[0]
    ml_olas      = ml_model.predict_proba(son_X)[0]
    ml_guven     = float(max(ml_olas))
    ml_karar     = {2: "AL", 0: "SAT", 1: "BEKLE"}[ml_tahmin]

    guven_esigi  = 0.38 / carpani
    if ml_guven < guven_esigi:
        ml_karar = "BEKLE"

    # LSTM modeli (varsa)
    lstm_model, lstm_scaler = lstm_sinyal_yukle(sembol)
    lstm_karar = None
    if lstm_model and lstm_scaler:
        lstm_karar, lstm_guven = lstm_tahmin_yap(lstm_model, lstm_scaler, df_ham)

    # Oylama — iki model aynı yöndeyse güveni artır
    final_karar = ml_karar
    final_guven = ml_guven
    lstm_onay   = False

    if lstm_karar and lstm_karar == ml_karar and ml_karar != "BEKLE":
        final_guven = min(ml_guven * 1.2, 0.99)
        lstm_onay   = True

    # Son fiyat ve teknik göstergeler
    ticker    = yf.Ticker(sembol)
    son_fiyat = ticker.fast_info.last_price
    onceki    = ticker.fast_info.regular_market_previous_close
    degisim   = ((son_fiyat - onceki) / onceki * 100) if onceki else 0
    rsi       = float(df_ml['RSI'].iloc[-1])
    atr       = float(df_ml['ATR'].iloc[-1])

    return {
        'sembol'     : sembol,
        'karar'      : final_karar,
        'guven'      : round(final_guven, 3),
        'ml_karar'   : ml_karar,
        'lstm_karar' : lstm_karar or "—",
        'lstm_onay'  : lstm_onay,
        'fiyat'      : round(float(son_fiyat), 2),
        'degisim'    : round(float(degisim), 2),
        'rsi'        : round(rsi, 1),
        'hedef'      : round(son_fiyat + atr * 2.5, 2) if final_karar == "AL" else None,
        'stop'       : round(son_fiyat - atr * 1.5, 2) if final_karar == "AL" else None,
        'zaman'      : datetime.now().strftime("%d.%m.%Y %H:%M"),
    }

# ══════════════════════════════════════════════════════
# MODÜL 6 — TRACK RECORD
# ══════════════════════════════════════════════════════
def track_record_kaydet(sinyaller):
    """Her sinyali kaydet — performans takibi için"""
    kayitlar = []
    for s in sinyaller:
        kayitlar.append({
            'zaman'      : s['zaman'],
            'sembol'     : s['sembol'],
            'karar'      : s['karar'],
            'guven'      : s['guven'],
            'fiyat_giris': s['fiyat'],
            'rsi'        : s['rsi'],
            'hedef'      : s.get('hedef', ''),
            'stop'       : s.get('stop', ''),
            'ml_karar'   : s['ml_karar'],
            'lstm_karar' : s['lstm_karar'],
            'lstm_onay'  : s['lstm_onay'],
            'fiyat_cikis': '',    # Sonradan doldurulacak
            'kar_zarar'  : '',    # Sonradan doldurulacak
            'sonuc'      : '',    # KAZANDI / KAYBETTI
        })

    df_yeni = pd.DataFrame(kayitlar)
    try:
        df_mevcut = pd.read_csv("track_record.csv", encoding='utf-8-sig')
        df_toplam = pd.concat([df_mevcut, df_yeni], ignore_index=True)
    except:
        df_toplam = df_yeni

    df_toplam.to_csv("track_record.csv", index=False, encoding='utf-8-sig')
    print(f"    {len(kayitlar)} sinyal track_record.csv'ye kaydedildi.")

def track_record_guncelle():
    """
    Kaydedilen sinyallerin sonuçlarını güncelle.
    3 gün önceki sinyaller için güncel fiyatı çek,
    kar/zarar hesapla.
    """
    try:
        df = pd.read_csv("track_record.csv", encoding='utf-8-sig')
        guncellendi = 0

        for idx, row in df.iterrows():
            if row['sonuc'] != '' or row['karar'] == 'BEKLE':
                continue

            try:
                zaman_giris = datetime.strptime(
                    str(row['zaman']), "%d.%m.%Y %H:%M")
                gun_fark    = (datetime.now() - zaman_giris).days

                if gun_fark < 3:
                    continue

                ticker      = yf.Ticker(row['sembol'])
                guncel      = ticker.fast_info.last_price
                fiyat_giris = float(row['fiyat_giris'])
                getiri      = (guncel - fiyat_giris) / fiyat_giris * 100

                if row['karar'] == 'AL':
                    kar_zarar = getiri
                    sonuc     = "KAZANDI" if getiri > 0 else "KAYBETTI"
                else:
                    kar_zarar = -getiri
                    sonuc     = "KAZANDI" if getiri < 0 else "KAYBETTI"

                df.at[idx, 'fiyat_cikis'] = round(guncel, 2)
                df.at[idx, 'kar_zarar']   = round(kar_zarar, 2)
                df.at[idx, 'sonuc']       = sonuc
                guncellendi += 1

            except:
                continue

        df.to_csv("track_record.csv", index=False, encoding='utf-8-sig')
        if guncellendi > 0:
            print(f"    {guncellendi} sinyal sonucu güncellendi.")

    except:
        pass

def track_record_ozet():
    """Track record istatistiklerini hesapla"""
    try:
        df = pd.read_csv("track_record.csv", encoding='utf-8-sig')
        tamamlanan = df[df['sonuc'].isin(['KAZANDI', 'KAYBETTI'])]

        if len(tamamlanan) == 0:
            return None

        kazanan   = len(tamamlanan[tamamlanan['sonuc'] == 'KAZANDI'])
        toplam    = len(tamamlanan)
        basari    = kazanan / toplam * 100
        ort_kar   = tamamlanan[tamamlanan['kar_zarar'] != '']['kar_zarar'].astype(float).mean()

        return {
            'toplam_sinyal' : len(df),
            'tamamlanan'    : toplam,
            'kazanan'       : kazanan,
            'basari_orani'  : round(basari, 1),
            'ort_kar_zarar' : round(float(ort_kar), 2),
        }
    except:
        return None

# ══════════════════════════════════════════════════════
# MODÜL 7 — E-POSTA
# ══════════════════════════════════════════════════════
def email_gonder(sinyaller, rejim, track_ozet=None):
    if not sinyaller:
        print("\n[5/5] Sinyal yok — e-posta atlanıyor.")
        return

    zaman = datetime.now().strftime("%d.%m.%Y %H:%M")
    konu  = (f"BIST Sinyal — {len(sinyaller)} sinyal | "
             f"{rejim['emoji']} {rejim['rejim']}")

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:680px;margin:auto">
    <h2 style="color:#1e40af;border-bottom:2px solid #1e40af;padding-bottom:8px">
        BIST Sinyal Raporu — {zaman}
    </h2>
    <div style="background:#f1f5f9;border-left:4px solid #64748b;
                padding:12px;margin-bottom:20px;border-radius:4px;font-size:14px">
        <b>Piyasa:</b> {rejim['emoji']} {rejim['rejim']} |
        <b>BIST100:</b> {rejim['bist_son']:,.0f} |
        <b>RSI:</b> {rejim['bist_rsi']:.1f} |
        <b>USD/TRY:</b> {rejim['usdtry']:.2f} |
        <b>1 Aylık:</b> %{rejim['getiri_1ay']:.1f}
    </div>
    """

    if track_ozet:
        html += f"""
    <div style="background:#ecfdf5;border-left:4px solid #22c55e;
                padding:12px;margin-bottom:20px;border-radius:4px;font-size:14px">
        <b>Track Record:</b>
        {track_ozet['tamamlanan']} tamamlanan sinyal |
        Başarı: %{track_ozet['basari_orani']} |
        Ort. Kar/Zarar: %{track_ozet['ort_kar_zarar']:+.2f}
    </div>
        """

    for s in sinyaller:
        renk = "#16a34a" if s['karar'] == "AL" else "#dc2626"
        bg   = "#f0fdf4" if s['karar'] == "AL" else "#fef2f2"
        ok   = "▲" if s['karar'] == "AL" else "▼"
        lstm_txt = f"✓ LSTM onaylı" if s['lstm_onay'] else ""

        html += f"""
    <div style="background:{bg};border-left:4px solid {renk};
                padding:16px;margin:12px 0;border-radius:6px">
        <h3 style="margin:0 0 10px;color:{renk}">
            {ok} {s['sembol'].replace('.IS','')} — {s['karar']}
            <span style="font-size:12px;color:#666;margin-left:8px">{lstm_txt}</span>
        </h3>
        <table style="width:100%;font-size:14px">
            <tr>
                <td><b>Fiyat</b></td><td>{s['fiyat']:.2f} TL</td>
                <td><b>Değişim</b></td>
                <td style="color:{'#16a34a' if s['degisim']>=0 else '#dc2626'}">
                    %{s['degisim']:+.2f}</td>
            </tr>
            <tr>
                <td><b>RSI</b></td><td>{s['rsi']:.1f}</td>
                <td><b>Güven</b></td><td>%{s['guven']*100:.0f}</td>
            </tr>
            <tr>
                <td><b>ML Karar</b></td><td>{s['ml_karar']}</td>
                <td><b>LSTM Karar</b></td><td>{s['lstm_karar']}</td>
            </tr>
            <tr>
                <td><b>Hedef</b></td>
                <td style="color:#16a34a">
                    {f"{s['hedef']:.2f} TL" if s['hedef'] else "—"}</td>
                <td><b>Stop</b></td>
                <td style="color:#dc2626">
                    {f"{s['stop']:.2f} TL" if s['stop'] else "—"}</td>
            </tr>
        </table>
    </div>
        """

    html += """
    <p style="color:#94a3b8;font-size:12px;margin-top:24px">
        Otomatik oluşturulmuştur. Yatırım tavsiyesi değildir.
    </p></body></html>
    """

    try:
        msg            = MIMEMultipart('alternative')
        msg['Subject'] = konu
        msg['From']    = GMAIL_ADRES
        msg['To']      = ALICI_ADRES
        msg.attach(MIMEText(html, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as srv:
            srv.login(GMAIL_ADRES, GMAIL_SIFRE)
            srv.sendmail(GMAIL_ADRES, ALICI_ADRES, msg.as_string())
        print(f"\n[5/5] E-posta gönderildi → {ALICI_ADRES}")
    except Exception as e:
        print(f"\n[5/5] E-posta hatası: {e}")

# ══════════════════════════════════════════════════════
# ANA AKIŞ
# ══════════════════════════════════════════════════════
def main():
    zaman = datetime.now().strftime("%d.%m.%Y %H:%M")
    print("\n" + "="*65)
    print(f"  BIST ANA SİSTEM v2.0 — {zaman}")
    print("="*65)

    # 1. Piyasa rejimi
    rejim = piyasa_rejimi_tespit()

    if rejim['rejim'] == "GUCLU_AYI":
        print("\n  ❌ Güçlü ayı piyasası — sinyal üretimi durduruldu.")
        return

    # 2. Temel analiz filtresi
    secilen = temel_filtre(HISSELER)

    # 3. Track record güncelle
    print("\n[3/5] Track record güncelleniyor...")
    track_record_guncelle()
    track_ozet = track_record_ozet()
    if track_ozet:
        print(f"    Başarı oranı: %{track_ozet['basari_orani']} | "
              f"Tamamlanan: {track_ozet['tamamlanan']} sinyal")

    # 4. ML + LSTM sinyalleri
    print(f"\n[4/5] Sinyal üretiliyor ({len(secilen)} hisse)...")
    sinyaller = []

    for s in secilen:
        print(f"  {s} analiz ediliyor...")
        try:
            sonuc = sinyal_uret_tam(s, rejim['carpani'])
            emoji = ("🟢" if sonuc['karar'] == "AL" else
                     "🔴" if sonuc['karar'] == "SAT" else "⚪")
            lstm  = "✓LSTM" if sonuc['lstm_onay'] else ""
            print(f"    {emoji} {sonuc['karar']:<6} "
                  f"Güven:%{sonuc['guven']*100:.0f} "
                  f"Fiyat:{sonuc['fiyat']:.2f} "
                  f"RSI:{sonuc['rsi']:.1f} {lstm}")

            if sonuc['karar'] != "BEKLE":
                sinyaller.append(sonuc)
        except Exception as e:
            print(f"    HATA: {e}")

    # 5. Track record kaydet
    track_record_kaydet(sinyaller + [
        s for s in [sinyal_uret_tam(h, rejim['carpani'])
                    for h in secilen
                    if h not in [x['sembol'] for x in sinyaller]]
        if s['karar'] == 'BEKLE'
    ])

    # 6. Özet rapor
    print("\n" + "="*65)
    print("  SINYAL ÖZETİ")
    print("="*65)
    al_lst  = [s for s in sinyaller if s['karar'] == "AL"]
    sat_lst = [s for s in sinyaller if s['karar'] == "SAT"]

    print(f"\n  🟢 AL Sinyalleri ({len(al_lst)}):")
    for s in al_lst:
        lstm = "✓LSTM" if s['lstm_onay'] else ""
        print(f"    {s['sembol']:<12} {s['fiyat']:.2f} TL | "
              f"Hedef: {s['hedef']:.2f} | "
              f"Stop: {s['stop']:.2f} | "
              f"Güven: %{s['guven']*100:.0f} {lstm}")

    print(f"\n  🔴 SAT Sinyalleri ({len(sat_lst)}):")
    for s in sat_lst:
        print(f"    {s['sembol']:<12} {s['fiyat']:.2f} TL | "
              f"Güven: %{s['guven']*100:.0f}")

    if track_ozet:
        print(f"\n  📊 Track Record: "
              f"%{track_ozet['basari_orani']} başarı | "
              f"{track_ozet['tamamlanan']} sinyal | "
              f"Ort: %{track_ozet['ort_kar_zarar']:+.2f}")

    # 7. E-posta
    email_gonder(sinyaller, rejim, track_ozet)
    print("="*65)

if __name__ == "__main__":
    main()