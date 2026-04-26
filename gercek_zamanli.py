import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import time
import os
from datetime import datetime, time as dtime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── Ayarlar ───────────────────────────────────────────
HISSELER = [
    "AKBNK.IS", "GARAN.IS", "YKBNK.IS", "EKGYO.IS",
    "PGSUS.IS", "TCELL.IS", "SISE.IS",  "FROTO.IS"
]
GUNCELLEME_SURESI = 300        # 5 dakika = 300 saniye
BORSA_ACILIS      = dtime(10, 0)
BORSA_KAPANIS     = dtime(18, 10)
LOG_DOSYASI       = "canli_sinyaller.csv"
# ───────────────────────────────────────────────────────

def borsa_acik_mi():
    """Borsa İstanbul saatleri: 10:00 - 18:10"""
    simdi = datetime.now().time()
    return BORSA_ACILIS <= simdi <= BORSA_KAPANIS

def veri_cek_canli(sembol):
    """
    Yahoo Finance 15 dakika gecikmeli veri çeker.
    1 dakikalık veri — son 5 günden.
    """
    ticker = yf.Ticker(sembol)

    # Günlük veri — model eğitimi için (5 yıl)
    df_gunluk = ticker.history(period="5y", interval="1d")
    df_gunluk.index = df_gunluk.index.tz_localize(None)

    # Saatlik veri — gün içi trend için (60 gün)
    df_saatlik = ticker.history(period="60d", interval="1h")
    df_saatlik.index = df_saatlik.index.tz_localize(None)

    # Son fiyat (en güncel)
    son_fiyat  = ticker.fast_info.last_price
    son_hacim  = ticker.fast_info.three_month_average_volume

    return df_gunluk, df_saatlik, son_fiyat, son_hacim

def ozellikler_ekle(df):
    df = df.copy()
    df['RSI']           = ta.rsi(df['Close'], length=14)
    df['RSI_fast']      = ta.rsi(df['Close'], length=7)
    macd                = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']          = macd['MACD_12_26_9']
    df['MACD_sinyal']   = macd['MACDs_12_26_9']
    df['MACD_hist']     = macd['MACDh_12_26_9']
    df['MA5']           = ta.sma(df['Close'], length=5)
    df['MA20']          = ta.sma(df['Close'], length=20)
    df['MA50']          = ta.sma(df['Close'], length=50)
    df['MA200']         = ta.sma(df['Close'], length=200)
    df['EMA20']         = ta.ema(df['Close'], length=20)
    bb                  = ta.bbands(df['Close'], length=20, std=2)
    bb_s                = bb.columns.tolist()
    df['BB_ust']        = bb[bb_s[2]]
    df['BB_alt']        = bb[bb_s[0]]
    df['BB_genislik']   = (df['BB_ust'] - bb[bb_s[1]]) / bb[bb_s[1]]
    df['ATR']           = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Volatilite']    = df['Close'].pct_change().rolling(20).std()
    df['Hacim_MA20']    = ta.sma(df['Volume'], length=20)
    df['Hacim_Oran']    = df['Volume'] / df['Hacim_MA20']
    df['Hacim_Trend']   = df['Hacim_Oran'].rolling(5).mean()
    df['RSI_Norm']      = (df['RSI'] - 50) / 50
    df['RSI_fast_Norm'] = (df['RSI_fast'] - 50) / 50
    df['BB_Konum']      = (df['Close'] - df['BB_alt']) / (df['BB_ust'] - df['BB_alt'])
    df['MA5_Fark']      = (df['Close'] - df['MA5'])   / df['MA5']
    df['MA20_Fark']     = (df['Close'] - df['MA20'])  / df['MA20']
    df['MA50_Fark']     = (df['Close'] - df['MA50'])  / df['MA50']
    df['MA200_Fark']    = (df['Close'] - df['MA200']) / df['MA200']
    df['MACD_Norm']     = df['MACD'] - df['MACD_sinyal']
    df['Trend_Guc']     = (df['MA5'] - df['MA50'])   / df['MA50']
    df['EMA_MA_Fark']   = (df['EMA20'] - df['MA20']) / df['MA20']
    for g in [1, 3, 5, 10, 20]:
        df[f'Getiri_{g}g'] = df['Close'].pct_change(g)
    df['Kanat']         = (df['High'] - df['Low']) / df['Close']
    df['Govde']         = abs(df['Close'] - df['Open']) / df['Close']
    df['Yon']           = np.where(df['Close'] > df['Open'], 1, -1)
    df['52H_Yuzde']     = df['Close'] / df['Close'].rolling(252).max()
    df['52L_Yuzde']     = df['Close'] / df['Close'].rolling(252).min()
    df['RSI_Trend']     = df['RSI'] - df['RSI'].shift(5)
    df['Hacim_Fiyat']   = df['Getiri_1g'] * df['Hacim_Oran']
    return df.dropna()

OZELLIKLER = [
    'RSI_Norm', 'RSI_fast_Norm', 'MACD_Norm', 'MACD_hist',
    'BB_Konum', 'BB_genislik', 'MA5_Fark', 'MA20_Fark',
    'MA50_Fark', 'MA200_Fark', 'Trend_Guc', 'EMA_MA_Fark',
    'Getiri_1g', 'Getiri_3g', 'Getiri_5g', 'Getiri_10g', 'Getiri_20g',
    'Hacim_Oran', 'Hacim_Trend', 'ATR', 'Volatilite',
    'Kanat', 'Govde', 'Yon', '52H_Yuzde', '52L_Yuzde',
    'RSI_Trend', 'Hacim_Fiyat'
]

def model_egit(df_gunluk):
    """Günlük veriyle model eğit — her taramada bir kez yapılır"""
    df = ozellikler_ekle(df_gunluk)
    df['Gelecek'] = df['Close'].shift(-3) / df['Close'] - 1
    df['Hedef']   = df['Gelecek'].apply(
        lambda g: 2 if g >= 0.015 else (0 if g <= -0.015 else 1))
    df = df.dropna()

    X      = df[OZELLIKLER].values
    y      = df['Hedef'].values
    bolme  = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_e    = scaler.fit_transform(X[:bolme])
    y_e    = y[:bolme]

    model = VotingClassifier(estimators=[
        ('rf',   RandomForestClassifier(
                    n_estimators=200, max_depth=6,
                    min_samples_leaf=5, random_state=42,
                    class_weight='balanced')),
        ('xgb',  XGBClassifier(
                    n_estimators=150, max_depth=5,
                    learning_rate=0.05, random_state=42,
                    eval_metric='mlogloss', verbosity=0)),
        ('lgbm', LGBMClassifier(
                    n_estimators=150, max_depth=5,
                    learning_rate=0.05, random_state=42,
                    class_weight='balanced', verbose=-1))
    ], voting='soft')
    model.fit(X_e, y_e)
    return model, scaler, df

def sinyal_uret(model, scaler, df, son_fiyat):
    """Son veriyle gerçek zamanlı sinyal üret"""
    son_satir   = df[OZELLIKLER].iloc[-1].values
    X           = scaler.transform([son_satir])
    tahmin      = model.predict(X)[0]
    olasiliklar = model.predict_proba(X)[0]
    guven       = max(olasiliklar)

    karar = {2: "AL", 0: "SAT", 1: "BEKLE"}[tahmin]
    if guven < 0.38:
        karar = "BEKLE"

    rsi  = df['RSI'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    atr  = df['ATR'].iloc[-1]

    # Dinamik hedef ve stop
    if karar == "AL":
        hedef = son_fiyat + (atr * 2.5)
        stop  = son_fiyat - (atr * 1.5)
    elif karar == "SAT":
        hedef = son_fiyat - (atr * 2.5)
        stop  = son_fiyat + (atr * 1.5)
    else:
        hedef = stop = None

    return {
        'karar'     : karar,
        'guven'     : guven,
        'al_olas'   : olasiliklar[2],
        'sat_olas'  : olasiliklar[0],
        'fiyat'     : son_fiyat,
        'rsi'       : rsi,
        'ma20'      : ma20,
        'ma50'      : ma50,
        'hedef'     : hedef,
        'stop'      : stop,
        'atr'       : atr,
    }

def guncelleme_yap(modeller):
    """Her 5 dakikada bir çalışır — anlık fiyat güncelle"""
    zaman = datetime.now().strftime("%H:%M:%S")

    # Ekranı temizle
    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 75)
    print(f"  BIST CANLI TAKİP SİSTEMİ  |  {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"  Sonraki güncelleme: {GUNCELLEME_SURESI//60} dakika sonra  |  "
          f"Borsa: {'🟢 AÇIK' if borsa_acik_mi() else '🔴 KAPALI'}")
    print("=" * 75)
    print(f"  {'Hisse':<12} {'Fiyat':>8} {'Değ%':>7} {'RSI':>6} "
          f"{'Karar':<8} {'Güven':>6} {'Hedef':>9} {'Stop':>9}")
    print(f"  {'─'*69}")

    kayitlar = []

    for sembol, (model, scaler, df_egitim) in modeller.items():
        try:
            ticker    = yf.Ticker(sembol)
            son_fiyat = ticker.fast_info.last_price
            onceki    = ticker.fast_info.regular_market_previous_close
            degisim   = ((son_fiyat - onceki) / onceki) * 100 if onceki else 0

            sonuc = sinyal_uret(model, scaler, df_egitim, son_fiyat)

            karar  = sonuc['karar']
            emoji  = "🟢" if karar == "AL" else ("🔴" if karar == "SAT" else "⚪")
            degisim_renk = "+" if degisim >= 0 else ""

            hedef_str = f"{sonuc['hedef']:.2f}" if sonuc['hedef'] else "  —  "
            stop_str  = f"{sonuc['stop']:.2f}"  if sonuc['stop']  else "  —  "

            print(f"  {sembol:<12} {son_fiyat:>8.2f} "
                  f"{degisim_renk}{degisim:>6.2f}% "
                  f"{sonuc['rsi']:>6.1f} "
                  f"{emoji} {karar:<6} "
                  f"%{sonuc['guven']*100:>5.0f} "
                  f"{hedef_str:>9} "
                  f"{stop_str:>9}")

            # Kayıt
            kayitlar.append({
                'zaman'  : datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                'sembol' : sembol,
                'fiyat'  : son_fiyat,
                'degisim': degisim,
                'karar'  : karar,
                'guven'  : sonuc['guven'],
                'rsi'    : sonuc['rsi'],
                'hedef'  : sonuc['hedef'],
                'stop'   : sonuc['stop'],
            })

        except Exception as e:
            print(f"  {sembol:<12} HATA: {e}")

    print(f"  {'─'*69}")

    # Özet satırı
    al_sayisi  = sum(1 for k in kayitlar if k['karar'] == 'AL')
    sat_sayisi = sum(1 for k in kayitlar if k['karar'] == 'SAT')
    print(f"\n  Özet: 🟢 {al_sayisi} AL  |  🔴 {sat_sayisi} SAT  |  "
          f"⚪ {len(kayitlar)-al_sayisi-sat_sayisi} BEKLE")
    print(f"  Son güncelleme: {zaman}")

    # CSV'ye kaydet
    if kayitlar:
        yeni_df = pd.DataFrame(kayitlar)
        if os.path.exists(LOG_DOSYASI):
            mevcut = pd.read_csv(LOG_DOSYASI)
            yeni_df = pd.concat([mevcut, yeni_df], ignore_index=True)
        yeni_df.to_csv(LOG_DOSYASI, index=False, encoding='utf-8-sig')

    return kayitlar

def modelleri_egit():
    """Başlangıçta tüm modelleri eğit — bu biraz uzun sürer"""
    print("Modeller eğitiliyor, lütfen bekle...")
    print("(Bu işlem yaklaşık 5-10 dakika sürer, sonra otomatik çalışır)\n")

    modeller = {}
    for i, sembol in enumerate(HISSELER, 1):
        print(f"  [{i:02d}/{len(HISSELER)}] {sembol} modeli eğitiliyor...")
        try:
            df_gunluk, df_saatlik, son_fiyat, _ = veri_cek_canli(sembol)
            df_gunluk = df_gunluk[['Open','High','Low','Close','Volume']]
            model, scaler, df_egitim = model_egit(df_gunluk)
            modeller[sembol] = (model, scaler, df_egitim)
            print(f"        ✅ Hazır  |  Son fiyat: {son_fiyat:.2f} TL")
        except Exception as e:
            print(f"        ❌ Hata: {e}")

    print(f"\n✅ {len(modeller)} model hazır. Canlı takip başlıyor...\n")
    return modeller

def main():
    print("\n" + "="*75)
    print("  BIST CANLI TAKİP SİSTEMİ BAŞLATILIYOR")
    print("  Durdurmak için CTRL+C'ye bas")
    print("="*75 + "\n")

    # Modelleri bir kez eğit
    modeller = modelleri_egit()

    guncelleme_sayisi = 0

    while True:
        try:
            guncelleme_sayisi += 1
            kayitlar = guncelleme_yap(modeller)

            # Her 50 güncellemede bir modeli yeniden eğit
            # (yaklaşık her 4 saatte bir)
            if guncelleme_sayisi % 50 == 0:
                print("\n  Model yenileniyor...")
                modeller = modelleri_egit()

            print(f"\n  Bekleniyor ({GUNCELLEME_SURESI//60} dakika)... "
                  f"CTRL+C ile durdurabilirsin.")
            time.sleep(GUNCELLEME_SURESI)

        except KeyboardInterrupt:
            print("\n\n  Sistem durduruldu.")
            print(f"  Tüm sinyaller '{LOG_DOSYASI}' dosyasına kaydedildi.")
            break
        except Exception as e:
            print(f"\n  HATA: {e} — 60 saniye sonra tekrar denenecek.")
            time.sleep(60)

if __name__ == "__main__":
    main()