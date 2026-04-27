import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── Ayarlar ───────────────────────────────────────────
HISSELER = [
    "THYAO.IS", "KCHOL.IS", "EREGL.IS", "ASELS.IS",
    "BIMAS.IS", "SAHOL.IS", "SISE.IS",  "AKBNK.IS",
    "GARAN.IS", "YKBNK.IS", "TUPRS.IS", "ARCLK.IS",
    "TOASO.IS", "FROTO.IS", "PGSUS.IS", "TCELL.IS",
    "EKGYO.IS", "KOZAL.IS", "PETKM.IS", "VESTL.IS"
]
PERIYOT           = "5y"
BASLANGIC_SERMAYE = 100000
KOMISYON          = 0.001
STOP_LOSS         = 0.07
GUVEN_ESIGI       = 0.38
# ───────────────────────────────────────────────────────

def veri_cek(sembol, periyot):
    df = yf.Ticker(sembol).history(period=periyot, interval="1d")
    if df.empty:
        raise ValueError(f"{sembol} verisi boş geldi")
    df = df[['Open','High','Low','Close','Volume']]
    df.index = df.index.tz_localize(None)
    return df

def ozellikler_ekle(df):
    df['RSI']           = ta.rsi(df['Close'], length=14)
    df['RSI_fast']      = ta.rsi(df['Close'], length=7)
    macd                = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']          = macd['MACD_12_26_9']
    df['MACD_sinyal']   = macd['MACDs_12_26_9']
    df['MACD_hist']     = macd['MACDh_12_26_9']
    df['MA5']           = ta.sma(df['Close'], length=5)
    df['MA10']          = ta.sma(df['Close'], length=10)
    df['MA20']          = ta.sma(df['Close'], length=20)
    df['MA50']          = ta.sma(df['Close'], length=50)
    df['MA200']         = ta.sma(df['Close'], length=200)
    df['EMA20']         = ta.ema(df['Close'], length=20)
    bb                  = ta.bbands(df['Close'], length=20, std=2)
    bb_s                = bb.columns.tolist()
    df['BB_ust']        = bb[bb_s[2]]
    df['BB_alt']        = bb[bb_s[0]]
    df['BB_orta']       = bb[bb_s[1]]
    df['BB_genislik']   = (df['BB_ust'] - df['BB_alt']) / df['BB_orta']
    df['Momentum_5']    = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10']   = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_20']   = df['Close'] / df['Close'].shift(20) - 1
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
    df['Trend_Guc']     = (df['MA5'] - df['MA50'])  / df['MA50']
    df['EMA_MA_Fark']   = (df['EMA20'] - df['MA20']) / df['MA20']
    df['Getiri_1g']     = df['Close'].pct_change(1)
    df['Getiri_3g']     = df['Close'].pct_change(3)
    df['Getiri_5g']     = df['Close'].pct_change(5)
    df['Getiri_10g']    = df['Close'].pct_change(10)
    df['Getiri_20g']    = df['Close'].pct_change(20)
    df['Kanat']         = (df['High'] - df['Low']) / df['Close']
    df['Govde']         = abs(df['Close'] - df['Open']) / df['Close']
    df['Yon']           = np.where(df['Close'] > df['Open'], 1, -1)

    # Yeni özellikler — piyasa rejimi
    df['52H_Yuzde']     = df['Close'] / df['Close'].rolling(252).max()
    df['52L_Yuzde']     = df['Close'] / df['Close'].rolling(252).min()
    df['RSI_Trend']     = df['RSI'] - df['RSI'].shift(5)
    df['Hacim_Fiyat']   = df['Getiri_1g'] * df['Hacim_Oran']

    return df.dropna()

def hedef_olustur(df, ileriye_gun=3, hedef_getiri=0.015):
    df['Gelecek_Getiri'] = df['Close'].shift(-ileriye_gun) / df['Close'] - 1
    def etiketle(g):
        if g >= hedef_getiri:  return 2
        elif g <= -hedef_getiri: return 0
        else: return 1
    df['Hedef'] = df['Gelecek_Getiri'].apply(etiketle)
    return df.dropna()

def model_egit(df, ozellikler):
    X = df[ozellikler].values
    y = df['Hedef'].values

    bolme    = int(len(X) * 0.8)
    X_e, X_t = X[:bolme], X[bolme:]
    y_e, y_t = y[:bolme], y[bolme:]

    scaler = StandardScaler()
    X_e    = scaler.fit_transform(X_e)
    X_t    = scaler.transform(X_t)

    # 4 farklı model — ensemble ile birleştir
    rf  = RandomForestClassifier(
        n_estimators=300, max_depth=6,
        min_samples_split=15, min_samples_leaf=7,
        random_state=42, class_weight='balanced'
    )
    gb  = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=42
    )
    xgb = XGBClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        eval_metric='mlogloss', verbosity=0
    )
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.05, random_state=42,
        class_weight='balanced', verbose=-1
    )

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('xgb', xgb), ('lgbm', lgbm)],
        voting='soft'
    )
    ensemble.fit(X_e, y_e)

    y_tahmin = ensemble.predict(X_t)
    rapor    = classification_report(y_t, y_tahmin,
               target_names=['SAT','BEKLE','AL'],
               zero_division=0, output_dict=True)

    al_f1  = rapor['AL']['f1-score']
    sat_f1 = rapor['SAT']['f1-score']
    acc    = rapor['accuracy']

    print(f"  Doğruluk: %{acc*100:.1f} | AL f1: {al_f1:.2f} | SAT f1: {sat_f1:.2f}")

    return ensemble, scaler, acc

def ml_backtest(df, model, scaler, ozellikler):
    sermaye     = BASLANGIC_SERMAYE
    hisse_adedi = 0
    pozisyon    = False
    alis_fiyati = 0
    islemler    = []
    portfoy     = []

    bolme   = int(len(df) * 0.8)
    test_df = df.iloc[bolme:].copy()

    for i in range(1, len(test_df)):
        satir = test_df.iloc[i]
        fiyat = satir['Close']
        tarih = test_df.index[i]

        X           = scaler.transform([satir[ozellikler].values])
        tahmin      = model.predict(X)[0]
        olasiliklar = model.predict_proba(X)[0]
        guven       = max(olasiliklar)

        if guven < GUVEN_ESIGI:
            tahmin = 1

        karar = {2: "AL", 0: "SAT", 1: "BEKLE"}[tahmin]

        if pozisyon and alis_fiyati > 0:
            if (fiyat - alis_fiyati) / alis_fiyati <= -STOP_LOSS:
                karar = "SAT"

        if karar == "AL" and not pozisyon:
            hisse_adedi = int(sermaye / fiyat)
            maliyet     = hisse_adedi * fiyat * (1 + KOMISYON)
            if hisse_adedi > 0:
                sermaye    -= maliyet
                pozisyon    = True
                alis_fiyati = fiyat
                islemler.append({'tarih': tarih, 'islem': 'AL',
                                 'fiyat': fiyat, 'adet': hisse_adedi,
                                 'tutar': maliyet, 'guven': guven})

        elif karar == "SAT" and pozisyon:
            gelir     = hisse_adedi * fiyat * (1 - KOMISYON)
            sermaye  += gelir
            kar_zarar = gelir - islemler[-1]['tutar']
            islemler.append({'tarih': tarih, 'islem': 'SAT',
                             'fiyat': fiyat, 'adet': hisse_adedi,
                             'tutar': gelir, 'kar_zarar': kar_zarar,
                             'guven': guven})
            hisse_adedi = 0
            pozisyon    = False
            alis_fiyati = 0

        portfoy.append({'tarih': tarih,
                        'deger': sermaye + hisse_adedi * fiyat,
                        'fiyat': fiyat})

    if pozisyon:
        sermaye += hisse_adedi * test_df['Close'].iloc[-1] * (1 - KOMISYON)

    return pd.DataFrame(islemler), pd.DataFrame(portfoy), sermaye

def sonuc_hesapla(sembol, islemler_df, portfoy_df, son_sermaye, acc):
    toplam_kar = son_sermaye - BASLANGIC_SERMAYE
    getiri     = (toplam_kar / BASLANGIC_SERMAYE) * 100
    ilk_fiyat  = portfoy_df['fiyat'].iloc[0]
    son_fiyat  = portfoy_df['fiyat'].iloc[-1]
    al_tut     = ((son_fiyat - ilk_fiyat) / ilk_fiyat) * 100
    portfoy_df['tepe']  = portfoy_df['deger'].cummax()
    portfoy_df['dusus'] = (portfoy_df['deger'] - portfoy_df['tepe']) / portfoy_df['tepe'] * 100
    maks_dusus = portfoy_df['dusus'].min()
    fark       = getiri - al_tut

    islem_sayisi  = 0
    basari_orani  = 0
    ort_guven     = 0

    if len(islemler_df) > 0:
        satislar = islemler_df[islemler_df['islem'] == 'SAT']
        if len(satislar) > 0:
            kazanan      = len(satislar[satislar['kar_zarar'] > 0])
            islem_sayisi = len(satislar)
            basari_orani = kazanan / islem_sayisi * 100
            ort_guven    = islemler_df['guven'].mean() * 100

    return {
        'sembol'      : sembol,
        'getiri'      : getiri,
        'al_tut'      : al_tut,
        'fark'        : fark,
        'maks_dusus'  : maks_dusus,
        'islem'       : islem_sayisi,
        'basari'      : basari_orani,
        'guven'       : ort_guven,
        'dogruluk'    : acc * 100,
    }

def ozet_tablosu(sonuclar):
    print("\n" + "="*80)
    print("   GENEL ÖZET — 20 HİSSE ML BACKTEST SONUÇLARI")
    print("="*80)
    print(f"  {'Hisse':<12} {'ML%':>7} {'AlTut%':>7} {'Fark%':>7} "
          f"{'MaksDüş%':>9} {'İşlem':>6} {'Başarı%':>8} {'Doğr%':>7}")
    print(f"  {'─'*74}")

    ml_gecer    = []
    al_tut_gecer = []

    for s in sorted(sonuclar, key=lambda x: x['fark'], reverse=True):
        isaretg = '+' if s['getiri'] >= 0 else ''
        isaretf = '+' if s['fark']   >= 0 else ''
        isareta = '+' if s['al_tut'] >= 0 else ''
        gosterge = '✓' if s['fark'] > 0 else ' '

        print(f"  {s['sembol']:<12} "
              f"{isaretg}{s['getiri']:>6.1f}  "
              f"{isareta}{s['al_tut']:>6.1f}  "
              f"{isaretf}{s['fark']:>6.1f}  "
              f"{s['maks_dusus']:>8.1f}  "
              f"{s['islem']:>5}  "
              f"{s['basari']:>7.0f}  "
              f"{s['dogruluk']:>6.1f} {gosterge}")

        if s['fark'] > 0:
            ml_gecer.append(s['sembol'])
        else:
            al_tut_gecer.append(s['sembol'])

    print(f"  {'─'*74}")
    print(f"\n  ML stratejisi al-tut'u geçen hisseler ({len(ml_gecer)}):")
    print(f"  {', '.join([s.replace('.IS','') for s in ml_gecer])}")
    print(f"\n  Al-tut daha iyi olan hisseler ({len(al_tut_gecer)}):")
    print(f"  {', '.join([s.replace('.IS','') for s in al_tut_gecer])}")

    # Ortalamalar
    ort_ml     = np.mean([s['getiri']  for s in sonuclar])
    ort_altut  = np.mean([s['al_tut'] for s in sonuclar])
    ort_basari = np.mean([s['basari']  for s in sonuclar if s['islem'] > 0])
    print(f"\n  Ortalama ML getirisi   : %{ort_ml:>+.1f}")
    print(f"  Ortalama al-tut getiri : %{ort_altut:>+.1f}")
    print(f"  Ortalama başarı oranı  : %{ort_basari:.1f}")
    print("="*80)

def main():
    ozellikler = [
        'RSI_Norm', 'RSI_fast_Norm', 'MACD_Norm', 'MACD_hist',
        'BB_Konum', 'BB_genislik', 'MA5_Fark', 'MA20_Fark',
        'MA50_Fark', 'MA200_Fark', 'Trend_Guc', 'EMA_MA_Fark',
        'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Getiri_1g', 'Getiri_3g', 'Getiri_5g', 'Getiri_10g', 'Getiri_20g',
        'Hacim_Oran', 'Hacim_Trend', 'ATR', 'Volatilite',
        'Kanat', 'Govde', 'Yon',
        '52H_Yuzde', '52L_Yuzde', 'RSI_Trend', 'Hacim_Fiyat'
    ]

    print("\n" + "="*80)
    print("   BIST ML SİNYAL SİSTEMİ — 20 HİSSE TARAMASI")
    print("="*80)

    sonuclar = []

    for i, sembol in enumerate(HISSELER, 1):
        print(f"\n[{i:02d}/20] {sembol} işleniyor...")
        try:
            df            = veri_cek(sembol, PERIYOT)
            df            = ozellikler_ekle(df)
            df            = hedef_olustur(df)
            model, scaler, acc = model_egit(df, ozellikler)
            islemler_df, portfoy_df, son_sermaye = ml_backtest(
                df, model, scaler, ozellikler)
            sonuc         = sonuc_hesapla(
                sembol, islemler_df, portfoy_df, son_sermaye, acc)
            sonuclar.append(sonuc)
            g = sonuc['getiri']
            print(f"  Getiri: %{'+' if g>=0 else ''}{g:.1f} | "
                  f"Al-tut: %{'+' if sonuc['al_tut']>=0 else ''}{sonuc['al_tut']:.1f} | "
                  f"Fark: %{'+' if sonuc['fark']>=0 else ''}{sonuc['fark']:.1f}")
        except Exception as e:
            print(f"  HATA: {e}")

    if sonuclar:
        ozet_tablosu(sonuclar)

        # Sonuçları CSV'ye kaydet
        pd.DataFrame(sonuclar).to_csv("ml_backtest_sonuclar.csv",
                                       index=False, encoding='utf-8-sig')
        print("\n  Sonuçlar 'ml_backtest_sonuclar.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()