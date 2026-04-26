import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

def piyasa_rejimi_tespit():
    """
    BIST100 endeksini analiz ederek piyasa rejimini tespit eder.
    
    Rejimler:
    GUCLU_BOGA  — güçlü yükseliş, agresif al
    BOGA        — yükseliş trendi, al sinyallerine güven
    YATAY       — kararsız, dikkatli ol
    AYI         — düşüş trendi, sat sinyallerine güven  
    GUCLU_AYI   — güçlü düşüş, pozisyon açma
    """
    print("Piyasa rejimi analiz ediliyor...")

    # BIST100 endeksi
    bist  = yf.Ticker("XU100.IS").history(period="1y", interval="1d")
    bist.index = bist.index.tz_localize(None)

    # USD/TRY kuru
    usdtry = yf.Ticker("USDTRY=X").history(period="1y", interval="1d")
    usdtry.index = usdtry.index.tz_localize(None)

    # Göstergeler
    bist['MA20']       = ta.sma(bist['Close'], length=20)
    bist['MA50']       = ta.sma(bist['Close'], length=50)
    bist['MA200']      = ta.sma(bist['Close'], length=200)
    bist['RSI']        = ta.rsi(bist['Close'], length=14)
    bist['ATR']        = ta.atr(bist['High'], bist['Low'], bist['Close'], length=14)
    bist['Volatilite'] = bist['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    bist['Getiri_1ay'] = bist['Close'].pct_change(20)
    bist['Getiri_3ay'] = bist['Close'].pct_change(60)

    son = bist.dropna().iloc[-1]

    # Kur analizi
    usdtry_son    = usdtry['Close'].iloc[-1]
    usdtry_1ay    = usdtry['Close'].iloc[-20]
    kur_degisim   = (usdtry_son - usdtry_1ay) / usdtry_1ay * 100

    # Puan sistemi
    puanlar = {}

    # Trend puanı
    if son['Close'] > son['MA20'] > son['MA50'] > son['MA200']:
        puanlar['trend'] = 2
    elif son['Close'] > son['MA50']:
        puanlar['trend'] = 1
    elif son['Close'] > son['MA200']:
        puanlar['trend'] = 0
    elif son['Close'] < son['MA50']:
        puanlar['trend'] = -1
    else:
        puanlar['trend'] = -2

    # RSI puanı
    if son['RSI'] > 60:
        puanlar['rsi'] = 1
    elif son['RSI'] < 40:
        puanlar['rsi'] = -1
    else:
        puanlar['rsi'] = 0

    # Momentum puanı
    if son['Getiri_1ay'] > 0.05:
        puanlar['momentum'] = 2
    elif son['Getiri_1ay'] > 0:
        puanlar['momentum'] = 1
    elif son['Getiri_1ay'] > -0.05:
        puanlar['momentum'] = -1
    else:
        puanlar['momentum'] = -2

    # Kur puanı — TL değer kaybı borsayı olumsuz etkiler
    if kur_degisim > 5:
        puanlar['kur'] = -2
    elif kur_degisim > 2:
        puanlar['kur'] = -1
    elif kur_degisim > -1:
        puanlar['kur'] = 0
    else:
        puanlar['kur'] = 1

    # Volatilite puanı — yüksek volatilite risk demek
    vol = son['Volatilite']
    if vol < 0.20:
        puanlar['volatilite'] = 1
    elif vol < 0.35:
        puanlar['volatilite'] = 0
    else:
        puanlar['volatilite'] = -1

    toplam = sum(puanlar.values())

    if toplam >= 5:
        rejim = "GUCLU_BOGA"
        emoji = "🚀"
        aciklama = "Güçlü yükseliş trendi — agresif al sinyalleri geçerli"
        guven_carpani = 1.3
    elif toplam >= 2:
        rejim = "BOGA"
        emoji = "🟢"
        aciklama = "Yükseliş trendi — al sinyallerine güven"
        guven_carpani = 1.1
    elif toplam >= -1:
        rejim = "YATAY"
        emoji = "🟡"
        aciklama = "Kararsız piyasa — dikkatli ol, güven eşiğini yükselt"
        guven_carpani = 0.9
    elif toplam >= -4:
        rejim = "AYI"
        emoji = "🔴"
        aciklama = "Düşüş trendi — al sinyallerini yoksay, sat sinyallerine güven"
        guven_carpani = 0.7
    else:
        rejim = "GUCLU_AYI"
        emoji = "❌"
        aciklama = "Güçlü düşüş — yeni pozisyon açma"
        guven_carpani = 0.5

    sonuc = {
        'rejim'          : rejim,
        'emoji'          : emoji,
        'aciklama'       : aciklama,
        'toplam_puan'    : toplam,
        'puanlar'        : puanlar,
        'guven_carpani'  : guven_carpani,
        'bist_son'       : son['Close'],
        'bist_ma200'     : son['MA200'],
        'bist_rsi'       : son['RSI'],
        'bist_getiri_1ay': son['Getiri_1ay'] * 100,
        'bist_getiri_3ay': son['Getiri_3ay'] * 100,
        'volatilite'     : vol * 100,
        'usdtry'         : usdtry_son,
        'kur_degisim_1ay': kur_degisim,
    }

    return sonuc

def rejim_raporu(sonuc):
    print("\n" + "="*60)
    print("   PİYASA REJİMİ ANALİZİ")
    print("="*60)
    print(f"\n  Rejim      : {sonuc['emoji']}  {sonuc['rejim']}")
    print(f"  Açıklama   : {sonuc['aciklama']}")
    print(f"  Puan       : {sonuc['toplam_puan']:+d} / 6")
    print(f"\n  ── BIST100 ─────────────────────────────────")
    print(f"  Endeks     : {sonuc['bist_son']:,.0f}")
    print(f"  MA200      : {sonuc['bist_ma200']:,.0f}  "
          f"({'ÜSTÜNde' if sonuc['bist_son'] > sonuc['bist_ma200'] else 'ALTINda'})")
    print(f"  RSI        : {sonuc['bist_rsi']:.1f}")
    print(f"  1 Aylık    : %{sonuc['bist_getiri_1ay']:+.1f}")
    print(f"  3 Aylık    : %{sonuc['bist_getiri_3ay']:+.1f}")
    print(f"  Volatilite : %{sonuc['volatilite']:.1f} (yıllık)")
    print(f"\n  ── KUR ─────────────────────────────────────")
    print(f"  USD/TRY    : {sonuc['usdtry']:.2f}")
    print(f"  1 Aylık Δ  : %{sonuc['kur_degisim_1ay']:+.1f}")
    print(f"\n  ── Puan Detayı ─────────────────────────────")
    for k, v in sonuc['puanlar'].items():
        bar = ('▲' * max(0, v)) + ('▼' * max(0, -v))
        print(f"  {k:<12}: {v:+2d}  {bar}")
    print(f"\n  ML Güven Çarpanı : x{sonuc['guven_carpani']}")
    print(f"  (AL sinyali güven eşiği bu çarpanla ayarlanır)")
    print("="*60)

def main():
    sonuc = piyasa_rejimi_tespit()
    rejim_raporu(sonuc)
    return sonuc

if __name__ == "__main__":
    main()