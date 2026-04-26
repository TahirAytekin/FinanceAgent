import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
from datetime import datetime

# ─── Ayarlar ───────────────────────────────────────────
HISSE = "EREGL.IS"   # Ereğli



PERIYOT      = "2y"        # 2 yıllık geçmiş veri
BASLANGIC_SERMAYE = 100000 # 100.000 TL başlangıç
KOMISYON     = 0.001       # %0.1 komisyon (her alım/satımda)
# ───────────────────────────────────────────────────────

def veri_cek(sembol, periyot):
    df = yf.Ticker(sembol).history(period=periyot, interval="1d")
    df = df[['Open','High','Low','Close','Volume']]
    df.index = df.index.tz_localize(None)
    return df

def gostergeler_ekle(df):
    df['RSI']         = ta.rsi(df['Close'], length=14)
    macd              = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_sinyal'] = macd['MACDs_12_26_9']
    df['MA20']        = ta.sma(df['Close'], length=20)
    df['MA50']        = ta.sma(df['Close'], length=50)
    bb                = ta.bbands(df['Close'], length=20, std=2)
    bb_s              = bb.columns.tolist()
    df['BB_ust']      = bb[bb_s[2]]
    df['BB_alt']      = bb[bb_s[0]]
    return df.dropna()

def sinyal_uret(satir, onceki, df_slice):
    puanlar = {}

    # ── Trend filtresi — düşüş trendinde AL sinyali engelle ──
    ma20 = satir['MA20']
    ma50 = satir['MA50']
    yukselis_trendi = ma20 > ma50   # MA20 > MA50 ise yükseliş trendi

    rsi = satir['RSI']
    puanlar['RSI'] = 1 if rsi < 35 else (-1 if rsi > 65 else 0)

    macd_yukari = (onceki['MACD'] < onceki['MACD_sinyal'] and
                   satir['MACD']  > satir['MACD_sinyal'])
    macd_asagi  = (onceki['MACD'] > onceki['MACD_sinyal'] and
                   satir['MACD']  < satir['MACD_sinyal'])
    if macd_yukari:
        puanlar['MACD'] = 2
    elif macd_asagi:
        puanlar['MACD'] = -2
    elif satir['MACD'] > satir['MACD_sinyal']:
        puanlar['MACD'] = 1
    else:
        puanlar['MACD'] = -1

    fiyat = satir['Close']
    if fiyat > ma20 and fiyat > ma50:
        puanlar['MA'] = 1
    elif fiyat < ma20 and fiyat < ma50:
        puanlar['MA'] = -1
    else:
        puanlar['MA'] = 0

    if fiyat <= satir['BB_alt']:
        puanlar['BB'] = 1
    elif fiyat >= satir['BB_ust']:
        puanlar['BB'] = -1
    else:
        puanlar['BB'] = 0

    toplam = sum(puanlar.values())

    # Trend filtresi uygula — düşüş trendinde AL engelle
    if toplam >= 2:
        if yukselis_trendi:
            return "AL"
        else:
            return "BEKLE"   # Düşüş trendinde AL sinyali engellendi
    elif toplam <= -2:
        return "SAT"
    return "BEKLE"

def backtest_calistir(df):
    sermaye      = BASLANGIC_SERMAYE
    hisse_adedi  = 0
    pozisyon     = False   # True = elimizde hisse var
    alis_fiyati  = 0
    STOP_LOSS    = 0.07   # %7 stop-loss
    islemler     = []
    portfoy_gecmis = []

    for i in range(1, len(df)):
        satir  = df.iloc[i]
        onceki = df.iloc[i - 1]
        fiyat  = satir['Close']
        tarih  = df.index[i]
        karar  = sinyal_uret(satir, onceki, df.iloc[:i])

        # AL sinyali — pozisyon yoksa al
        if karar == "AL" and not pozisyon:
            hisse_adedi = int(sermaye / fiyat)
            maliyet     = hisse_adedi * fiyat * (1 + KOMISYON)
            if hisse_adedi > 0:
                sermaye  -= maliyet
                pozisyon  = True
                alis_fiyati = fiyat
                islemler.append({
                    'tarih' : tarih,
                    'islem' : 'AL',
                    'fiyat' : fiyat,
                    'adet'  : hisse_adedi,
                    'tutar' : maliyet,
                    'sermaye': sermaye
                })
                
# Stop-loss kontrolü
        if pozisyon and alis_fiyati > 0:
            kayip_orani = (fiyat - alis_fiyati) / alis_fiyati
            if kayip_orani <= -STOP_LOSS:
                karar = "SAT"

        # AL sinyali — pozisyon yoksa al
        if karar == "AL" and not pozisyon:
            hisse_adedi = int(sermaye / fiyat)
            maliyet     = hisse_adedi * fiyat * (1 + KOMISYON)
            if hisse_adedi > 0:
                sermaye    -= maliyet
                pozisyon    = True
                alis_fiyati = fiyat
                islemler.append({
                    'tarih'  : tarih,
                    'islem'  : 'AL',
                    'fiyat'  : fiyat,
                    'adet'   : hisse_adedi,
                    'tutar'  : maliyet,
                    'sermaye': sermaye
                })

        # SAT sinyali — pozisyon varsa sat
        elif karar == "SAT" and pozisyon:
            gelir     = hisse_adedi * fiyat * (1 - KOMISYON)
            sermaye  += gelir
            kar_zarar = gelir - islemler[-1]['tutar']
            islemler.append({
                'tarih'    : tarih,
                'islem'    : 'SAT',
                'fiyat'    : fiyat,
                'adet'     : hisse_adedi,
                'tutar'    : gelir,
                'kar_zarar': kar_zarar,
                'sermaye'  : sermaye
            })
            hisse_adedi = 0
            pozisyon    = False
            alis_fiyati = 0

        # Portföy değerini kaydet
        portfoy_degeri = sermaye + (hisse_adedi * fiyat)
        portfoy_gecmis.append({
            'tarih'  : tarih,
            'deger'  : portfoy_degeri,
            'fiyat'  : fiyat
        })

    # Son pozisyonu kapat
    if pozisyon:
        son_fiyat = df['Close'].iloc[-1]
        gelir     = hisse_adedi * son_fiyat * (1 - KOMISYON)
        sermaye  += gelir

    return pd.DataFrame(islemler), pd.DataFrame(portfoy_gecmis), sermaye

def istatistikler(islemler_df, portfoy_df, son_sermaye):
    print("\n" + "="*55)
    print("       BACKTEST SONUÇLARI")
    print("="*55)

    toplam_kar = son_sermaye - BASLANGIC_SERMAYE
    getiri_yuzde = (toplam_kar / BASLANGIC_SERMAYE) * 100

    print(f"\n  Başlangıç sermayesi : {BASLANGIC_SERMAYE:>12,.0f} TL")
    print(f"  Son sermaye         : {son_sermaye:>12,.0f} TL")
    print(f"  Toplam kar/zarar    : {toplam_kar:>+12,.0f} TL")
    print(f"  Toplam getiri       : %{getiri_yuzde:>+.1f}")

    if len(islemler_df) > 0:
        satislar = islemler_df[islemler_df['islem'] == 'SAT']
        if len(satislar) > 0:
            kazanan = len(satislar[satislar['kar_zarar'] > 0])
            kaybeden = len(satislar[satislar['kar_zarar'] <= 0])
            basari_orani = (kazanan / len(satislar)) * 100

            print(f"\n  Toplam işlem        : {len(satislar)}")
            print(f"  Kazanan işlem       : {kazanan}")
            print(f"  Kaybeden işlem      : {kaybeden}")
            print(f"  Başarı oranı        : %{basari_orani:.1f}")

            ort_kar = satislar[satislar['kar_zarar'] > 0]['kar_zarar'].mean()
            ort_zarar = satislar[satislar['kar_zarar'] <= 0]['kar_zarar'].mean()
            print(f"\n  Ort. kazanç         : {ort_kar:>+,.0f} TL")
            print(f"  Ort. kayıp          : {ort_zarar:>+,.0f} TL")

    # Maksimum düşüş (Drawdown)
    portfoy_df['tepe'] = portfoy_df['deger'].cummax()
    portfoy_df['dusus'] = (portfoy_df['deger'] - portfoy_df['tepe']) / portfoy_df['tepe'] * 100
    maks_dusus = portfoy_df['dusus'].min()
    print(f"\n  Maks. düşüş         : %{maks_dusus:.1f}")

    # Al-tut karşılaştırması
    ilk_fiyat = portfoy_df['fiyat'].iloc[0]
    son_fiyat = portfoy_df['fiyat'].iloc[-1]
    al_tut_getiri = ((son_fiyat - ilk_fiyat) / ilk_fiyat) * 100
    print(f"\n  Al-tut getirisi     : %{al_tut_getiri:>+.1f}  (karşılaştırma)")
    print(f"  Strateji getirisi   : %{getiri_yuzde:>+.1f}")

    fark = getiri_yuzde - al_tut_getiri
    print(f"  Fark                : %{fark:>+.1f}  ", end="")
    print("(strateji daha iyi)" if fark > 0 else "(al-tut daha iyi)")
    print("="*55)

def grafik_ciz(df, islemler_df, portfoy_df, sembol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10),
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f"{sembol} — Backtest Sonuçları", fontsize=14, fontweight='bold')

    # Fiyat + alım/satım noktaları
    ax1.plot(df.index, df['Close'], color='#94a3b8', lw=1, label='Fiyat', zorder=1)

    if len(islemler_df) > 0:
        alislar = islemler_df[islemler_df['islem'] == 'AL']
        satislar = islemler_df[islemler_df['islem'] == 'SAT']

        ax1.scatter(alislar['tarih'], alislar['fiyat'],
                    marker='^', color='#16a34a', s=120, zorder=3,
                    label='Alım', linewidths=0.5, edgecolors='white')
        ax1.scatter(satislar['tarih'], satislar['fiyat'],
                    marker='v', color='#dc2626', s=120, zorder=3,
                    label='Satım', linewidths=0.5, edgecolors='white')

    ax1.set_ylabel('Fiyat (TL)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Portföy değeri vs Al-tut
    ilk_fiyat  = portfoy_df['fiyat'].iloc[0]
    al_tut     = BASLANGIC_SERMAYE * (portfoy_df['fiyat'] / ilk_fiyat)

    ax2.plot(portfoy_df['tarih'], portfoy_df['deger'],
             color='#2563eb', lw=1.5, label='Strateji')
    ax2.plot(portfoy_df['tarih'], al_tut,
             color='#f59e0b', lw=1.5, linestyle='--', label='Al-tut')
    ax2.axhline(y=BASLANGIC_SERMAYE, color='gray', lw=0.8, linestyle=':')
    ax2.set_ylabel('Portföy Değeri (TL)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{sembol.replace('.IS','')}_backtest.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Grafik kaydedildi: {sembol.replace('.IS','')}_backtest.png")

def main():
    print(f"\n{HISSE} backtest başlıyor... (2 yıllık veri)")
    df = veri_cek(HISSE, PERIYOT)
    df = gostergeler_ekle(df)
    islemler_df, portfoy_df, son_sermaye = backtest_calistir(df)
    istatistikler(islemler_df, portfoy_df, son_sermaye)
    grafik_ciz(df, islemler_df, portfoy_df, HISSE)

if __name__ == "__main__":
    main()