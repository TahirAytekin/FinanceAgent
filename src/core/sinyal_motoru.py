import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime

# ─── Ayarlar ───────────────────────────────────────────
HISSE   = "THYAO.IS"
PERIYOT = "6mo"
# ───────────────────────────────────────────────────────

def veri_cek(sembol, periyot):
    df = yf.Ticker(sembol).history(period=periyot, interval="1d")
    df = df[['Open','High','Low','Close','Volume']]
    df.index = df.index.tz_localize(None)
    return df

def gostergeler_ekle(df):
    df['RSI']  = ta.rsi(df['Close'], length=14)
    macd       = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_sinyal'] = macd['MACDs_12_26_9']
    df['MA20'] = ta.sma(df['Close'], length=20)
    df['MA50'] = ta.sma(df['Close'], length=50)
    bb         = ta.bbands(df['Close'], length=20, std=2)
    bb_sutunlar = bb.columns.tolist()
    df['BB_ust'] = bb[bb_sutunlar[2]]
    df['BB_alt'] = bb[bb_sutunlar[0]]
    return df

def destek_direnc_bul(df, hassasiyet=5):
    destekler, direncler = [], []
    fiyatlar = df['Close'].values
    for i in range(hassasiyet, len(fiyatlar) - hassasiyet):
        pencere = fiyatlar[i - hassasiyet: i + hassasiyet + 1]
        if fiyatlar[i] == min(pencere):
            destekler.append(fiyatlar[i])
        if fiyatlar[i] == max(pencere):
            direncler.append(fiyatlar[i])
    return destekler, direncler

def onemli_seviyeler(destekler, direncler, son_fiyat, tolerans=0.03):
    def grupla(seviyeler):
        if not seviyeler:
            return []
        fiyatlar = sorted(seviyeler)
        gruplar, grup = [], [fiyatlar[0]]
        for f in fiyatlar[1:]:
            if (f - grup[-1]) / grup[-1] < tolerans:
                grup.append(f)
            else:
                gruplar.append(np.mean(grup))
                grup = [f]
        gruplar.append(np.mean(grup))
        return gruplar

    d = grupla(destekler)
    r = grupla(direncler)
    aktif_d = sorted([f for f in d if f < son_fiyat], reverse=True)[:3]
    aktif_r = sorted([f for f in r if f > son_fiyat])[:3]
    return aktif_d, aktif_r

def sinyal_uret(df, destekler, direncler):
    """
    Her gösterge için puan üretir.
    +1 = alım sinyali
    -1 = satım sinyali
     0 = nötr
    Toplam puan → karar
    """
    son   = df.iloc[-1]
    onceki = df.iloc[-2]
    puanlar = {}
    nedenler = []

    # ── RSI ──────────────────────────────────────────
    rsi = son['RSI']
    if rsi < 35:
        puanlar['RSI'] = 1
        nedenler.append(f"RSI {rsi:.1f} — aşırı satım bölgesi (AL sinyali)")
    elif rsi > 65:
        puanlar['RSI'] = -1
        nedenler.append(f"RSI {rsi:.1f} — aşırı alım bölgesi (SAT sinyali)")
    else:
        puanlar['RSI'] = 0
        nedenler.append(f"RSI {rsi:.1f} — nötr bölge")

    # ── MACD Kesişimi ────────────────────────────────
    macd_yukari = (onceki['MACD'] < onceki['MACD_sinyal'] and
                   son['MACD']    > son['MACD_sinyal'])
    macd_asagi  = (onceki['MACD'] > onceki['MACD_sinyal'] and
                   son['MACD']    < son['MACD_sinyal'])

    if macd_yukari:
        puanlar['MACD'] = 2   # Kesişim güçlü sinyaldir
        nedenler.append("MACD sinyal çizgisini yukarı kesti (güçlü AL)")
    elif macd_asagi:
        puanlar['MACD'] = -2
        nedenler.append("MACD sinyal çizgisini aşağı kesti (güçlü SAT)")
    elif son['MACD'] > son['MACD_sinyal']:
        puanlar['MACD'] = 1
        nedenler.append("MACD sinyal çizgisinin üstünde (zayıf AL)")
    else:
        puanlar['MACD'] = -1
        nedenler.append("MACD sinyal çizgisinin altında (zayıf SAT)")

    # ── Hareketli Ortalama ───────────────────────────
    fiyat = son['Close']
    if fiyat > son['MA20'] and fiyat > son['MA50']:
        puanlar['MA'] = 1
        nedenler.append(f"Fiyat hem MA20 hem MA50 üstünde (yükseliş trendi)")
    elif fiyat < son['MA20'] and fiyat < son['MA50']:
        puanlar['MA'] = -1
        nedenler.append(f"Fiyat hem MA20 hem MA50 altında (düşüş trendi)")
    else:
        puanlar['MA'] = 0
        nedenler.append(f"Fiyat MA'lar arasında (kararsız)")

    # ── Bollinger Bantları ───────────────────────────
    if fiyat <= son['BB_alt']:
        puanlar['BB'] = 1
        nedenler.append("Fiyat alt Bollinger bandına değdi (AL sinyali)")
    elif fiyat >= son['BB_ust']:
        puanlar['BB'] = -1
        nedenler.append("Fiyat üst Bollinger bandına değdi (SAT sinyali)")
    else:
        puanlar['BB'] = 0
        nedenler.append("Fiyat Bollinger bantları içinde (nötr)")

    # ── Destek/Direnç Yakınlığı ──────────────────────
    if destekler:
        en_yakin_destek = destekler[0]
        destek_uzaklik  = (fiyat - en_yakin_destek) / fiyat
        if destek_uzaklik < 0.02:   # %2 yakınında
            puanlar['SD'] = 2
            nedenler.append(f"Fiyat güçlü destek seviyesine çok yakın: {en_yakin_destek:.2f} TL (güçlü AL)")
        else:
            puanlar['SD'] = 0
            nedenler.append(f"En yakın destek: {en_yakin_destek:.2f} TL")
    
    if direncler:
        en_yakin_direnc = direncler[0]
        direnc_uzaklik  = (en_yakin_direnc - fiyat) / fiyat
        if direnc_uzaklik < 0.02:   # %2 yakınında
            puanlar['SD'] = puanlar.get('SD', 0) - 2
            nedenler.append(f"Fiyat güçlü direnç seviyesine çok yakın: {en_yakin_direnc:.2f} TL (güçlü SAT)")

    # ── Toplam Puan & Karar ──────────────────────────
    toplam = sum(puanlar.values())
    maks   = 8   # Teorik maksimum puan

    if toplam >= 3:
        karar = "AL"
        guc   = min(int((toplam / maks) * 100), 100)
    elif toplam <= -3:
        karar = "SAT"
        guc   = min(int((abs(toplam) / maks) * 100), 100)
    else:
        karar = "BEKLE"
        guc   = 50

    return {
        'karar'   : karar,
        'guc'     : guc,
        'toplam'  : toplam,
        'puanlar' : puanlar,
        'nedenler': nedenler,
        'fiyat'   : fiyat,
        'rsi'     : rsi,
        'destekler': destekler,
        'direncler': direncler,
    }

def rapor_yazdir(sembol, sonuc):
    zaman = datetime.now().strftime("%d.%m.%Y %H:%M")
    karar = sonuc['karar']

    semboller = {'AL': '▲  AL', 'SAT': '▼  SAT', 'BEKLE': '■  BEKLE'}
    renkler   = {'AL': '✅', 'SAT': '🔴', 'BEKLE': '🟡'}

    print("\n" + "="*55)
    print(f"  {renkler[karar]}  SİNYAL RAPORU — {sembol}")
    print(f"  Tarih : {zaman}")
    print("="*55)
    print(f"\n  KARAR  :  {semboller[karar]}")
    print(f"  GÜÇ    :  %{sonuc['guc']}  (Puan: {sonuc['toplam']:+d})")
    print(f"  FİYAT  :  {sonuc['fiyat']:.2f} TL")

    print("\n  ── Gösterge Puanları ──────────────────────")
    for gosterge, puan in sonuc['puanlar'].items():
        bar = "+" * max(0, puan) + "-" * max(0, -puan)
        print(f"  {gosterge:<6}: {puan:+2d}  {bar}")

    print("\n  ── Nedenler ───────────────────────────────")
    for neden in sonuc['nedenler']:
        print(f"  • {neden}")

    print("\n  ── Destek & Direnç ────────────────────────")
    for i, d in enumerate(sonuc['destekler'], 1):
        uzaklik = ((sonuc['fiyat'] - d) / sonuc['fiyat']) * 100
        print(f"  Destek {i}: {d:.2f} TL  (aşağıda %{uzaklik:.1f})")
    for i, r in enumerate(sonuc['direncler'], 1):
        uzaklik = ((r - sonuc['fiyat']) / sonuc['fiyat']) * 100
        print(f"  Direnç {i}: {r:.2f} TL  (yukarıda %{uzaklik:.1f})")

    # Öneri
    print("\n  ── Öneri ──────────────────────────────────")
    if karar == "AL":
        if sonuc['destekler']:
            stop = sonuc['destekler'][0] * 0.98
            hedef = sonuc['direncler'][0] if sonuc['direncler'] else sonuc['fiyat'] * 1.05
            print(f"  Giriş  : {sonuc['fiyat']:.2f} TL civarı")
            print(f"  Hedef  : {hedef:.2f} TL")
            print(f"  Stop   : {stop:.2f} TL  (destek altı -%2)")
    elif karar == "SAT":
        print(f"  Mevcut pozisyonu koru veya çık")
    else:
        print(f"  Daha net sinyal oluşana kadar bekle")

    print("="*55 + "\n")

    # Dosyaya kaydet
    with open("sinyaller.txt", "a", encoding="utf-8") as f:
        f.write(f"{zaman} | {sembol} | {karar} | Puan:{sonuc['toplam']:+d} | "
                f"Fiyat:{sonuc['fiyat']:.2f} | RSI:{sonuc['rsi']:.1f}\n")
    print("  Sinyal 'sinyaller.txt' dosyasına kaydedildi.")

def main():
    print(f"\n{HISSE} sinyal analizi başlıyor...")
    df               = veri_cek(HISSE, PERIYOT)
    df               = gostergeler_ekle(df)
    ham_d, ham_r     = destek_direnc_bul(df)
    son_fiyat        = df['Close'].iloc[-1]
    destekler, direncler = onemli_seviyeler(ham_d, ham_r, son_fiyat)
    sonuc            = sinyal_uret(df, destekler, direncler)
    rapor_yazdir(HISSE, sonuc)

if __name__ == "__main__":
    main()