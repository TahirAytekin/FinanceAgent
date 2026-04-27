import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import pandas_ta as ta

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
    # RSI (14 günlük)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_sinyal'] = macd['MACDs_12_26_9']
    df['MACD_hist']   = macd['MACDh_12_26_9']

    # Bollinger Bantları (20 günlük)
    bb = ta.bbands(df['Close'], length=20, std=2)
    bb_sutunlar = bb.columns.tolist()
    df['BB_ust']  = bb[bb_sutunlar[2]]
    df['BB_orta'] = bb[bb_sutunlar[1]]
    df['BB_alt']  = bb[bb_sutunlar[0]]

    # Hareketli Ortalamalar
    df['MA20']  = ta.sma(df['Close'], length=20)
    df['MA50']  = ta.sma(df['Close'], length=50)

    return df

def destek_direnc_bul(df, hassasiyet=5):
    """
    Yerel minimum = destek noktası
    Yerel maksimum = direnç noktası
    hassasiyet: kaç günlük pencerede zirve/dip aranır
    """
    destekler  = []
    direncler = []
    
    fiyatlar = df['Close'].values
    
    for i in range(hassasiyet, len(fiyatlar) - hassasiyet):
        pencere = fiyatlar[i - hassasiyet : i + hassasiyet + 1]
        
        if fiyatlar[i] == min(pencere):
            destekler.append((df.index[i], fiyatlar[i]))
        
        if fiyatlar[i] == max(pencere):
            direncler.append((df.index[i], fiyatlar[i]))
    
    return destekler, direncler

def onemli_seviyeler(destekler, direncler, son_fiyat, tolerans=0.03):
    """
    Son fiyata en yakın destek ve direnç seviyelerini seç
    tolerans: %3 yakınındakileri grupla
    """
    # Gruplama — birbirine çok yakın seviyeleri birleştir
    def grupla(seviyeler):
        if not seviyeler:
            return []
        fiyatlar = sorted([f for _, f in seviyeler])
        gruplar = []
        grup = [fiyatlar[0]]
        for f in fiyatlar[1:]:
            if (f - grup[-1]) / grup[-1] < tolerans:
                grup.append(f)
            else:
                gruplar.append(np.mean(grup))
                grup = [f]
        gruplar.append(np.mean(grup))
        return gruplar

    d_seviyeleri = grupla(destekler)
    r_seviyeleri = grupla(direncler)

    # Son fiyatın altındakiler destek, üstündekiler direnç
    aktif_destekler  = sorted([f for f in d_seviyeleri if f < son_fiyat], reverse=True)[:3]
    aktif_direncler = sorted([f for f in r_seviyeleri if f > son_fiyat])[:3]

    return aktif_destekler, aktif_direncler

def grafik_ciz(df, sembol, destekler, direncler):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12),
                             gridspec_kw={'height_ratios': [4, 2, 2]})
    fig.suptitle(f"{sembol} — Teknik Analiz", fontsize=15, fontweight='bold')

    # ── 1. GRAFİK: Fiyat + BB + MA + Destek/Direnç ──
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'],  color='#1e40af', lw=1.8, label='Kapanış', zorder=3)
    ax1.plot(df.index, df['MA20'],   color='#f59e0b', lw=1,   label='MA20',    linestyle='--')
    ax1.plot(df.index, df['MA50'],   color='#8b5cf6', lw=1,   label='MA50',    linestyle='--')
    ax1.plot(df.index, df['BB_ust'], color='#94a3b8', lw=0.8, linestyle=':')
    ax1.plot(df.index, df['BB_alt'], color='#94a3b8', lw=0.8, linestyle=':')
    ax1.fill_between(df.index, df['BB_ust'], df['BB_alt'], alpha=0.05, color='gray')

    # Destek çizgileri (yeşil)
    for seviye in destekler:
        ax1.axhline(y=seviye, color='#16a34a', linestyle='--', lw=1.2, alpha=0.8)
        ax1.text(df.index[-1], seviye, f'  D: {seviye:.1f}',
                 color='#16a34a', fontsize=9, va='center', fontweight='bold')

    # Direnç çizgileri (kırmızı)
    for seviye in direncler:
        ax1.axhline(y=seviye, color='#dc2626', linestyle='--', lw=1.2, alpha=0.8)
        ax1.text(df.index[-1], seviye, f'  R: {seviye:.1f}',
                 color='#dc2626', fontsize=9, va='center', fontweight='bold')

    ax1.set_ylabel('Fiyat (TL)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── 2. GRAFİK: RSI ──
    ax2 = axes[1]
    ax2.plot(df.index, df['RSI'], color='#0891b2', lw=1.5)
    ax2.axhline(y=70, color='#dc2626', linestyle='--', lw=1, alpha=0.7, label='Aşırı alım (70)')
    ax2.axhline(y=30, color='#16a34a', linestyle='--', lw=1, alpha=0.7, label='Aşırı satım (30)')
    ax2.fill_between(df.index, df['RSI'], 70,
                     where=(df['RSI'] >= 70), alpha=0.2, color='#dc2626')
    ax2.fill_between(df.index, df['RSI'], 30,
                     where=(df['RSI'] <= 30), alpha=0.2, color='#16a34a')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── 3. GRAFİK: MACD ──
    ax3 = axes[2]
    ax3.plot(df.index, df['MACD'],        color='#2563eb', lw=1.5, label='MACD')
    ax3.plot(df.index, df['MACD_sinyal'], color='#f97316', lw=1.5, label='Sinyal')
    renkler = ['#16a34a' if v >= 0 else '#dc2626' for v in df['MACD_hist']]
    ax3.bar(df.index, df['MACD_hist'], color=renkler, alpha=0.6, width=0.8, label='Histogram')
    ax3.axhline(y=0, color='gray', lw=0.8)
    ax3.set_ylabel('MACD')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{sembol.replace('.IS','')}_teknik.png", dpi=150, bbox_inches='tight')
    plt.show()

def analiz_ozeti(df, destekler, direncler):
    son      = df['Close'].iloc[-1]
    rsi      = df['RSI'].iloc[-1]
    macd     = df['MACD'].iloc[-1]
    macd_sig = df['MACD_sinyal'].iloc[-1]
    ma20     = df['MA20'].iloc[-1]
    ma50     = df['MA50'].iloc[-1]

    print("\n" + "="*50)
    print("       TEKNİK ANALİZ ÖZETİ")
    print("="*50)
    print(f"Son fiyat : {son:.2f} TL")
    print(f"MA20      : {ma20:.2f} TL  →  Fiyat MA20'nin {'ÜSTÜNde' if son > ma20 else 'ALTINda'}")
    print(f"MA50      : {ma50:.2f} TL  →  Fiyat MA50'nin {'ÜSTÜNde' if son > ma50 else 'ALTINda'}")
    print(f"\nRSI (14)  : {rsi:.1f}", end="  →  ")
    if rsi > 70:
        print("AŞIRI ALIM — düzeltme gelebilir")
    elif rsi < 30:
        print("AŞIRI SATIM — toparlanma gelebilir")
    else:
        print("Nötr bölge")
    print(f"\nMACD      : {macd:.3f}")
    print(f"MACD Sinyal: {macd_sig:.3f}  →  MACD sinyal çizgisinin {'ÜSTÜNde (Alım sinyali)' if macd > macd_sig else 'ALTINda (Satım sinyali)'}")
    print("\n--- Yakın Destek Seviyeleri ---")
    for i, d in enumerate(destekler, 1):
        uzaklik = ((son - d) / son) * 100
        print(f"  Destek {i}: {d:.2f} TL  (-%{uzaklik:.1f} uzakta)")
    print("\n--- Yakın Direnç Seviyeleri ---")
    for i, r in enumerate(direncler, 1):
        uzaklik = ((r - son) / son) * 100
        print(f"  Direnç {i}: {r:.2f} TL  (+%{uzaklik:.1f} uzakta)")
    print("="*50)

def main():
    print(f"{HISSE} teknik analizi başlıyor...\n")
    df = veri_cek(HISSE, PERIYOT)
    df = gostergeler_ekle(df)
    
    ham_destekler, ham_direncler = destek_direnc_bul(df, hassasiyet=5)
    son_fiyat = df['Close'].iloc[-1]
    aktif_destekler, aktif_direncler = onemli_seviyeler(
        ham_destekler, ham_direncler, son_fiyat
    )
    
    analiz_ozeti(df, aktif_destekler, aktif_direncler)
    grafik_ciz(df, HISSE, aktif_destekler, aktif_direncler)

if __name__ == "__main__":
    main()