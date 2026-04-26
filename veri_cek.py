import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─── Ayarlar ───────────────────────────────────────────
HISSE    = "THYAO.IS"   # .IS eki = Borsa İstanbul
PERIYOT  = "6mo"        # 1mo, 3mo, 6mo, 1y, 2y
ARALIK   = "1d"         # 1d = günlük, 1wk = haftalık
# ───────────────────────────────────────────────────────

def veri_cek(sembol, periyot, aralik):
    print(f"{sembol} verisi çekiliyor...")
    hisse = yf.Ticker(sembol)
    df = hisse.history(period=periyot, interval=aralik)
    
    if df.empty:
        print("HATA: Veri gelmedi. Sembolü kontrol et.")
        return None
    
    # Gereksiz sütunları kaldır
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = df.index.tz_localize(None)  # Timezone temizle
    
    print(f"Toplam {len(df)} günlük veri alındı.")
    print(f"İlk tarih : {df.index[0].strftime('%d.%m.%Y')}")
    print(f"Son tarih : {df.index[-1].strftime('%d.%m.%Y')}")
    print(f"Son kapanış: {df['Close'].iloc[-1]:.2f} TL\n")
    
    return df

def grafik_ciz(df, sembol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"{sembol} — Fiyat & Hacim", fontsize=14)
    
    # Fiyat grafiği
    ax1.plot(df.index, df['Close'], color='#2563eb', linewidth=1.5, label='Kapanış')
    ax1.fill_between(df.index, df['Close'], alpha=0.1, color='#2563eb')
    ax1.set_ylabel('Fiyat (TL)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Hacim grafiği
    renkler = ['#16a34a' if df['Close'].iloc[i] >= df['Open'].iloc[i]
               else '#dc2626' for i in range(len(df))]
    ax2.bar(df.index, df['Volume'], color=renkler, alpha=0.7, width=0.8)
    ax2.set_ylabel('Hacim')
    ax2.grid(True, alpha=0.3)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.tight_layout()
    plt.savefig(f"{sembol.replace('.IS','')}_grafik.png", dpi=150)
    plt.show()
    print("Grafik kaydedildi.")

def main():
    df = veri_cek(HISSE, PERIYOT, ARALIK)
    if df is not None:
        grafik_ciz(df, HISSE)
        # CSV olarak kaydet (sonraki adımlarda kullanacağız)
        df.to_csv(f"{HISSE.replace('.IS','')}_veri.csv")
        print(f"Veri CSV olarak kaydedildi: {HISSE.replace('.IS','')}_veri.csv")

if __name__ == "__main__":
    main()