import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Ayarlar ───────────────────────────────────────────
SERMAYE       = 100000    # TL
RISK_SEVIYESI = "orta"    # "dusuk", "orta", "yuksek"
PERIYOT       = "2y"
# ───────────────────────────────────────────────────────

def getiri_verisi_cek(hisseler, periyot):
    print("Fiyat verileri çekiliyor...")
    fiyatlar = {}
    for s in hisseler:
        try:
            df = yf.Ticker(s).history(period=periyot, interval="1d")
            df.index = df.index.tz_localize(None)
            fiyatlar[s] = df['Close']
            print(f"  {s} ✅")
        except Exception as e:
            print(f"  {s} ❌ {e}")

    fiyat_df = pd.DataFrame(fiyatlar).dropna()
    getiri_df = fiyat_df.pct_change().dropna()
    return fiyat_df, getiri_df

def portfoy_metrikleri(agirliklar, getiriler, risk_free=0.45):
    """
    risk_free: Türkiye faiz oranı ~%45 (yıllık)
    Günlük getiri ve kovaryans kullanılıyor
    """
    yillik_getiri = np.sum(getiriler.mean() * agirliklar) * 252
    kovaryans     = getiriler.cov() * 252
    yillik_risk   = np.sqrt(np.dot(agirliklar.T, np.dot(kovaryans, agirliklar)))
    sharpe        = (yillik_getiri - risk_free) / yillik_risk if yillik_risk > 0 else 0
    return yillik_getiri, yillik_risk, sharpe

def portfoy_optimize_et(getiriler, hedef):
    """
    3 farklı optimizasyon stratejisi:
    - maksimum_sharpe: en iyi risk/getiri oranı
    - minimum_risk: en düşük volatilite
    - maksimum_getiri: en yüksek beklenen getiri
    """
    n = len(getiriler.columns)

    sinirlar  = tuple((0.05, 0.40) for _ in range(n))  # Her hisse min %5 max %40
    kisitlar  = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Toplam %100
    baslangic = np.array([1/n] * n)

    if hedef == "maksimum_sharpe":
        def hedef_fn(w):
            _, _, sharpe = portfoy_metrikleri(w, getiriler)
            return -sharpe  # Minimize ettiğimiz için negatif

    elif hedef == "minimum_risk":
        def hedef_fn(w):
            _, risk, _ = portfoy_metrikleri(w, getiriler)
            return risk

    else:  # maksimum_getiri
        def hedef_fn(w):
            getiri, _, _ = portfoy_metrikleri(w, getiriler)
            return -getiri

    sonuc = minimize(hedef_fn, baslangic,
                     method='SLSQP',
                     bounds=sinirlar,
                     constraints=kisitlar,
                     options={'maxiter': 1000})

    return sonuc.x if sonuc.success else baslangic

def etkin_sinir_ciz(getiriler, n_portfoy=500):
    """Monte Carlo ile rastgele portföyler üret — etkin sınır görselleştirmesi"""
    print("Etkin sınır hesaplanıyor...")
    n          = len(getiriler.columns)
    getiri_lst = []
    risk_lst   = []
    sharpe_lst = []
    agirlik_lst = []

    for _ in range(n_portfoy):
        w = np.random.dirichlet(np.ones(n))
        g, r, s = portfoy_metrikleri(w, getiriler)
        getiri_lst.append(g)
        risk_lst.append(r)
        sharpe_lst.append(s)
        agirlik_lst.append(w)

    return np.array(getiri_lst), np.array(risk_lst), np.array(sharpe_lst)

def grafik_ciz(getiriler, opt_agirliklar, hisseler):
    getiri_lst, risk_lst, sharpe_lst = etkin_sinir_ciz(getiriler)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Portföy Optimizasyonu — Markowitz", fontsize=14, fontweight='bold')

    # Etkin sınır grafiği
    sc = ax1.scatter(risk_lst * 100, getiri_lst * 100,
                     c=sharpe_lst, cmap='RdYlGn', alpha=0.5, s=10)
    plt.colorbar(sc, ax=ax1, label='Sharpe Oranı')

    # Optimize portföyü işaretle
    for isim, agirlik in opt_agirliklar.items():
        g, r, s = portfoy_metrikleri(agirlik, getiriler)
        semboller = {'maksimum_sharpe': '*', 'minimum_risk': 'o', 'maksimum_getiri': 'v'}
        renkler   = {'maksimum_sharpe': '#2563eb', 'minimum_risk': '#16a34a', 'maksimum_getiri': '#dc2626'}
        ax1.scatter(r * 100, g * 100, marker=semboller[isim],
                    color=renkler[isim], s=300, zorder=5,
                    label=isim.replace('_', ' ').title())

    ax1.set_xlabel('Yıllık Risk (Volatilite %)')
    ax1.set_ylabel('Yıllık Beklenen Getiri %')
    ax1.set_title('Etkin Sınır & Optimize Portföyler')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ağırlık dağılımı (önerilen portföy)
    oneri = list(opt_agirliklar.values())[0]  # Sharpe optimum
    isimler = [s.replace('.IS', '') for s in hisseler]
    renkler  = plt.cm.Set3(np.linspace(0, 1, len(isimler)))
    ax2.pie(oneri, labels=isimler, autopct='%1.1f%%',
            colors=renkler, startangle=90)
    ax2.set_title('Önerilen Portföy Dağılımı (Maks. Sharpe)')

    plt.tight_layout()
    plt.savefig("portfoy_optimizasyon.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Grafik kaydedildi: portfoy_optimizasyon.png")

def rapor_yazdir(getiriler, opt_agirliklar, fiyat_df, hisseler):
    print("\n" + "="*65)
    print("   PORTFÖY OPTİMİZASYON RAPORU")
    print("="*65)

    for strateji, agirliklar in opt_agirliklar.items():
        g, r, s = portfoy_metrikleri(agirliklar, getiriler)
        print(f"\n  ── {strateji.replace('_',' ').upper()} ───────────────────────")
        print(f"  Beklenen Yıllık Getiri : %{g*100:>+.1f}")
        print(f"  Yıllık Risk (Volatilite): %{r*100:>.1f}")
        print(f"  Sharpe Oranı           : {s:>.2f}")
        print(f"\n  {'Hisse':<12} {'Ağırlık':>8} {'Tutar (TL)':>12} {'Lot*':>8}")
        print(f"  {'─'*44}")

        son_fiyatlar = fiyat_df.iloc[-1]
        for hisse, agirlik in zip(hisseler, agirliklar):
            if agirlik > 0.01:
                tutar     = SERMAYE * agirlik
                fiyat     = son_fiyatlar[hisse]
                lot       = int(tutar / fiyat)
                print(f"  {hisse:<12} %{agirlik*100:>6.1f}   "
                      f"{tutar:>10,.0f}   {lot:>7}")

        print(f"  {'─'*44}")
        print(f"  * Lot: {SERMAYE:,.0f} TL sermaye için tahmini hisse adedi")

    # Korelasyon özeti
    print(f"\n  ── KORELASYON MATRİSİ (Düşük = Daha İyi Çeşitlendirme) ──")
    korelasyon = getiriler.corr()
    isimler    = [s.replace('.IS','') for s in hisseler]
    print(f"  {'':>8}", end="")
    for i in isimler:
        print(f"  {i:>6}", end="")
    print()
    for i, row in enumerate(isimler):
        print(f"  {row:>8}", end="")
        for j in range(len(isimler)):
            val = korelasyon.iloc[i, j]
            print(f"  {val:>6.2f}", end="")
        print()

    print("="*65)

    # CSV kaydet
    sonuc_listesi = []
    for strateji, agirliklar in opt_agirliklar.items():
        for hisse, agirlik in zip(hisseler, agirliklar):
            sonuc_listesi.append({
                'strateji': strateji,
                'hisse'   : hisse,
                'agirlik' : agirlik,
                'tutar_tl': SERMAYE * agirlik
            })
    pd.DataFrame(sonuc_listesi).to_csv(
        "portfoy_optimizasyon.csv", index=False, encoding='utf-8-sig')
    print("\n  Sonuçlar 'portfoy_optimizasyon.csv' dosyasına kaydedildi.")

def main():
    # Ana sistemden gelen güçlü hisseler
    HISSELER = [
        "AKBNK.IS", "GARAN.IS", "YKBNK.IS",
        "EKGYO.IS", "PGSUS.IS", "TCELL.IS",
        "SISE.IS",  "FROTO.IS"
    ]

    print("\n" + "="*65)
    print("   PORTFÖY OPTİMİZASYONU — MARKOWITZ MODELİ")
    print(f"   Sermaye: {SERMAYE:,.0f} TL | Risk: {RISK_SEVIYESI.upper()}")
    print("="*65)

    fiyat_df, getiriler = getiri_verisi_cek(HISSELER, PERIYOT)

    if getiriler.empty:
        print("Veri alınamadı!")
        return

    # Boş sütunları temizle
    getiriler = getiriler.dropna(axis=1)
    hisseler  = list(getiriler.columns)
    fiyat_df  = fiyat_df[hisseler]

    print(f"\n{len(hisseler)} hisse, {len(getiriler)} günlük veri kullanılıyor.")

    # 3 farklı strateji optimize et
    print("Portföyler optimize ediliyor...")
    stratejiler = ["maksimum_sharpe", "minimum_risk", "maksimum_getiri"]
    opt_agirliklar = {}

    for strateji in stratejiler:
        agirlik = portfoy_optimize_et(getiriler, strateji)
        opt_agirliklar[strateji] = agirlik
        g, r, s = portfoy_metrikleri(agirlik, getiriler)
        print(f"  {strateji:<20} Getiri: %{g*100:>+.1f} | "
              f"Risk: %{r*100:.1f} | Sharpe: {s:.2f}")

    rapor_yazdir(getiriler, opt_agirliklar, fiyat_df, hisseler)
    grafik_ciz(getiriler, opt_agirliklar, hisseler)

if __name__ == "__main__":
    main()