import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Ayarlar ───────────────────────────────────────────
HISSELER          = ["AKBNK.IS", "GARAN.IS", "YKBNK.IS",
                     "EKGYO.IS", "PGSUS.IS", "TCELL.IS",
                     "SISE.IS",  "FROTO.IS"]
SERMAYE           = 100000
RISKSIZ_FAIZ      = 0.45      # Türkiye faiz oranı ~%45
PERIYOT           = "2y"
MAKS_DRAWDOWN     = 0.15      # Max %15 düşüşe izin ver
REBALANS_SURESI   = 30        # Her 30 günde bir rebalans
# ───────────────────────────────────────────────────────

def veri_cek(hisseler, periyot):
    print("Veriler çekiliyor...")
    fiyatlar = {}
    for s in hisseler:
        try:
            df = yf.Ticker(s).history(period=periyot, interval="1d")
            df.index = df.index.tz_localize(None)
            fiyatlar[s] = df['Close']
            print(f"  {s} ✅")
        except Exception as e:
            print(f"  {s} ❌ {e}")
    fiyat_df  = pd.DataFrame(fiyatlar).dropna()
    getiri_df = fiyat_df.pct_change().dropna()
    return fiyat_df, getiri_df

# ── KELLY KRİTERİ ──────────────────────────────────────
def kelly_pozisyon(kazanma_orani, ort_kazanc, ort_kayip):
    """
    Kelly Criterion — optimal pozisyon büyüklüğü
    f = (p*b - q) / b
    p = kazanma olasılığı
    q = kaybetme olasılığı
    b = kazanç/kayıp oranı
    """
    if ort_kayip == 0 or kazanma_orani == 0:
        return 0.0

    p = kazanma_orani
    q = 1 - p
    b = abs(ort_kazanc / ort_kayip)
    f = (p * b - q) / b

    # Yarı Kelly kullan — daha güvenli
    yari_kelly = f / 2
    return max(0, min(yari_kelly, 0.25))  # Max %25

def kelly_hesapla(getiriler):
    """Her hisse için Kelly pozisyon büyüklüğü hesapla"""
    kelly_agirliklar = {}
    for hisse in getiriler.columns:
        g = getiriler[hisse].dropna()
        kazananlar = g[g > 0]
        kaybedenler = g[g < 0]

        if len(kazananlar) == 0 or len(kaybedenler) == 0:
            kelly_agirliklar[hisse] = 0.05
            continue

        kazanma_orani = len(kazananlar) / len(g)
        ort_kazanc    = kazananlar.mean()
        ort_kayip     = abs(kaybedenler.mean())
        kelly         = kelly_pozisyon(kazanma_orani, ort_kazanc, ort_kayip)
        kelly_agirliklar[hisse] = kelly

    # Normalize et — toplamı 1 yap
    toplam = sum(kelly_agirliklar.values())
    if toplam > 0:
        kelly_agirliklar = {k: v/toplam for k, v in kelly_agirliklar.items()}
    else:
        n = len(getiriler.columns)
        kelly_agirliklar = {k: 1/n for k in getiriler.columns}

    return kelly_agirliklar

# ── SHARPE OPTİMİZASYONU ───────────────────────────────
def sharpe_hesapla(agirliklar, getiriler, risksiz=RISKSIZ_FAIZ):
    yillik_getiri = np.sum(getiriler.mean() * agirliklar) * 252
    kovaryans     = getiriler.cov() * 252
    yillik_risk   = np.sqrt(
        np.dot(agirliklar.T, np.dot(kovaryans, agirliklar)))
    if yillik_risk == 0:
        return 0
    return (yillik_getiri - risksiz) / yillik_risk

def portfoy_optimize_et(getiriler):
    n         = len(getiriler.columns)
    sinirlar  = tuple((0.05, 0.40) for _ in range(n))
    kisitlar  = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    baslangic = np.array([1/n] * n)

    sonuc = minimize(
        lambda w: -sharpe_hesapla(w, getiriler),
        baslangic,
        method='SLSQP',
        bounds=sinirlar,
        constraints=kisitlar,
        options={'maxiter': 1000}
    )
    return sonuc.x if sonuc.success else baslangic

# ── TRAİLİNG STOP-LOSS ─────────────────────────────────
def trailing_stop_hesapla(fiyatlar, stop_yuzde=0.08):
    """
    Trailing stop: fiyat yükselince stop da yükselir
    ama fiyat düşünce stop sabit kalır.
    """
    stop_seviyeleri = []
    en_yuksek       = fiyatlar.iloc[0]

    for fiyat in fiyatlar:
        if fiyat > en_yuksek:
            en_yuksek = fiyat
        stop = en_yuksek * (1 - stop_yuzde)
        stop_seviyeleri.append(stop)

    return pd.Series(stop_seviyeleri, index=fiyatlar.index)

# ── DİNAMİK REBALANS ───────────────────────────────────
def dinamik_rebalans_backtest(fiyat_df, getiri_df):
    """
    Her 30 günde bir portföyü yeniden dengele.
    Kelly Criterion ile pozisyon büyüklüğü belirle.
    Trailing stop-loss uygula.
    Drawdown limitini aşarsa tüm pozisyonları kapat.
    """
    sermaye        = SERMAYE
    pozisyonlar    = {}   # {hisse: adet}
    alis_fiyatlari = {}
    en_yuksek_deger = sermaye
    islemler       = []
    portfoy_gecmis = []
    rebalans_sayac = 0

    tarihler = fiyat_df.index

    for i in range(60, len(tarihler)):
        tarih       = tarihler[i]
        guncel_fiyat = fiyat_df.iloc[i]

        # Portföy değeri hesapla
        hisse_degeri = sum(
            pozisyonlar.get(h, 0) * guncel_fiyat[h]
            for h in fiyat_df.columns
        )
        portfoy_degeri = sermaye + hisse_degeri

        # En yüksek değeri güncelle
        if portfoy_degeri > en_yuksek_deger:
            en_yuksek_deger = portfoy_degeri

        # Drawdown kontrolü
        drawdown = (portfoy_degeri - en_yuksek_deger) / en_yuksek_deger
        if drawdown <= -MAKS_DRAWDOWN:
            # Tüm pozisyonları kapat
            for hisse in list(pozisyonlar.keys()):
                if pozisyonlar[hisse] > 0:
                    gelir      = pozisyonlar[hisse] * guncel_fiyat[hisse] * 0.999
                    sermaye   += gelir
                    kar_zarar  = gelir - alis_fiyatlari.get(hisse, gelir)
                    islemler.append({
                        'tarih'    : tarih,
                        'hisse'    : hisse,
                        'islem'    : 'SAT (DRAWDOWN)',
                        'fiyat'    : guncel_fiyat[hisse],
                        'kar_zarar': kar_zarar
                    })
            pozisyonlar    = {}
            alis_fiyatlari = {}
            portfoy_gecmis.append({
                'tarih': tarih,
                'deger': sermaye,
                'drawdown': drawdown
            })
            rebalans_sayac = 0
            continue

        # Trailing stop-loss kontrolü
        for hisse in list(pozisyonlar.keys()):
            if pozisyonlar[hisse] > 0:
                alis   = alis_fiyatlari.get(hisse, guncel_fiyat[hisse])
                kayip  = (guncel_fiyat[hisse] - alis) / alis

                # Trailing stop — %8 düşüşte sat
                if kayip <= -0.08:
                    gelir      = pozisyonlar[hisse] * guncel_fiyat[hisse] * 0.999
                    sermaye   += gelir
                    kar_zarar  = gelir - (pozisyonlar[hisse] * alis * 1.001)
                    islemler.append({
                        'tarih'    : tarih,
                        'hisse'    : hisse,
                        'islem'    : 'SAT (STOP)',
                        'fiyat'    : guncel_fiyat[hisse],
                        'kar_zarar': kar_zarar
                    })
                    del pozisyonlar[hisse]
                    del alis_fiyatlari[hisse]

        # Rebalans zamanı geldi mi?
        rebalans_sayac += 1
        if rebalans_sayac >= REBALANS_SURESI:
            rebalans_sayac = 0

            # Son 60 günlük getiriyle Kelly hesapla
            son_getiriler = getiri_df.iloc[i-60:i]
            kelly         = kelly_hesapla(son_getiriler)

            # Sharpe optimize et
            opt_agirlik   = portfoy_optimize_et(son_getiriler)
            opt_dict      = dict(zip(fiyat_df.columns, opt_agirlik))

            # Kelly ve Sharpe'ı birleştir (%50 Kelly, %50 Sharpe)
            birlesik = {}
            for h in fiyat_df.columns:
                birlesik[h] = 0.5 * kelly.get(h, 1/len(fiyat_df.columns)) + \
                              0.5 * opt_dict.get(h, 1/len(fiyat_df.columns))

            # Normalize
            toplam = sum(birlesik.values())
            birlesik = {k: v/toplam for k, v in birlesik.items()}

            # Mevcut pozisyonları kapat
            for hisse in list(pozisyonlar.keys()):
                if pozisyonlar[hisse] > 0:
                    gelir      = pozisyonlar[hisse] * guncel_fiyat[hisse] * 0.999
                    sermaye   += gelir
                    kar_zarar  = gelir - (pozisyonlar[hisse] *
                                         alis_fiyatlari.get(hisse, guncel_fiyat[hisse]) * 1.001)
                    islemler.append({
                        'tarih'    : tarih,
                        'hisse'    : hisse,
                        'islem'    : 'SAT (REBALANS)',
                        'fiyat'    : guncel_fiyat[hisse],
                        'kar_zarar': kar_zarar
                    })

            pozisyonlar    = {}
            alis_fiyatlari = {}

            # Yeni pozisyonlar aç
            for hisse, agirlik in birlesik.items():
                tutar       = sermaye * agirlik
                fiyat       = guncel_fiyat[hisse]
                adet        = int(tutar / fiyat)
                if adet > 0:
                    maliyet               = adet * fiyat * 1.001
                    sermaye              -= maliyet
                    pozisyonlar[hisse]    = adet
                    alis_fiyatlari[hisse] = fiyat
                    islemler.append({
                        'tarih'    : tarih,
                        'hisse'    : hisse,
                        'islem'    : 'AL (REBALANS)',
                        'fiyat'    : fiyat,
                        'kar_zarar': None
                    })

        # Portföy değerini kaydet
        hisse_degeri = sum(
            pozisyonlar.get(h, 0) * guncel_fiyat[h]
            for h in fiyat_df.columns
        )
        portfoy_gecmis.append({
            'tarih'   : tarih,
            'deger'   : sermaye + hisse_degeri,
            'drawdown': drawdown
        })

    # Son pozisyonları kapat
    son_fiyat = fiyat_df.iloc[-1]
    for hisse, adet in pozisyonlar.items():
        if adet > 0:
            sermaye += adet * son_fiyat[hisse] * 0.999

    return pd.DataFrame(islemler), pd.DataFrame(portfoy_gecmis), sermaye

def sonuc_yazdir(islemler_df, portfoy_df, son_sermaye, fiyat_df):
    toplam_kar   = son_sermaye - SERMAYE
    getiri       = toplam_kar / SERMAYE * 100
    portfoy_df['tepe']  = portfoy_df['deger'].cummax()
    portfoy_df['dusus'] = (portfoy_df['deger'] - portfoy_df['tepe']) / \
                           portfoy_df['tepe'] * 100
    maks_dusus   = portfoy_df['dusus'].min()

    # Yıllık Sharpe
    gunluk_getiri = portfoy_df['deger'].pct_change().dropna()
    yillik_getiri = gunluk_getiri.mean() * 252
    yillik_risk   = gunluk_getiri.std() * np.sqrt(252)
    sharpe        = (yillik_getiri - RISKSIZ_FAIZ) / yillik_risk \
                    if yillik_risk > 0 else 0

    # Al-tut karşılaştırması (eşit ağırlıklı)
    ilk_fiyat   = fiyat_df.iloc[60]
    son_fiyat_v = fiyat_df.iloc[-1]
    al_tut      = ((son_fiyat_v / ilk_fiyat) - 1).mean() * 100

    print("\n" + "="*60)
    print("   SHARPE OPTİMİZASYON SONUÇLARI")
    print("="*60)
    print(f"\n  Başlangıç sermayesi : {SERMAYE:>12,.0f} TL")
    print(f"  Son sermaye         : {son_sermaye:>12,.0f} TL")
    print(f"  Toplam kar/zarar    : {toplam_kar:>+12,.0f} TL")
    print(f"  Toplam getiri       : %{getiri:>+.1f}")
    print(f"\n  Sharpe oranı        : {sharpe:>+.2f}  "
          f"({'POZİTİF ✅' if sharpe > 0 else 'NEGATİF ❌'})")
    print(f"  Maks. düşüş         : %{maks_dusus:.1f}")
    print(f"  Al-tut getirisi     : %{al_tut:>+.1f}")
    print(f"  Fark                : %{getiri-al_tut:>+.1f}")

    if len(islemler_df) > 0:
        satislar = islemler_df[islemler_df['islem'].str.startswith('SAT')]
        satislar = satislar.dropna(subset=['kar_zarar'])
        if len(satislar) > 0:
            kazanan  = len(satislar[satislar['kar_zarar'] > 0])
            basari   = kazanan / len(satislar) * 100
            rebalans = len(islemler_df[islemler_df['islem'] == 'AL (REBALANS)'])
            print(f"\n  Toplam rebalans     : {rebalans}")
            print(f"  Başarı oranı        : %{basari:.1f}")

    # Kelly ağırlıkları
    getiri_df = fiyat_df.pct_change().dropna()
    kelly     = kelly_hesapla(getiri_df)
    print(f"\n  ── Kelly Criterion Ağırlıkları ──────────────")
    for h, a in sorted(kelly.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(a * 40)
        print(f"  {h:<12}: %{a*100:>5.1f}  {bar}")

    print("="*60)
    return sharpe, getiri, al_tut

def grafik_ciz(portfoy_df, fiyat_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("Sharpe Optimizasyonu — Kelly + Trailing Stop + Rebalans",
                 fontsize=13, fontweight='bold')

    # Portföy vs al-tut
    ilk_deger = portfoy_df['deger'].iloc[0]
    ilk_fiyat = fiyat_df.iloc[60]
    al_tut    = SERMAYE * (fiyat_df.iloc[60:] / ilk_fiyat).mean(axis=1)

    ax1.plot(portfoy_df['tarih'], portfoy_df['deger'],
             color='#2563eb', lw=2, label='Optimize Portföy')
    ax1.plot(fiyat_df.index[60:], al_tut,
             color='#f59e0b', lw=1.5, linestyle='--', label='Al-tut (eşit ağırlık)')
    ax1.axhline(y=SERMAYE, color='gray', lw=0.8, linestyle=':')
    ax1.set_ylabel('Portföy Değeri (TL)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(portfoy_df['tarih'], portfoy_df['dusus'],
                     0, color='#dc2626', alpha=0.4, label='Drawdown')
    ax2.axhline(y=-MAKS_DRAWDOWN*100, color='#dc2626',
                linestyle='--', lw=1, label=f'Limit (%{MAKS_DRAWDOWN*100:.0f})')
    ax2.set_ylabel('Drawdown %')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sharpe_optimizasyon.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Grafik kaydedildi: sharpe_optimizasyon.png")

def main():
    print("\n" + "="*60)
    print("   SHARPE OPTİMİZASYONU BAŞLIYOR")
    print(f"   Sermaye: {SERMAYE:,.0f} TL | "
          f"Rebalans: {REBALANS_SURESI} gün | "
          f"Max Drawdown: %{MAKS_DRAWDOWN*100:.0f}")
    print("="*60)

    fiyat_df, getiri_df = veri_cek(HISSELER, PERIYOT)

    if fiyat_df.empty:
        print("Veri alınamadı!")
        return

    print(f"\n{len(fiyat_df.columns)} hisse, {len(fiyat_df)} günlük veri.")
    print("Optimizasyon çalışıyor...")

    islemler_df, portfoy_df, son_sermaye = dinamik_rebalans_backtest(
        fiyat_df, getiri_df)

    sharpe, getiri, al_tut = sonuc_yazdir(
        islemler_df, portfoy_df, son_sermaye, fiyat_df)

    grafik_ciz(portfoy_df, fiyat_df)

    # Sonuçları kaydet
    islemler_df.to_csv("sharpe_islemler.csv",
                        index=False, encoding='utf-8-sig')
    portfoy_df.to_csv("sharpe_portfoy.csv",
                       index=False, encoding='utf-8-sig')
    print("  Sonuçlar kaydedildi.")

if __name__ == "__main__":
    main()