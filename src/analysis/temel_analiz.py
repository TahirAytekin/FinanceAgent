import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── Ayarlar ───────────────────────────────────────────
HISSELER = [
    "THYAO.IS", "KCHOL.IS", "EREGL.IS", "ASELS.IS",
    "BIMAS.IS", "SAHOL.IS", "SISE.IS",  "AKBNK.IS",
    "GARAN.IS", "YKBNK.IS", "TUPRS.IS", "ARCLK.IS",
    "TOASO.IS", "FROTO.IS", "PGSUS.IS", "TCELL.IS",
    "EKGYO.IS", "PETKM.IS", "VESTL.IS"
]
# ───────────────────────────────────────────────────────

def temel_veri_cek(sembol):
    """
    Yahoo Finance'den temel analiz verilerini çeker.
    F/K, PD/DD, ROE, büyüme, borç oranı gibi metrikler.
    """
    ticker = yf.Ticker(sembol)
    info   = ticker.info

    def al(key, varsayilan=None):
        deger = info.get(key, varsayilan)
        if deger is None or deger == 'N/A':
            return varsayilan
        try:
            return float(deger)
        except:
            return varsayilan

    # Değerleme metrikleri
    fk_orani     = al('trailingPE')
    fk_ileri     = al('forwardPE')
    pd_dd        = al('priceToBook')
    fd_favok     = al('enterpriseToEbitda')
    fiyat_satis  = al('priceToSalesTrailingTwelveMonths')

    # Karlılık
    roe          = al('returnOnEquity')
    roa          = al('returnOnAssets')
    kar_marji    = al('profitMargins')
    favok_marji  = al('ebitdaMargins')
    brut_marji   = al('grossMargins')

    # Büyüme
    gelir_buyume = al('revenueGrowth')
    kazanc_buyume= al('earningsGrowth')

    # Borç & finansal sağlık
    borc_oz      = al('debtToEquity')
    guncel_oran  = al('currentRatio')
    nakit        = al('totalCash')
    toplam_borc  = al('totalDebt')

    # Temettü
    temettu_ver  = al('dividendYield')

    # Piyasa değeri
    piyasa_deger = al('marketCap')
    hisse_fiyat  = al('currentPrice') or al('regularMarketPrice')

    return {
        'sembol'        : sembol,
        'fiyat'         : hisse_fiyat,
        'piyasa_deger_m': piyasa_deger / 1e6 if piyasa_deger else None,

        # Değerleme
        'fk'            : fk_orani,
        'fk_ileri'      : fk_ileri,
        'pd_dd'         : pd_dd,
        'fd_favok'      : fd_favok,
        'fiyat_satis'   : fiyat_satis,

        # Karlılık
        'roe_pct'       : roe * 100 if roe else None,
        'roa_pct'       : roa * 100 if roa else None,
        'kar_marji_pct' : kar_marji * 100 if kar_marji else None,
        'favok_marji_pct': favok_marji * 100 if favok_marji else None,

        # Büyüme
        'gelir_buyume_pct'  : gelir_buyume * 100 if gelir_buyume else None,
        'kazanc_buyume_pct' : kazanc_buyume * 100 if kazanc_buyume else None,

        # Borç
        'borc_oz'       : borc_dd / 100 if (borc_dd := borc_oz) else None,
        'guncel_oran'   : guncel_oran,

        # Temettü
        'temettu_pct'   : temettu_ver * 100 if temettu_ver else None,
    }

def temel_skor_hesapla(veri):
    """
    Her metrik için -2 ile +2 arası puan üretir.
    Toplam skor = temel analizin özeti.
    Pozitif = iyi, negatif = zayıf.
    """
    puan = {}

    # F/K oranı — düşük daha iyi (sektöre göre değişir)
    fk = veri.get('fk')
    if fk:
        if fk < 8:    puan['FK'] = 2
        elif fk < 15: puan['FK'] = 1
        elif fk < 25: puan['FK'] = 0
        elif fk < 40: puan['FK'] = -1
        else:         puan['FK'] = -2
    else:
        puan['FK'] = 0

    # PD/DD — 1'in altı ucuz, çok yüksek pahalı
    pd_dd = veri.get('pd_dd')
    if pd_dd:
        if pd_dd < 1:   puan['PD_DD'] = 2
        elif pd_dd < 2: puan['PD_DD'] = 1
        elif pd_dd < 4: puan['PD_DD'] = 0
        elif pd_dd < 7: puan['PD_DD'] = -1
        else:           puan['PD_DD'] = -2
    else:
        puan['PD_DD'] = 0

    # ROE — yüksek daha iyi
    roe = veri.get('roe_pct')
    if roe:
        if roe > 30:   puan['ROE'] = 2
        elif roe > 15: puan['ROE'] = 1
        elif roe > 5:  puan['ROE'] = 0
        elif roe > 0:  puan['ROE'] = -1
        else:          puan['ROE'] = -2
    else:
        puan['ROE'] = 0

    # Kar marjı
    kar = veri.get('kar_marji_pct')
    if kar:
        if kar > 20:   puan['KAR'] = 2
        elif kar > 10: puan['KAR'] = 1
        elif kar > 5:  puan['KAR'] = 0
        elif kar > 0:  puan['KAR'] = -1
        else:          puan['KAR'] = -2
    else:
        puan['KAR'] = 0

    # Gelir büyümesi
    buy = veri.get('gelir_buyume_pct')
    if buy:
        if buy > 30:   puan['BUYUME'] = 2
        elif buy > 15: puan['BUYUME'] = 1
        elif buy > 5:  puan['BUYUME'] = 0
        elif buy > 0:  puan['BUYUME'] = -1
        else:          puan['BUYUME'] = -2
    else:
        puan['BUYUME'] = 0

    # Borç/Özsermaye — düşük daha iyi
    borc = veri.get('borc_oz')
    if borc is not None:
        if borc < 0.3:   puan['BORC'] = 2
        elif borc < 0.6: puan['BORC'] = 1
        elif borc < 1.0: puan['BORC'] = 0
        elif borc < 2.0: puan['BORC'] = -1
        else:            puan['BORC'] = -2
    else:
        puan['BORC'] = 0

    toplam = sum(puan.values())
    maks   = len(puan) * 2

    if toplam >= 4:
        karar = "GÜÇLÜ"
        renk  = "✅"
    elif toplam >= 1:
        karar = "İYİ"
        renk  = "🟢"
    elif toplam >= -2:
        karar = "NÖTR"
        renk  = "🟡"
    elif toplam >= -4:
        karar = "ZAYIF"
        renk  = "🔴"
    else:
        karar = "ÇOK ZAYIF"
        renk  = "❌"

    return puan, toplam, karar, renk

def rapor_yazdir(veriler):
    print("\n" + "="*90)
    print("   TEMEL ANALİZ RAPORU — BIST HİSSELERİ")
    print("="*90)
    print(f"  {'Hisse':<12} {'F/K':>6} {'PD/DD':>6} {'ROE%':>6} {'Kar%':>6} "
          f"{'Büy%':>6} {'Borç/Öz':>8} {'Skor':>5} {'Karar':<10}")
    print(f"  {'─'*84}")

    guclu   = []
    iyi     = []
    notr    = []
    zayif   = []

    for v in sorted(veriler, key=lambda x: x['skor'], reverse=True):
        fk   = f"{v['fk']:.1f}"   if v['fk']   else "—"
        pddd = f"{v['pd_dd']:.1f}" if v['pd_dd'] else "—"
        roe  = f"{v['roe_pct']:.0f}" if v['roe_pct'] else "—"
        kar  = f"{v['kar_marji_pct']:.0f}" if v['kar_marji_pct'] else "—"
        buy  = f"{v['gelir_buyume_pct']:.0f}" if v['gelir_buyume_pct'] else "—"
        borc = f"{v['borc_oz']:.2f}" if v['borc_oz'] is not None else "—"

        print(f"  {v['sembol']:<12} {fk:>6} {pddd:>6} {roe:>6} {kar:>6} "
              f"{buy:>6} {borc:>8} {v['skor']:>+5}  {v['renk']} {v['karar']}")

        if v['karar'] == 'GÜÇLÜ':   guclu.append(v['sembol'])
        elif v['karar'] == 'İYİ':   iyi.append(v['sembol'])
        elif v['karar'] == 'NÖTR':  notr.append(v['sembol'])
        else:                        zayif.append(v['sembol'])

    print(f"  {'─'*84}")
    print(f"\n  ✅ Güçlü  : {', '.join([s.replace('.IS','') for s in guclu])  or '—'}")
    print(f"  🟢 İyi    : {', '.join([s.replace('.IS','') for s in iyi])    or '—'}")
    print(f"  🟡 Nötr   : {', '.join([s.replace('.IS','') for s in notr])   or '—'}")
    print(f"  🔴 Zayıf  : {', '.join([s.replace('.IS','') for s in zayif])  or '—'}")
    print("="*90)

    # CSV kaydet
    df = pd.DataFrame(veriler)
    df.to_csv("temel_analiz.csv", index=False, encoding='utf-8-sig')
    print("\n  Sonuçlar 'temel_analiz.csv' dosyasına kaydedildi.")

    return guclu + iyi  # ML modeline gönderilecek hisseler

def main():
    print("\nTemel analiz verileri çekiliyor...")
    veriler = []

    for i, sembol in enumerate(HISSELER, 1):
        print(f"  [{i:02d}/{len(HISSELER)}] {sembol}...", end=' ')
        try:
            veri              = temel_veri_cek(sembol)
            puan, toplam, karar, renk = temel_skor_hesapla(veri)
            veri['puan']      = puan
            veri['skor']      = toplam
            veri['karar']     = karar
            veri['renk']      = renk
            veriler.append(veri)
            print(f"{renk} {karar} (Skor: {toplam:+d})")
        except Exception as e:
            print(f"HATA: {e}")

    if veriler:
        secilen = rapor_yazdir(veriler)
        print(f"\n  ML modeline gönderilecek hisseler ({len(secilen)}):")
        print(f"  {', '.join([s.replace('.IS','') for s in secilen])}")
        return secilen

if __name__ == "__main__":
    main()