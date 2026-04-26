import requests
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
import yfinance as yf
import feedparser
import time
import warnings
warnings.filterwarnings('ignore')

# ─── Ayarlar ───────────────────────────────────────────
HISSE_SEKTÖR = {
    "AKBNK.IS": "bankacılık",
    "GARAN.IS": "bankacılık",
    "YKBNK.IS": "bankacılık",
    "EKGYO.IS": "gayrimenkul",
    "PGSUS.IS": "havacılık",
    "TCELL.IS": "telekom",
    "SISE.IS" : "cam_kimya",
    "FROTO.IS": "otomotiv",
    "THYAO.IS": "havacılık",
    "EREGL.IS": "demir_celik",
    "ASELS.IS": "savunma",
    "TUPRS.IS": "petrol_enerji",
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
}
# ───────────────────────────────────────────────────────

# ── DUYGU ANALİZİ KELİME LİSTELERİ ───────────────────
POZITIF_TR = [
    'artış', 'yükseliş', 'büyüme', 'kâr', 'kar', 'rekor', 'güçlü',
    'olumlu', 'başarı', 'kazanç', 'ihracat', 'anlaşma', 'sözleşme',
    'sipariş', 'yatırım', 'genişleme', 'temettü', 'işbirliği',
    'prim', 'toparlanma', 'iyileşme', 'pozitif', 'yukarı', 'talep',
    'rallisi', 'güçlendi', 'yükseldi', 'açıkladı', 'arttı', 'tamamlandı'
]
NEGATIF_TR = [
    'düşüş', 'kayıp', 'zarar', 'gerileme', 'kriz', 'risk', 'baskı',
    'satış', 'enflasyon', 'borç', 'daralma', 'yavaşlama', 'olumsuz',
    'ceza', 'soruşturma', 'dava', 'zayıf', 'negatif', 'düştü',
    'geriledi', 'azaldı', 'kesinti', 'ertelendi', 'iptal', 'endişe'
]
POZITIF_EN = [
    'growth', 'profit', 'record', 'strong', 'beat', 'exceed', 'upgrade',
    'buy', 'bullish', 'rally', 'surge', 'gain', 'rise', 'increase',
    'contract', 'deal', 'agreement', 'expansion', 'dividend', 'revenue',
    'outperform', 'recovery', 'demand', 'positive', 'boost', 'jump'
]
NEGATIF_EN = [
    'loss', 'decline', 'fall', 'drop', 'weak', 'miss', 'downgrade',
    'sell', 'bearish', 'crash', 'risk', 'concern', 'debt', 'inflation',
    'pressure', 'cut', 'reduce', 'warning', 'investigation', 'fine',
    'lawsuit', 'uncertainty', 'volatile', 'slump', 'tumble', 'plunge'
]

def metin_duygu_skoru(metin):
    if not metin:
        return 0.0
    metin = metin.lower()
    poz = sum(1 for k in POZITIF_TR + POZITIF_EN if k in metin)
    neg = sum(1 for k in NEGATIF_TR + NEGATIF_EN if k in metin)
    top = poz + neg
    return round((poz - neg) / top, 3) if top > 0 else 0.0

# ── HABER KAYNAKLARI ───────────────────────────────────

def google_news_rss(sorgu, dil='tr'):
    """Google News RSS — en güvenilir ücretsiz kaynak"""
    haberler = []
    try:
        if dil == 'tr':
            url = (f"https://news.google.com/rss/search?"
                   f"q={sorgu}&hl=tr&gl=TR&ceid=TR:tr")
        else:
            url = (f"https://news.google.com/rss/search?"
                   f"q={sorgu}&hl=en&gl=US&ceid=US:en")

        feed = feedparser.parse(url)

        for entry in feed.entries[:8]:
            baslik = entry.get('title', '')
            ozet   = entry.get('summary', '')
            metin  = baslik + ' ' + ozet
            skor   = metin_duygu_skoru(metin)

            zaman = datetime.now()
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                import calendar
                zaman = datetime.fromtimestamp(
                    calendar.timegm(entry.published_parsed))

            if baslik:
                haberler.append({
                    'kaynak': f'Google News ({dil.upper()})',
                    'baslik': baslik[:150],
                    'zaman' : zaman,
                    'skor'  : skor,
                    'dil'   : dil,
                    'tur'   : 'hisse'
                })
    except Exception as e:
        pass
    return haberler

def yahoo_news_cek(sembol):
    """Yahoo Finance API ile haber çek"""
    haberler = []
    try:
        ticker = yf.Ticker(sembol)
        news   = ticker.get_news()

        for h in news[:10]:
            # Yeni Yahoo Finance API formatı
            content = h.get('content', {})
            baslik  = (content.get('title', '') or
                      h.get('title', ''))
            ozet    = (content.get('summary', '') or
                      h.get('summary', '') or '')
            zaman_ts = (content.get('pubDate', '') or
                       h.get('providerPublishTime', 0))

            if isinstance(zaman_ts, (int, float)) and zaman_ts > 0:
                zaman = datetime.fromtimestamp(zaman_ts)
            else:
                zaman = datetime.now()

            if baslik:
                metin = baslik + ' ' + ozet
                skor  = metin_duygu_skoru(metin)
                haberler.append({
                    'kaynak': 'Yahoo Finance',
                    'baslik': baslik[:150],
                    'zaman' : zaman,
                    'skor'  : skor,
                    'dil'   : 'en',
                    'tur'   : 'hisse'
                })
    except Exception as e:
        pass
    return haberler

def finans_gundem_cek():
    """Finansgundem.com RSS"""
    haberler = []
    try:
        url  = "https://www.finansgundem.com/rss/haberler.xml"
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            baslik = entry.get('title', '')
            if baslik:
                skor = metin_duygu_skoru(baslik)
                haberler.append({
                    'kaynak': 'Finans Gündem',
                    'baslik': baslik[:150],
                    'zaman' : datetime.now(),
                    'skor'  : skor,
                    'dil'   : 'tr',
                    'tur'   : 'makro'
                })
    except:
        pass
    return haberler

def bloomberght_cek():
    """Bloomberg HT RSS"""
    haberler = []
    try:
        url  = "https://www.bloomberght.com/rss"
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            baslik = entry.get('title', '')
            ozet   = entry.get('summary', '')
            if baslik:
                metin = baslik + ' ' + ozet
                skor  = metin_duygu_skoru(metin)
                haberler.append({
                    'kaynak': 'Bloomberg HT',
                    'baslik': baslik[:150],
                    'zaman' : datetime.now(),
                    'skor'  : skor,
                    'dil'   : 'tr',
                    'tur'   : 'makro'
                })
    except:
        pass
    return haberler

def doviz_net_cek():
    """Doviz.com RSS"""
    haberler = []
    try:
        url  = "https://www.doviz.com/rss/haberler"
        feed = feedparser.parse(url)
        for entry in feed.entries[:8]:
            baslik = entry.get('title', '')
            if baslik:
                skor = metin_duygu_skoru(baslik)
                haberler.append({
                    'kaynak': 'Doviz.com',
                    'baslik': baslik[:150],
                    'zaman' : datetime.now(),
                    'skor'  : skor,
                    'dil'   : 'tr',
                    'tur'   : 'makro'
                })
    except:
        pass
    return haberler

def reuters_rss_cek():
    """Reuters İngilizce haberler"""
    haberler = []
    try:
        url  = "https://feeds.reuters.com/reuters/businessNews"
        feed = feedparser.parse(url)
        for entry in feed.entries[:8]:
            baslik = entry.get('title', '')
            ozet   = entry.get('summary', '')
            if baslik:
                metin = baslik + ' ' + ozet
                skor  = metin_duygu_skoru(metin)
                haberler.append({
                    'kaynak': 'Reuters',
                    'baslik': baslik[:150],
                    'zaman' : datetime.now(),
                    'skor'  : skor,
                    'dil'   : 'en',
                    'tur'   : 'makro'
                })
    except:
        pass
    return haberler

# ── ANA ANALİZ ─────────────────────────────────────────

def hisse_duygu_skoru(sembol):
    sembol_adi = sembol.replace('.IS', '')
    sektor     = HISSE_SEKTÖR.get(sembol, 'genel')
    haberler   = []

    # Yahoo Finance haberleri
    haberler += yahoo_news_cek(sembol)

    # Google News — Türkçe
    haberler += google_news_rss(f"{sembol_adi} hisse borsa", 'tr')
    time.sleep(0.3)

    # Google News — İngilizce
    haberler += google_news_rss(f"{sembol_adi} stock BIST Turkey", 'en')
    time.sleep(0.3)

    if not haberler:
        return 0.0, [], "NÖTR"

    agirliklar = {'hisse': 1.5, 'sektor': 1.0, 'makro': 0.7}
    toplam_skor    = sum(h['skor'] * agirliklar.get(h['tur'], 1.0) for h in haberler)
    toplam_agirlik = sum(agirliklar.get(h['tur'], 1.0) for h in haberler)
    ort = toplam_skor / toplam_agirlik if toplam_agirlik > 0 else 0

    yorum = "POZİTİF" if ort > 0.1 else ("NEGATİF" if ort < -0.1 else "NÖTR")
    return round(ort, 3), haberler, yorum

def makro_duygu_skoru():
    haberler = []
    haberler += bloomberght_cek()
    haberler += finans_gundem_cek()
    haberler += doviz_net_cek()
    haberler += reuters_rss_cek()
    haberler += google_news_rss("Türkiye ekonomi faiz enflasyon", 'tr')
    haberler += google_news_rss("Turkey economy interest rate inflation", 'en')

    if not haberler:
        return 0.0, []

    ort = np.mean([h['skor'] for h in haberler])
    return round(ort, 3), haberler

def tam_analiz_yap(hisseler):
    print("\n" + "="*70)
    print("   HABER & DUYGU ANALİZİ RAPORU")
    print(f"   {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print("="*70)

    print("\n  Makro haberler çekiliyor...")
    makro_skor, makro_haberler = makro_duygu_skoru()
    print(f"  Makro skor: {makro_skor:>+.3f}  "
          f"({'POZİTİF' if makro_skor > 0.1 else 'NEGATİF' if makro_skor < -0.1 else 'NÖTR'})"
          f"  |  {len(makro_haberler)} haber")

    print(f"\n  ── Hisse Duygu Skorları ─────────────────────────")
    print(f"  {'Hisse':<12} {'Skor':>7} {'Yorum':<10} {'Haber':>6} {'Sektör'}")
    print(f"  {'─'*58}")

    sonuclar = {}
    tum_haberler = list(makro_haberler)

    for sembol in hisseler:
        try:
            skor, haberler, yorum = hisse_duygu_skoru(sembol)
            sektor = HISSE_SEKTÖR.get(sembol, '—')
            emoji  = "🟢" if yorum == "POZİTİF" else ("🔴" if yorum == "NEGATİF" else "🟡")
            print(f"  {sembol:<12} {skor:>+7.3f}  {emoji} {yorum:<8} "
                  f"{len(haberler):>5}  {sektor}")
            sonuclar[sembol] = {
                'skor': skor, 'yorum': yorum,
                'haberler': haberler, 'makro': makro_skor
            }
            for h in haberler:
                h['sembol'] = sembol
            tum_haberler += haberler

        except Exception as e:
            print(f"  {sembol:<12} HATA: {e}")
            sonuclar[sembol] = {
                'skor': 0, 'yorum': 'NÖTR',
                'haberler': [], 'makro': makro_skor
            }

    print(f"  {'─'*58}")

    # Öne çıkan haberler
    guclu = [h for h in tum_haberler if abs(h.get('skor', 0)) > 0.1]
    guclu_sirali = sorted(guclu, key=lambda x: abs(x['skor']), reverse=True)

    if guclu_sirali:
        print(f"\n  ── Öne Çıkan Haberler ───────────────────────────")
        for h in guclu_sirali[:8]:
            emoji  = "🟢" if h['skor'] > 0 else "🔴"
            sembol = h.get('sembol', '——').replace('.IS', '')
            baslik = h['baslik'][:60] + "..." if len(h['baslik']) > 60 else h['baslik']
            print(f"  {emoji} [{sembol:<6}] {baslik}")

    print("="*70)

    # CSV kaydet
    kayitlar = [{
        'zaman'      : datetime.now().strftime("%d.%m.%Y %H:%M"),
        'sembol'     : s,
        'duygu_skor' : v['skor'],
        'yorum'      : v['yorum'],
        'haber_sayisi': len(v['haberler']),
        'makro_skor' : v['makro']
    } for s, v in sonuclar.items()]

    pd.DataFrame(kayitlar).to_csv(
        "duygu_analizi.csv", index=False, encoding='utf-8-sig')
    print("\n  Sonuçlar 'duygu_analizi.csv' dosyasına kaydedildi.")

    return sonuclar

def main():
    # feedparser kurulu değilse kur
    try:
        import feedparser
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'feedparser'])

    HISSELER = [
        "AKBNK.IS", "GARAN.IS", "YKBNK.IS", "EKGYO.IS",
        "PGSUS.IS", "TCELL.IS", "SISE.IS",  "FROTO.IS",
        "THYAO.IS", "ASELS.IS"
    ]
    return tam_analiz_yap(HISSELER)

if __name__ == "__main__":
    main()