import smtplib
import os
import yfinance as yf
import pandas_ta as ta
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ─── Ayarlar ───────────────────────────────────────────
HISSELER     = ["THYAO.IS", "EREGL.IS", "ASELS.IS"]  # İstediğin hisseleri ekle
GMAIL_ADRES  = "tahir.aytekin72@gmail.com"       # Kendi Gmail adresin
GMAIL_SIFRE = os.getenv("GMAIL_SIFRE")   # Az önce aldığın uygulama şifresi
ALICI_ADRES  = "tahir.aytekin72@gmail.com"       # Bildirimin gideceği adres (aynı olabilir)
MIN_GUC      = 30                       # Sadece bu güçten yüksek sinyaller gönderilsin
# ───────────────────────────────────────────────────────

def veri_cek(sembol):
    df = yf.Ticker(sembol).history(period="6mo", interval="1d")
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
    return sorted([f for f in d if f < son_fiyat], reverse=True)[:3], \
           sorted([f for f in r if f > son_fiyat])[:3]

def sinyal_uret(df, destekler, direncler):
    son    = df.iloc[-1]
    onceki = df.iloc[-2]
    puanlar = {}

    rsi = son['RSI']
    if rsi < 35:
        puanlar['RSI'] = 1
    elif rsi > 65:
        puanlar['RSI'] = -1
    else:
        puanlar['RSI'] = 0

    macd_yukari = (onceki['MACD'] < onceki['MACD_sinyal'] and
                   son['MACD']    > son['MACD_sinyal'])
    macd_asagi  = (onceki['MACD'] > onceki['MACD_sinyal'] and
                   son['MACD']    < son['MACD_sinyal'])
    if macd_yukari:
        puanlar['MACD'] = 2
    elif macd_asagi:
        puanlar['MACD'] = -2
    elif son['MACD'] > son['MACD_sinyal']:
        puanlar['MACD'] = 1
    else:
        puanlar['MACD'] = -1

    fiyat = son['Close']
    if fiyat > son['MA20'] and fiyat > son['MA50']:
        puanlar['MA'] = 1
    elif fiyat < son['MA20'] and fiyat < son['MA50']:
        puanlar['MA'] = -1
    else:
        puanlar['MA'] = 0

    if fiyat <= son['BB_alt']:
        puanlar['BB'] = 1
    elif fiyat >= son['BB_ust']:
        puanlar['BB'] = -1
    else:
        puanlar['BB'] = 0

    if destekler:
        uzaklik = (fiyat - destekler[0]) / fiyat
        puanlar['SD'] = 2 if uzaklik < 0.02 else 0
    if direncler:
        uzaklik = (direncler[0] - fiyat) / fiyat
        if uzaklik < 0.02:
            puanlar['SD'] = puanlar.get('SD', 0) - 2

    toplam = sum(puanlar.values())
    if toplam >= 3:
        karar = "AL"
        guc   = min(int((toplam / 8) * 100), 100)
    elif toplam <= -3:
        karar = "SAT"
        guc   = min(int((abs(toplam) / 8) * 100), 100)
    else:
        karar = "BEKLE"
        guc   = 50

    return karar, guc, toplam, fiyat, rsi, destekler, direncler

def email_icerigi_olustur(sinyaller):
    zaman = datetime.now().strftime("%d.%m.%Y %H:%M")
    
    html = f"""
    <html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
    <h2 style="color: #1e40af; border-bottom: 2px solid #1e40af; padding-bottom: 8px;">
        BIST Sinyal Raporu — {zaman}
    </h2>
    """

    for s in sinyaller:
        sembol, karar, guc, fiyat, rsi, destekler, direncler, toplam = s

        if karar == "AL":
            renk, emoji, bg = "#16a34a", "▲", "#f0fdf4"
        elif karar == "SAT":
            renk, emoji, bg = "#dc2626", "▼", "#fef2f2"
        else:
            renk, emoji, bg = "#d97706", "■", "#fffbeb"

        hedef_str = f"{direncler[0]:.2f} TL" if direncler else "—"
        stop_str  = f"{destekler[0]*0.98:.2f} TL" if destekler else "—"

        html += f"""
        <div style="background:{bg}; border-left: 4px solid {renk};
                    padding: 16px; margin: 16px 0; border-radius: 6px;">
            <h3 style="margin:0 0 8px 0; color:{renk};">
                {emoji} {sembol.replace('.IS','')} — {karar}
            </h3>
            <table style="width:100%; font-size:14px;">
                <tr>
                    <td><b>Fiyat</b></td>
                    <td>{fiyat:.2f} TL</td>
                    <td><b>Sinyal Gücü</b></td>
                    <td>%{guc}</td>
                </tr>
                <tr>
                    <td><b>RSI</b></td>
                    <td>{rsi:.1f}</td>
                    <td><b>Puan</b></td>
                    <td>{toplam:+d} / 8</td>
                </tr>
                <tr>
                    <td><b>Hedef</b></td>
                    <td style="color:#16a34a">{hedef_str}</td>
                    <td><b>Stop-Loss</b></td>
                    <td style="color:#dc2626">{stop_str}</td>
                </tr>
            </table>
        </div>
        """

    html += """
    <p style="color:#6b7280; font-size:12px; margin-top:24px;">
        Bu rapor otomatik oluşturulmuştur. Yatırım tavsiyesi değildir.
        Kararlarınızı kendi araştırmanızla destekleyin.
    </p>
    </body></html>
    """
    return html

def email_gonder(html_icerik, sinyal_sayisi):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"BIST Sinyal Raporu — {sinyal_sayisi} sinyal"
    msg['From']    = GMAIL_ADRES
    msg['To']      = ALICI_ADRES
    msg.attach(MIMEText(html_icerik, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as sunucu:
            sunucu.login(GMAIL_ADRES, GMAIL_SIFRE)
            sunucu.sendmail(GMAIL_ADRES, ALICI_ADRES, msg.as_string())
        print(f"E-posta gonderildi: {ALICI_ADRES}")
    except Exception as e:
        print(f"E-posta hatasi: {e}")

def main():
    print(f"\nTaranan hisseler: {', '.join(HISSELER)}\n")
    sinyaller = []

    for sembol in HISSELER:
        print(f"{sembol} analiz ediliyor...")
        try:
            df = veri_cek(sembol)
            df = gostergeler_ekle(df)
            ham_d, ham_r = destek_direnc_bul(df)
            son_fiyat    = df['Close'].iloc[-1]
            d, r         = onemli_seviyeler(ham_d, ham_r, son_fiyat)
            karar, guc, toplam, fiyat, rsi, destekler, direncler = sinyal_uret(df, d, r)

            print(f"  Karar: {karar}  |  Guc: %{guc}  |  Fiyat: {fiyat:.2f} TL")

            if karar != "BEKLE":
                sinyaller.append((sembol, karar, guc, fiyat, rsi,
                                  destekler, direncler, toplam))
        except Exception as e:
            print(f"  HATA: {sembol} — {e}")

    if sinyaller:
        print(f"\n{len(sinyaller)} sinyal bulundu, e-posta gonderiliyor...")
        html = email_icerigi_olustur(sinyaller)
        email_gonder(html, len(sinyaller))
    else:
        print("\nBu taramada gonderilecek guclu sinyal bulunamadi.")

if __name__ == "__main__":
    main()