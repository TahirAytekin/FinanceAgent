import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from datetime import datetime
warnings.filterwarnings('ignore')

# ─── Ayarlar ───────────────────────────────────────────
HISSELER = ["THYAO.IS", "GARAN.IS", "YKBNK.IS",
            "AKBNK.IS", "EKGYO.IS", "PGSUS.IS"]
PERIYOT  = "5y"
PENCERE  = 20     # 20 günlük pencere (30'dan daha iyi)
EPOCH    = 100
BATCH    = 64
LR       = 0.001
# ───────────────────────────────────────────────────────

# ── GELİŞTİRİLMİŞ LSTM MİMARİSİ ───────────────────────
class LSTMv2(nn.Module):
    def __init__(self, giris, gizli=64, cikis=3):
        super().__init__()

        # Daha basit — 1 katman, daha az nöron
        self.lstm = nn.LSTM(
            input_size=giris,
            hidden_size=gizli,
            num_layers=1,        # 2'den 1'e düşürdük
            batch_first=True,
            dropout=0.0          # Tek katmanda dropout olmuyor
        )

        # Batch normalization — overfitting'i azaltır
        self.batch_norm = nn.BatchNorm1d(gizli)
        self.dropout1   = nn.Dropout(0.4)
        self.fc1        = nn.Linear(gizli, 32)
        self.dropout2   = nn.Dropout(0.3)
        self.fc2        = nn.Linear(32, cikis)
        self.relu       = nn.ReLU()

    def forward(self, x):
        lstm_out, (hn, _) = self.lstm(x)
        # Son adımın çıktısını al
        son_cikis = hn[-1]
        son_cikis = self.batch_norm(son_cikis)
        son_cikis = self.dropout1(son_cikis)
        out       = self.relu(self.fc1(son_cikis))
        out       = self.dropout2(out)
        return self.fc2(out)

# ── VERİ HAZIRLAMA ─────────────────────────────────────
def veri_cek(sembol):
    df = yf.Ticker(sembol).history(period=PERIYOT, interval="1d")
    df = df[['Open','High','Low','Close','Volume']]
    df.index = df.index.tz_localize(None)
    return df

def ozellikler_ekle(df):
    df = df.copy()

    # Temel göstergeler
    df['RSI']         = ta.rsi(df['Close'], length=14)
    df['RSI_fast']    = ta.rsi(df['Close'], length=7)
    macd              = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD_hist']   = macd['MACDh_12_26_9']
    df['MACD_norm']   = (macd['MACD_12_26_9'] - macd['MACDs_12_26_9'])

    # Hareketli ortalamalar
    df['MA5']         = ta.sma(df['Close'], length=5)
    df['MA20']        = ta.sma(df['Close'], length=20)
    df['MA50']        = ta.sma(df['Close'], length=50)
    df['MA200']       = ta.sma(df['Close'], length=200)

    # Bollinger
    bb                = ta.bbands(df['Close'], length=20, std=2)
    bb_s              = bb.columns.tolist()
    df['BB_ust']      = bb[bb_s[2]]
    df['BB_alt']      = bb[bb_s[0]]

    # Volatilite & hacim
    df['ATR']         = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Volatilite']  = df['Close'].pct_change().rolling(10).std()
    df['Hacim_MA']    = ta.sma(df['Volume'], length=20)
    df['Hacim_Oran']  = df['Volume'] / df['Hacim_MA']

    # Normalize özellikler
    df['RSI_N']       = (df['RSI'] - 50) / 50
    df['RSI_f_N']     = (df['RSI_fast'] - 50) / 50
    df['BB_Konum']    = (df['Close'] - df['BB_alt']) / (df['BB_ust'] - df['BB_alt'])
    df['MA5_N']       = (df['Close'] - df['MA5'])   / df['MA5']
    df['MA20_N']      = (df['Close'] - df['MA20'])  / df['MA20']
    df['MA50_N']      = (df['Close'] - df['MA50'])  / df['MA50']
    df['MA200_N']     = (df['Close'] - df['MA200']) / df['MA200']
    df['Trend']       = (df['MA5']   - df['MA50'])  / df['MA50']

    # Getiriler
    for g in [1, 3, 5, 10]:
        df[f'G{g}']   = df['Close'].pct_change(g)

    # Mum özellikleri
    df['Kanat']       = (df['High'] - df['Low']) / df['Close']
    df['Yon']         = np.where(df['Close'] > df['Open'], 1.0, -1.0)
    df['52H']         = df['Close'] / df['Close'].rolling(252).max()

    return df.dropna()

OZELLIKLER = [
    'RSI_N', 'RSI_f_N', 'MACD_norm', 'MACD_hist',
    'BB_Konum', 'MA5_N', 'MA20_N', 'MA50_N', 'MA200_N', 'Trend',
    'G1', 'G3', 'G5', 'G10',
    'Hacim_Oran', 'ATR', 'Volatilite',
    'Kanat', 'Yon', '52H'
]

def hedef_olustur(df, ileriye=5, esik=0.02):
    df = df.copy()
    df['Gelecek'] = df['Close'].shift(-ileriye) / df['Close'] - 1
    df['Hedef']   = df['Gelecek'].apply(
        lambda g: 2 if g >= esik else (0 if g <= -esik else 1))
    return df.dropna()

def pencere_olustur(X, y, pencere):
    Xp, yp = [], []
    for i in range(pencere, len(X)):
        Xp.append(X[i-pencere:i])
        yp.append(y[i])
    return np.array(Xp), np.array(yp)

# ── MODEL EĞİTİMİ ──────────────────────────────────────
def model_egit(sembol):
    print(f"\n  {sembol} hazırlanıyor...")
    df     = veri_cek(sembol)
    df     = ozellikler_ekle(df)
    df     = hedef_olustur(df)

    X      = df[OZELLIKLER].values
    y      = df['Hedef'].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    Xp, yp = pencere_olustur(X_sc, y, PENCERE)

    # Zaman bazlı bölünme
    bolme    = int(len(Xp) * 0.8)
    X_e, X_t = Xp[:bolme], Xp[bolme:]
    y_e, y_t = yp[:bolme], yp[bolme:]

    Xe_t = torch.FloatTensor(X_e)
    ye_t = torch.LongTensor(y_e)
    Xt_t = torch.FloatTensor(X_t)
    yt_t = torch.LongTensor(y_t)

    dataset    = torch.utils.data.TensorDataset(Xe_t, ye_t)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=False)

    model     = LSTMv2(giris=len(OZELLIKLER))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    # Sınıf ağırlıkları
    sayilar   = np.bincount(y_e, minlength=3)
    agirliklar = torch.FloatTensor(
        [1.0/(s+1) for s in sayilar])
    criterion = nn.CrossEntropyLoss(weight=agirliklar)

    # Early stopping
    en_iyi_kayip  = float('inf')
    en_iyi_durum  = None
    sabir_sayaci  = 0
    SABIR         = 20

    print(f"  Eğitim başlıyor ({len(Xp)} örnek, {EPOCH} epoch max)...")

    for epoch in range(EPOCH):
        model.train()
        toplam = 0
        for Xb, yb in dataloader:
            optimizer.zero_grad()
            cikis = model(Xb)
            kayip = criterion(cikis, yb)
            kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            toplam += kayip.item()

        ort = toplam / len(dataloader)

        # Validasyon
        model.eval()
        with torch.no_grad():
            val_cikis = model(Xt_t)
            val_kayip = criterion(val_cikis, yt_t).item()
            tahmin    = torch.argmax(val_cikis, dim=1)
            dogruluk  = (tahmin == yt_t).float().mean().item()

        scheduler.step(val_kayip)

        # Early stopping
        if val_kayip < en_iyi_kayip:
            en_iyi_kayip = val_kayip
            en_iyi_durum = {k: v.clone() for k, v in model.state_dict().items()}
            sabir_sayaci  = 0
        else:
            sabir_sayaci += 1
            if sabir_sayaci >= SABIR:
                print(f"  Early stopping: epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:>3} | "
                  f"Eğitim: {ort:.4f} | "
                  f"Val: {val_kayip:.4f} | "
                  f"Doğruluk: %{dogruluk*100:.1f}")

    # En iyi modeli yükle
    model.load_state_dict(en_iyi_durum)

    # Final değerlendirme
    model.eval()
    with torch.no_grad():
        final_cikis = model(Xt_t)
        tahminler   = torch.argmax(final_cikis, dim=1).numpy()
        olasiliklar = torch.softmax(final_cikis, dim=1).numpy()

    print(f"\n  ── {sembol} Final Sonuçları ──────────────────")
    rapor = classification_report(
        y_t, tahminler,
        target_names=['SAT','BEKLE','AL'],
        zero_division=0, output_dict=True)
    print(classification_report(
        y_t, tahminler,
        target_names=['SAT','BEKLE','AL'],
        zero_division=0))

    # Güncel sinyal
    son_pencere = df[OZELLIKLER].values[-PENCERE:]
    son_sc      = scaler.transform(son_pencere)
    son_X       = torch.FloatTensor(son_sc).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        son_cikis   = model(son_X)
        son_olas    = torch.softmax(son_cikis, dim=1).numpy()[0]
        son_tahmin  = np.argmax(son_olas)

    karar = {2:"AL", 0:"SAT", 1:"BEKLE"}[son_tahmin]
    guven = son_olas[son_tahmin]
    if guven < 0.42:
        karar = "BEKLE"

    emoji = "🟢" if karar=="AL" else ("🔴" if karar=="SAT" else "⚪")
    print(f"  Güncel sinyal: {emoji} {karar} "
          f"(Güven: %{guven*100:.1f} | "
          f"AL: %{son_olas[2]*100:.1f} | "
          f"SAT: %{son_olas[0]*100:.1f})")

    # Modeli kaydet
    torch.save({
        'model_state': model.state_dict(),
        'scaler'     : scaler,
        'ozellikler' : OZELLIKLER,
        'pencere'    : PENCERE,
        'hisse'      : sembol,
        'tarih'      : datetime.now().strftime("%d.%m.%Y"),
    }, f"{sembol.replace('.IS','')}_lstm_v2.pt")

    return {
        'sembol'  : sembol,
        'dogruluk': rapor['accuracy'],
        'al_f1'   : rapor['AL']['f1-score'],
        'sat_f1'  : rapor['SAT']['f1-score'],
        'karar'   : karar,
        'guven'   : guven,
    }

def main():
    print("\n" + "="*60)
    print("   LSTM v2 — GELİŞTİRİLMİŞ MODEL")
    print(f"   Pencere: {PENCERE} gün | Early stopping | Batch norm")
    print("="*60)

    sonuclar = []
    for sembol in HISSELER:
        try:
            s = model_egit(sembol)
            sonuclar.append(s)
        except Exception as e:
            print(f"  {sembol} HATA: {e}")

    # Genel özet
    print("\n" + "="*60)
    print("   GENEL ÖZET")
    print("="*60)
    print(f"  {'Hisse':<12} {'Doğruluk':>9} {'AL f1':>7} "
          f"{'SAT f1':>7} {'Sinyal':<8} {'Güven':>6}")
    print(f"  {'─'*52}")
    for s in sorted(sonuclar, key=lambda x: x['dogruluk'], reverse=True):
        emoji = "🟢" if s['karar']=="AL" else ("🔴" if s['karar']=="SAT" else "⚪")
        print(f"  {s['sembol']:<12} "
              f"%{s['dogruluk']*100:>7.1f}  "
              f"{s['al_f1']:>7.2f}  "
              f"{s['sat_f1']:>7.2f}  "
              f"{emoji} {s['karar']:<6} "
              f"%{s['guven']*100:>5.1f}")
    print("="*60)

    ort_dogru = np.mean([s['dogruluk'] for s in sonuclar])
    print(f"\n  Ortalama doğruluk: %{ort_dogru*100:.1f}")
    if ort_dogru > 0.42:
        print("  RF/XGBoost'tan daha iyi — LSTM entegre edilebilir!")
    else:
        print("  RF/XGBoost hâlâ daha iyi — LSTM yardımcı katman olarak kullanılabilir.")

if __name__ == "__main__":
    main()