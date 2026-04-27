import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')

# ─── Ayarlar ───────────────────────────────────────────
HISSE    = "GARAN.IS"
PERIYOT  = "5y"
PENCERE  = 30    # 30 günlük geçmişe bakarak tahmin yap
EPOCH    = 50
BATCH    = 32
# ───────────────────────────────────────────────────────

# ── LSTM MODELİ ────────────────────────────────────────
class BISTLSTMModel(nn.Module):
    def __init__(self, giris_boyutu, gizli_boyut=128, katman_sayisi=2, cikis=3):
        super(BISTLSTMModel, self).__init__()
        self.gizli_boyut   = gizli_boyut
        self.katman_sayisi = katman_sayisi

        self.lstm = nn.LSTM(
            input_size=giris_boyutu,
            hidden_size=gizli_boyut,
            num_layers=katman_sayisi,
            batch_first=True,
            dropout=0.3
        )
        self.attention = nn.Linear(gizli_boyut, 1)
        self.fc1       = nn.Linear(gizli_boyut, 64)
        self.fc2       = nn.Linear(64, cikis)
        self.relu      = nn.ReLU()
        self.dropout   = nn.Dropout(0.3)
        self.softmax   = nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM katmanı
        lstm_cikis, (hn, cn) = self.lstm(x)

        # Attention mekanizması — hangi güne daha çok dikkat?
        attention_agirlik = torch.softmax(
            self.attention(lstm_cikis), dim=1)
        context = (attention_agirlik * lstm_cikis).sum(dim=1)

        # Tam bağlantılı katmanlar
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ── VERİ HAZIRLAMA ─────────────────────────────────────
def veri_cek(sembol, periyot):
    df = yf.Ticker(sembol).history(period=periyot, interval="1d")
    df = df[['Open','High','Low','Close','Volume']]
    df.index = df.index.tz_localize(None)
    return df

def ozellikler_ekle(df):
    df = df.copy()
    df['RSI']         = ta.rsi(df['Close'], length=14)
    df['RSI_fast']    = ta.rsi(df['Close'], length=7)
    macd              = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_hist']   = macd['MACDh_12_26_9']
    df['MA5']         = ta.sma(df['Close'], length=5)
    df['MA20']        = ta.sma(df['Close'], length=20)
    df['MA50']        = ta.sma(df['Close'], length=50)
    df['MA200']       = ta.sma(df['Close'], length=200)
    bb                = ta.bbands(df['Close'], length=20, std=2)
    bb_s              = bb.columns.tolist()
    df['BB_ust']      = bb[bb_s[2]]
    df['BB_alt']      = bb[bb_s[0]]
    df['BB_genislik'] = (df['BB_ust'] - bb[bb_s[1]]) / bb[bb_s[1]]
    df['ATR']         = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Volatilite']  = df['Close'].pct_change().rolling(20).std()
    df['Hacim_MA20']  = ta.sma(df['Volume'], length=20)
    df['Hacim_Oran']  = df['Volume'] / df['Hacim_MA20']

    # Normalize edilmiş özellikler
    df['RSI_Norm']    = (df['RSI'] - 50) / 50
    df['RSI_f_Norm']  = (df['RSI_fast'] - 50) / 50
    df['BB_Konum']    = (df['Close'] - df['BB_alt']) / (df['BB_ust'] - df['BB_alt'])
    df['MA5_Fark']    = (df['Close'] - df['MA5'])   / df['MA5']
    df['MA20_Fark']   = (df['Close'] - df['MA20'])  / df['MA20']
    df['MA50_Fark']   = (df['Close'] - df['MA50'])  / df['MA50']
    df['MA200_Fark']  = (df['Close'] - df['MA200']) / df['MA200']
    df['MACD_Norm']   = df['MACD'] - ta.sma(df['MACD'], length=9)
    df['Trend_Guc']   = (df['MA5'] - df['MA50'])   / df['MA50']

    for g in [1, 3, 5, 10, 20]:
        df[f'Getiri_{g}g'] = df['Close'].pct_change(g)

    df['Kanat']       = (df['High'] - df['Low']) / df['Close']
    df['Govde']       = abs(df['Close'] - df['Open']) / df['Close']
    df['Yon']         = np.where(df['Close'] > df['Open'], 1.0, -1.0)
    df['52H_Yuzde']   = df['Close'] / df['Close'].rolling(252).max()
    df['RSI_Trend']   = df['RSI'] - df['RSI'].shift(5)
    df['Hacim_Fiyat'] = df['Getiri_1g'] * df['Hacim_Oran']

    return df.dropna()

def hedef_olustur(df, ileriye=3, esik=0.015):
    df = df.copy()
    df['Gelecek'] = df['Close'].shift(-ileriye) / df['Close'] - 1
    df['Hedef']   = df['Gelecek'].apply(
        lambda g: 2 if g >= esik else (0 if g <= -esik else 1))
    return df.dropna()

OZELLIKLER = [
    'RSI_Norm', 'RSI_f_Norm', 'MACD_Norm', 'MACD_hist',
    'BB_Konum', 'BB_genislik', 'MA5_Fark', 'MA20_Fark',
    'MA50_Fark', 'MA200_Fark', 'Trend_Guc',
    'Getiri_1g', 'Getiri_3g', 'Getiri_5g', 'Getiri_10g', 'Getiri_20g',
    'Hacim_Oran', 'ATR', 'Volatilite',
    'Kanat', 'Govde', 'Yon', '52H_Yuzde', 'RSI_Trend', 'Hacim_Fiyat'
]

def pencere_olustur(X, y, pencere):
    """30 günlük pencereler oluştur"""
    X_pencere, y_pencere = [], []
    for i in range(pencere, len(X)):
        X_pencere.append(X[i-pencere:i])
        y_pencere.append(y[i])
    return np.array(X_pencere), np.array(y_pencere)

# ── MODEL EĞİTİMİ ──────────────────────────────────────
def model_egit(df):
    X = df[OZELLIKLER].values
    y = df['Hedef'].values

    # Ölçeklendir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pencere oluştur
    X_p, y_p = pencere_olustur(X_scaled, y, PENCERE)

    # Eğitim/test bölünmesi
    bolme    = int(len(X_p) * 0.8)
    X_e, X_t = X_p[:bolme], X_p[bolme:]
    y_e, y_t = y_p[:bolme], y_p[bolme:]

    # PyTorch tensor
    X_e_t = torch.FloatTensor(X_e)
    y_e_t = torch.LongTensor(y_e)
    X_t_t = torch.FloatTensor(X_t)
    y_t_t = torch.LongTensor(y_t)

    # DataLoader
    dataset    = torch.utils.data.TensorDataset(X_e_t, y_e_t)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=False)

    # Model
    model     = BISTLSTMModel(giris_boyutu=len(OZELLIKLER))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Sınıf ağırlıkları — dengesiz veri için
    sinif_sayilari = np.bincount(y_e)
    agirliklar     = torch.FloatTensor(
        [1.0 / (s + 1) for s in sinif_sayilari])
    criterion = nn.CrossEntropyLoss(weight=agirliklar)

    # Eğitim döngüsü
    print(f"\n  Model eğitiliyor ({EPOCH} epoch)...")
    en_iyi_kayip = float('inf')
    en_iyi_model = None

    for epoch in range(EPOCH):
        model.train()
        toplam_kayip = 0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            cikis = model(X_batch)
            kayip = criterion(cikis, y_batch)
            kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            toplam_kayip += kayip.item()

        ort_kayip = toplam_kayip / len(dataloader)

        # En iyi modeli kaydet
        if ort_kayip < en_iyi_kayip:
            en_iyi_kayip = ort_kayip
            en_iyi_model = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_cikis  = model(X_t_t)
                test_kayip  = criterion(test_cikis, y_t_t)
                tahminler   = torch.argmax(test_cikis, dim=1)
                dogru       = (tahminler == y_t_t).float().mean()
            print(f"  Epoch {epoch+1:>3}/{EPOCH} | "
                  f"Eğitim kaybı: {ort_kayip:.4f} | "
                  f"Test kaybı: {test_kayip:.4f} | "
                  f"Doğruluk: %{dogru*100:.1f}")

    # En iyi modeli yükle
    model.load_state_dict(en_iyi_model)

    # Final değerlendirme
    model.eval()
    with torch.no_grad():
        test_cikis = model(X_t_t)
        tahminler  = torch.argmax(test_cikis, dim=1).numpy()

    print(f"\n  ── Final Model Performansı ──────────────────────")
    print(classification_report(
        y_t, tahminler,
        target_names=['SAT', 'BEKLE', 'AL'],
        zero_division=0))

    return model, scaler, X_t_t, y_t

def canli_sinyal_uret(model, scaler, df):
    """Son 30 günün verisini kullanarak gerçek zamanlı sinyal üret"""
    model.eval()

    son_30_gun = df[OZELLIKLER].values[-PENCERE:]
    son_scaled = scaler.transform(son_30_gun)
    X_son      = torch.FloatTensor(son_scaled).unsqueeze(0)

    with torch.no_grad():
        cikis       = model(X_son)
        olasiliklar = torch.softmax(cikis, dim=1).numpy()[0]
        tahmin      = np.argmax(olasiliklar)

    karar = {2: "AL", 0: "SAT", 1: "BEKLE"}[tahmin]
    guven = olasiliklar[tahmin]

    if guven < 0.40:
        karar = "BEKLE"

    son_fiyat = df['Close'].iloc[-1]
    atr       = df['ATR'].iloc[-1]

    return {
        'karar'    : karar,
        'guven'    : guven,
        'al_olas'  : olasiliklar[2],
        'bekle_olas': olasiliklar[1],
        'sat_olas' : olasiliklar[0],
        'fiyat'    : son_fiyat,
        'hedef'    : son_fiyat + atr * 2.5 if karar == "AL" else None,
        'stop'     : son_fiyat - atr * 1.5 if karar == "AL" else None,
    }

def lstm_backtest(model, scaler, df, X_t_t, y_t):
    """LSTM ile backtest"""
    sermaye     = 100000
    hisse_adedi = 0
    pozisyon    = False
    alis_fiyati = 0
    islemler    = []
    portfoy     = []

    bolme   = int((len(df) - PENCERE) * 0.8)
    test_df = df.iloc[bolme + PENCERE:]

    model.eval()
    with torch.no_grad():
        cikislar    = model(X_t_t)
        olasiliklar = torch.softmax(cikislar, dim=1).numpy()
        tahminler   = np.argmax(olasiliklar, axis=1)

    for i in range(len(test_df) - 1):
        fiyat = test_df['Close'].iloc[i]
        tarih = test_df.index[i]

        if i >= len(tahminler):
            break

        tahmin = tahminler[i]
        guven  = olasiliklar[i][tahmin]
        karar  = {2: "AL", 0: "SAT", 1: "BEKLE"}[tahmin]

        if guven < 0.40:
            karar = "BEKLE"

        if pozisyon and alis_fiyati > 0:
            if (fiyat - alis_fiyati) / alis_fiyati <= -0.07:
                karar = "SAT"

        if karar == "AL" and not pozisyon:
            hisse_adedi = int(sermaye / fiyat)
            maliyet     = hisse_adedi * fiyat * 1.001
            if hisse_adedi > 0:
                sermaye    -= maliyet
                pozisyon    = True
                alis_fiyati = fiyat
                islemler.append({
                    'tarih': tarih, 'islem': 'AL',
                    'fiyat': fiyat, 'tutar': maliyet})

        elif karar == "SAT" and pozisyon:
            gelir     = hisse_adedi * fiyat * 0.999
            sermaye  += gelir
            kar_zarar = gelir - islemler[-1]['tutar']
            islemler.append({
                'tarih': tarih, 'islem': 'SAT',
                'fiyat': fiyat, 'kar_zarar': kar_zarar})
            hisse_adedi = 0
            pozisyon    = False
            alis_fiyati = 0

        portfoy.append({'tarih': tarih,
                        'deger': sermaye + hisse_adedi * fiyat,
                        'fiyat': fiyat})

    if pozisyon:
        sermaye += hisse_adedi * test_df['Close'].iloc[-1] * 0.999

    portfoy_df = pd.DataFrame(portfoy)
    islem_df   = pd.DataFrame(islemler)

    # Sonuçlar
    getiri    = (sermaye - 100000) / 100000 * 100
    ilk_fiyat = portfoy_df['fiyat'].iloc[0]
    son_fiyat = portfoy_df['fiyat'].iloc[-1]
    al_tut    = (son_fiyat - ilk_fiyat) / ilk_fiyat * 100

    portfoy_df['tepe']  = portfoy_df['deger'].cummax()
    portfoy_df['dusus'] = (portfoy_df['deger'] - portfoy_df['tepe']) / portfoy_df['tepe'] * 100
    maks_dusus = portfoy_df['dusus'].min()

    satislar  = islem_df[islem_df['islem'] == 'SAT'] if len(islem_df) > 0 else pd.DataFrame()
    kazanan   = len(satislar[satislar['kar_zarar'] > 0]) if len(satislar) > 0 else 0
    basari    = kazanan / len(satislar) * 100 if len(satislar) > 0 else 0

    print(f"\n  ── LSTM Backtest Sonuçları ──────────────────────")
    print(f"  LSTM Getiri    : %{getiri:>+.1f}")
    print(f"  Al-tut Getiri  : %{al_tut:>+.1f}")
    print(f"  Fark           : %{getiri-al_tut:>+.1f}  "
          f"({'LSTM daha iyi' if getiri > al_tut else 'Al-tut daha iyi'})")
    print(f"  İşlem sayısı   : {len(satislar)}")
    print(f"  Başarı oranı   : %{basari:.1f}")
    print(f"  Maks. düşüş    : %{maks_dusus:.1f}")

    return getiri, al_tut

def main():
    print("\n" + "="*55)
    print(f"   LSTM MODELİ — {HISSE}")
    print("="*55)

    print("\n  Veri çekiliyor ve özellikler hazırlanıyor...")
    df = veri_cek(HISSE, PERIYOT)
    df = ozellikler_ekle(df)
    df = hedef_olustur(df)
    print(f"  Toplam {len(df)} günlük veri hazır.")

    model, scaler, X_t_t, y_t = model_egit(df)

    # Gerçek zamanlı sinyal
    print(f"\n  ── Güncel Sinyal ────────────────────────────────")
    sinyal = canli_sinyal_uret(model, scaler, df)
    emoji  = "🟢" if sinyal['karar'] == "AL" else ("🔴" if sinyal['karar'] == "SAT" else "⚪")
    print(f"  Karar   : {emoji} {sinyal['karar']}")
    print(f"  Güven   : %{sinyal['guven']*100:.1f}")
    print(f"  Fiyat   : {sinyal['fiyat']:.2f} TL")
    print(f"  AL olas.: %{sinyal['al_olas']*100:.1f}")
    print(f"  SAT olas: %{sinyal['sat_olas']*100:.1f}")
    if sinyal['hedef']:
        print(f"  Hedef   : {sinyal['hedef']:.2f} TL")
        print(f"  Stop    : {sinyal['stop']:.2f} TL")

    lstm_backtest(model, scaler, df, X_t_t, y_t)

    # Modeli kaydet
    torch.save({
        'model_state': model.state_dict(),
        'scaler'     : scaler,
        'ozellikler' : OZELLIKLER,
        'pencere'    : PENCERE,
        'hisse'      : HISSE,
    }, f"{HISSE.replace('.IS','')}_lstm.pt")
    print(f"\n  Model kaydedildi: {HISSE.replace('.IS','')}_lstm.pt")
    print("="*55)

if __name__ == "__main__":
    main()