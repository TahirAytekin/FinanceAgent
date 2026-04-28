from flask import Flask, render_template_string, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import math
import os
import threading
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── Ayarlar ───────────────────────────────────────────
HISSELER   = ["AKBNK.IS", "GARAN.IS", "YKBNK.IS",
              "EKGYO.IS", "PGSUS.IS", "TCELL.IS",
              "SISE.IS",  "FROTO.IS"]
GUNCELLEME = 300
PORT       = int(os.environ.get('PORT', 5000))
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ───────────────────────────────────────────────────────

app = Flask(__name__)

SISTEM_VERISI = {
    'sinyaller'     : [],
    'piyasa'        : {},
    'son_guncelleme': None,
    'modeller'      : {},
    'grafik_verisi' : {},
    'hazir'         : False,
}

HTML = '''<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BIST Trading Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background:#0f172a; color:#e2e8f0; min-height:100vh; }
  .header { background:#1e293b; border-bottom:1px solid #334155;
            padding:16px 24px; display:flex; align-items:center;
            justify-content:space-between; }
  .header h1 { font-size:20px; font-weight:600; color:#f1f5f9; }
  .badge { background:#22c55e22; border:1px solid #22c55e44; color:#22c55e;
           padding:4px 12px; border-radius:20px; font-size:13px; }
  .badge.kapali { background:#ef444422; border-color:#ef444444; color:#ef4444; }
  .container { padding:24px; max-width:1400px; margin:0 auto; }
  .grid-3 { display:grid; grid-template-columns:repeat(3,1fr);
            gap:16px; margin-bottom:24px; }
  .card { background:#1e293b; border:1px solid #334155;
          border-radius:12px; padding:20px; margin-bottom:24px; }
  .card h2 { font-size:13px; color:#94a3b8; text-transform:uppercase;
             letter-spacing:.05em; margin-bottom:12px; }
  .stat-value { font-size:28px; font-weight:700; color:#f1f5f9; }
  .stat-sub { font-size:13px; color:#64748b; margin-top:4px; }
  .green { color:#22c55e; } .red { color:#ef4444; }
  .yellow { color:#f59e0b; } .blue { color:#3b82f6; }
  .sinyal-tablo { width:100%; border-collapse:collapse; font-size:14px; }
  .sinyal-tablo th { text-align:left; padding:10px 12px; color:#64748b;
                     font-size:12px; text-transform:uppercase;
                     border-bottom:1px solid #334155; }
  .sinyal-tablo td { padding:12px; border-bottom:1px solid #1e293b; }
  .sinyal-tablo tr:hover td { background:#0f172a; }
  .pill { display:inline-block; padding:3px 10px; border-radius:20px;
          font-size:12px; font-weight:600; }
  .pill.al  { background:#22c55e22; color:#22c55e; border:1px solid #22c55e44; }
  .pill.sat { background:#ef444422; color:#ef4444; border:1px solid #ef444444; }
  .pill.bekle { background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b44; }
  .guven-bar { background:#334155; border-radius:4px; height:6px; margin-top:4px; }
  .guven-bar-ic { height:100%; border-radius:4px;
                  background:linear-gradient(90deg,#3b82f6,#22c55e); }
  .piyasa-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }
  .piyasa-item { background:#0f172a; border-radius:8px; padding:12px; }
  .piyasa-item .label { font-size:12px; color:#64748b; margin-bottom:4px; }
  .piyasa-item .value { font-size:18px; font-weight:600; }
  #grafik-alan { height:400px; }
  .yukluyor { text-align:center; padding:40px; color:#475569; }
  .rejim-badge { display:inline-block; padding:6px 16px; border-radius:8px;
                 font-weight:600; font-size:15px; }
  .rejim-boga { background:#22c55e22; color:#22c55e; }
  .rejim-ayi  { background:#ef444422; color:#ef4444; }
  .rejim-yatay { background:#f59e0b22; color:#f59e0b; }
</style>
</head>
<body>
<div class="header">
  <h1>BIST Trading Dashboard</h1>
  <div style="display:flex;gap:12px;align-items:center">
    <span id="borsa-durum" class="badge">● YÜKLENIYOR</span>
    <span id="son-guncelleme" style="font-size:13px;color:#475569"></span>
  </div>
</div>
<div class="container">
  <div class="grid-3">
    <div class="card" style="margin-bottom:0">
      <h2>Piyasa Rejimi</h2>
      <div id="rejim-badge" class="rejim-badge rejim-yatay">—</div>
      <div class="stat-sub" id="rejim-aciklama">Yükleniyor...</div>
    </div>
    <div class="card" style="margin-bottom:0">
      <h2>Aktif Sinyaller</h2>
      <div class="stat-value" id="sinyal-sayisi">—</div>
      <div class="stat-sub" id="sinyal-ozet">AL / SAT / BEKLE</div>
    </div>
    <div class="card" style="margin-bottom:0">
      <h2>BIST100 & Kur</h2>
      <div class="piyasa-grid" id="piyasa-grid">
        <div class="yukluyor">Yükleniyor...</div>
      </div>
    </div>
  </div>
  <br>
  <div class="card">
    <h2>Hisse Sinyalleri</h2>
    <div id="sinyal-tablo-alani">
      <div class="yukluyor">Modeller eğitiliyor, lütfen bekle...</div>
    </div>
  </div>
  <div class="card">
    <h2>Track Record & Performans</h2>
    <div id="track-record-alani">
      <div class="yukluyor">Yükleniyor...</div>
    </div>
  </div>
  <div class="card">
    <h2>Fiyat Grafiği</h2>
    <select id="hisse-sec" onchange="grafikGuncelle()"
      style="background:#0f172a;color:#e2e8f0;border:1px solid #334155;
             padding:6px 12px;border-radius:6px;margin-bottom:12px;font-size:13px">
      {% for h in hisseler %}
      <option value="{{ h }}">{{ h.replace('.IS','') }}</option>
      {% endfor %}
    </select>
    <div id="grafik-alan"></div>
  </div>
</div>

<script>
let grafikVerisi = {};

function veriCek() {
  fetch('/api/veri')
    .then(r => r.json())
    .then(data => {
      grafikVerisi = data.grafik_verisi || {};
      sayfaGuncelle(data);
    })
    .catch(e => console.log('Veri hatası:', e));
}

function sayfaGuncelle(data) {
  if (!data.hazir) {
    document.getElementById('sinyal-tablo-alani').innerHTML =
      '<div class="yukluyor">Modeller eğitiliyor, lütfen bekle...</div>';
    return;
  }

  const borsa = data.borsa_acik;
  const durumEl = document.getElementById('borsa-durum');
  durumEl.textContent = borsa ? '● BORSA AÇIK' : '● BORSA KAPALI';
  durumEl.className   = 'badge' + (borsa ? '' : ' kapali');

  document.getElementById('son-guncelleme').textContent =
    'Son: ' + (data.son_guncelleme || '—');

  const rejim   = data.piyasa?.rejim || 'YATAY';
  const rejimEl = document.getElementById('rejim-badge');
  rejimEl.textContent = (data.piyasa?.emoji || '') + ' ' + rejim;
  rejimEl.className   = 'rejim-badge rejim-' +
    (rejim.includes('BOGA') ? 'boga' : rejim.includes('AYI') ? 'ayi' : 'yatay');
  document.getElementById('rejim-aciklama').textContent =
    data.piyasa?.aciklama || '';

  const sinyaller = data.sinyaller || [];
  const al    = sinyaller.filter(s => s.karar === 'AL').length;
  const sat   = sinyaller.filter(s => s.karar === 'SAT').length;
  const bekle = sinyaller.filter(s => s.karar === 'BEKLE').length;
  document.getElementById('sinyal-sayisi').textContent = sinyaller.length;
  document.getElementById('sinyal-ozet').innerHTML =
    `<span class="green">${al} AL</span> / ` +
    `<span class="red">${sat} SAT</span> / ` +
    `<span class="yellow">${bekle} BEKLE</span>`;

  const pGrid = document.getElementById('piyasa-grid');
  if (data.piyasa?.bist_son) {
    pGrid.innerHTML = `
      <div class="piyasa-item">
        <div class="label">BIST100</div>
        <div class="value">${Number(data.piyasa.bist_son).toLocaleString('tr-TR')}</div>
      </div>
      <div class="piyasa-item">
        <div class="label">1 Aylık</div>
        <div class="value ${data.piyasa.bist_getiri_1ay >= 0 ? 'green' : 'red'}">
          %${Number(data.piyasa.bist_getiri_1ay).toFixed(1)}
        </div>
      </div>
      <div class="piyasa-item">
        <div class="label">USD/TRY</div>
        <div class="value">${Number(data.piyasa.usdtry).toFixed(2)}</div>
      </div>
      <div class="piyasa-item">
        <div class="label">RSI</div>
        <div class="value ${data.piyasa.bist_rsi < 40 ? 'green' :
                            data.piyasa.bist_rsi > 60 ? 'red' : 'yellow'}">
          ${Number(data.piyasa.bist_rsi).toFixed(1)}
        </div>
      </div>`;
  }

  if (sinyaller.length > 0) {
    let tablo = `<table class="sinyal-tablo"><thead><tr>
      <th>Hisse</th><th>Fiyat</th><th>Değişim</th><th>RSI</th>
      <th>Karar</th><th>Güven</th><th>Hedef</th><th>Stop</th>
    </tr></thead><tbody>`;
    sinyaller.forEach(s => {
      const dR = s.degisim >= 0 ? 'green' : 'red';
      const dI = s.degisim >= 0 ? '+' : '';
      const kC = s.karar === 'AL' ? 'al' : s.karar === 'SAT' ? 'sat' : 'bekle';
      const gP = (s.guven * 100).toFixed(0);
      tablo += `<tr>
        <td style="font-weight:600">${s.sembol.replace('.IS','')}</td>
        <td>${Number(s.fiyat).toFixed(2)} ₺</td>
        <td class="${dR}">${dI}${Number(s.degisim).toFixed(2)}%</td>
        <td class="${s.rsi<40?'green':s.rsi>60?'red':'yellow'}">${Number(s.rsi).toFixed(1)}</td>
        <td><span class="pill ${kC}">${s.karar}</span></td>
        <td>%${gP}<div class="guven-bar">
          <div class="guven-bar-ic" style="width:${gP}%"></div></div></td>
        <td class="green">${s.hedef ? Number(s.hedef).toFixed(2)+' ₺' : '—'}</td>
        <td class="red">${s.stop ? Number(s.stop).toFixed(2)+' ₺' : '—'}</td>
      </tr>`;
    });
    tablo += '</tbody></table>';
    document.getElementById('sinyal-tablo-alani').innerHTML = tablo;
  }

  const tr = data.track_record;
  if (tr) {
    const bR = tr.basari >= 55 ? 'green' : tr.basari >= 45 ? 'yellow' : 'red';
    const kR = tr.ort_kar >= 0 ? 'green' : 'red';
    let trHTML = `
      <div style="display:grid;grid-template-columns:repeat(4,1fr);
                  gap:12px;margin-bottom:20px">
        <div style="background:#0f172a;border-radius:8px;padding:14px;text-align:center">
          <div style="font-size:11px;color:#64748b;margin-bottom:6px">TOPLAM SİNYAL</div>
          <div style="font-size:22px;font-weight:700">${tr.toplam}</div>
        </div>
        <div style="background:#0f172a;border-radius:8px;padding:14px;text-align:center">
          <div style="font-size:11px;color:#64748b;margin-bottom:6px">TAMAMLANAN</div>
          <div style="font-size:22px;font-weight:700">${tr.tamamlanan}</div>
        </div>
        <div style="background:#0f172a;border-radius:8px;padding:14px;text-align:center">
          <div style="font-size:11px;color:#64748b;margin-bottom:6px">BAŞARI ORANI</div>
          <div style="font-size:22px;font-weight:700" class="${bR}">%${tr.basari}</div>
        </div>
        <div style="background:#0f172a;border-radius:8px;padding:14px;text-align:center">
          <div style="font-size:11px;color:#64748b;margin-bottom:6px">ORT. KAR/ZARAR</div>
          <div style="font-size:22px;font-weight:700" class="${kR}">
            %${tr.ort_kar > 0 ? '+' : ''}${tr.ort_kar}</div>
        </div>
      </div>`;
    if (tr.son_sinyaller && tr.son_sinyaller.length > 0) {
      trHTML += `<table class="sinyal-tablo"><thead><tr>
        <th>Tarih</th><th>Hisse</th><th>Karar</th>
        <th>Giriş</th><th>Çıkış</th><th>Kar/Zarar</th><th>Sonuç</th>
      </tr></thead><tbody>`;
      tr.son_sinyaller.forEach(s => {
        const sR = s.sonuc==='KAZANDI' ? 'green' : s.sonuc==='KAYBETTI' ? 'red' : 'yellow';
        const kC = s.karar==='AL' ? 'al' : s.karar==='SAT' ? 'sat' : 'bekle';
        const kStr = s.kar_zarar ?
          `%${parseFloat(s.kar_zarar)>0?'+':''}${parseFloat(s.kar_zarar).toFixed(1)}` : '—';
        trHTML += `<tr>
          <td style="font-size:12px">${s.zaman||'—'}</td>
          <td style="font-weight:600">${(s.sembol||'').replace('.IS','')}</td>
          <td><span class="pill ${kC}">${s.karar}</span></td>
          <td>${s.fiyat_giris ? parseFloat(s.fiyat_giris).toFixed(2)+' ₺' : '—'}</td>
          <td>${s.fiyat_cikis ? parseFloat(s.fiyat_cikis).toFixed(2)+' ₺' : '—'}</td>
          <td class="${s.kar_zarar&&parseFloat(s.kar_zarar)>=0?'green':'red'}">${kStr}</td>
          <td class="${sR}" style="font-weight:600">${s.sonuc||'Bekliyor'}</td>
        </tr>`;
      });
      trHTML += '</tbody></table>';
    } else {
      trHTML += '<p style="color:#475569;text-align:center;padding:20px">' +
        'Henüz tamamlanan sinyal yok — 3 gün sonra sonuçlar burada görünecek.</p>';
    }
    document.getElementById('track-record-alani').innerHTML = trHTML;
  }

  grafikGuncelle();
}

function grafikGuncelle() {
  const hisse = document.getElementById('hisse-sec').value;
  const veri  = grafikVerisi[hisse];
  if (!veri) return;

  Plotly.newPlot('grafik-alan', [
    { type:'candlestick', x:veri.tarihler,
      open:veri.open, high:veri.high, low:veri.low, close:veri.close,
      name:hisse.replace('.IS',''),
      increasing:{line:{color:'#22c55e'}},
      decreasing:{line:{color:'#ef4444'}} },
    { type:'scatter', x:veri.tarihler, y:veri.ma20,
      name:'MA20', line:{color:'#f59e0b',width:1.5} },
    { type:'scatter', x:veri.tarihler, y:veri.ma50,
      name:'MA50', line:{color:'#8b5cf6',width:1.5} }
  ], {
    paper_bgcolor:'#1e293b', plot_bgcolor:'#1e293b',
    font:{color:'#94a3b8',size:12},
    xaxis:{gridcolor:'#334155',rangeslider:{visible:false}},
    yaxis:{gridcolor:'#334155'},
    margin:{t:20,r:20,b:40,l:60},
    legend:{bgcolor:'#0f172a',bordercolor:'#334155'},
    showlegend:true
  }, {responsive:true, displayModeBar:false});
}

// Her 10 saniyede bir veri çek
veriCek();
setInterval(veriCek, 10000);
</script>
</body>
</html>'''

# ── BACKEND ────────────────────────────────────────────

def borsa_acik_mi():
    from datetime import time as dtime
    return dtime(10, 0) <= datetime.now().time() <= dtime(18, 10)

def piyasa_bilgisi_cek():
    try:
        bist   = yf.Ticker("XU100.IS").history(period="3mo", interval="1d")
        usdtry = yf.Ticker("USDTRY=X").history(period="3mo", interval="1d")
        bist.index   = bist.index.tz_localize(None)
        usdtry.index = usdtry.index.tz_localize(None)
        if len(bist) < 5:
            raise ValueError("Yetersiz veri")

        bist['MA20']       = ta.sma(bist['Close'], length=20)
        bist['RSI']        = ta.rsi(bist['Close'], length=14)
        bist['Getiri_1ay'] = bist['Close'].pct_change(20)
        bist = bist.dropna()
        son  = bist.iloc[-1]

        usdtry_son = float(usdtry['Close'].iloc[-1])
        idx        = min(20, len(usdtry)-1)
        kur_eski   = float(usdtry['Close'].iloc[-idx])
        kur_deg    = (usdtry_son - kur_eski) / kur_eski * 100

        puan = 0
        puan += 1 if son['Close'] > son['MA20'] else -1
        puan += 1 if son['RSI'] > 60 else (-1 if son['RSI'] < 40 else 0)
        puan += 1 if son['Getiri_1ay'] > 0 else -1
        puan += -1 if kur_deg > 2 else 0

        if puan >= 2:
            rejim, emoji, aciklama, carpani = "BOGA", "🟢", "Yükseliş trendi", 1.1
        elif puan <= -2:
            rejim, emoji, aciklama, carpani = "AYI", "🔴", "Düşüş trendi", 0.7
        else:
            rejim, emoji, aciklama, carpani = "YATAY", "🟡", "Kararsız piyasa", 0.9

        return {
            'rejim': rejim, 'emoji': emoji, 'aciklama': aciklama,
            'carpani': carpani, 'bist_son': float(son['Close']),
            'bist_rsi': float(son['RSI']),
            'bist_getiri_1ay': float(son['Getiri_1ay'] * 100),
            'usdtry': usdtry_son, 'kur_degisim': kur_deg,
        }
    except Exception as e:
        return {'rejim':'YATAY','emoji':'🟡','aciklama':'Veri alınamadı',
                'carpani':0.9,'bist_son':0,'bist_rsi':50,
                'bist_getiri_1ay':0,'usdtry':0,'kur_degisim':0}

def ozellikler_ekle(df):
    df = df.copy()
    df['RSI']        = ta.rsi(df['Close'], length=14)
    df['RSI_fast']   = ta.rsi(df['Close'], length=7)
    macd             = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']       = macd['MACD_12_26_9']
    df['MACD_hist']  = macd['MACDh_12_26_9']
    df['MA5']        = ta.sma(df['Close'], length=5)
    df['MA20']       = ta.sma(df['Close'], length=20)
    df['MA50']       = ta.sma(df['Close'], length=50)
    df['MA200']      = ta.sma(df['Close'], length=200)
    bb               = ta.bbands(df['Close'], length=20, std=2)
    bb_s             = bb.columns.tolist()
    df['BB_ust']     = bb[bb_s[2]]
    df['BB_alt']     = bb[bb_s[0]]
    df['BB_genislik']= (df['BB_ust'] - bb[bb_s[1]]) / bb[bb_s[1]]
    df['ATR']        = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Volatilite'] = df['Close'].pct_change().rolling(20).std()
    df['Hacim_MA20'] = ta.sma(df['Volume'], length=20)
    df['Hacim_Oran'] = df['Volume'] / df['Hacim_MA20']
    df['RSI_Norm']   = (df['RSI'] - 50) / 50
    df['RSI_f_Norm'] = (df['RSI_fast'] - 50) / 50
    df['BB_Konum']   = (df['Close'] - df['BB_alt']) / (df['BB_ust'] - df['BB_alt'])
    df['MA5_Fark']   = (df['Close'] - df['MA5'])  / df['MA5']
    df['MA20_Fark']  = (df['Close'] - df['MA20']) / df['MA20']
    df['MA50_Fark']  = (df['Close'] - df['MA50']) / df['MA50']
    df['MA200_Fark'] = (df['Close'] - df['MA200'])/ df['MA200']
    df['MACD_Norm']  = df['MACD'] - ta.sma(df['MACD'], length=9)
    df['Trend_Guc']  = (df['MA5'] - df['MA50']) / df['MA50']
    for g in [1, 3, 5, 10, 20]:
        df[f'Getiri_{g}g'] = df['Close'].pct_change(g)
    df['Kanat']      = (df['High'] - df['Low']) / df['Close']
    df['Govde']      = abs(df['Close'] - df['Open']) / df['Close']
    df['Yon']        = np.where(df['Close'] > df['Open'], 1.0, -1.0)
    df['52H_Yuzde']  = df['Close'] / df['Close'].rolling(252).max()
    df['RSI_Trend']  = df['RSI'] - df['RSI'].shift(5)
    df['Hacim_Fiyat']= df['Getiri_1g'] * df['Hacim_Oran']
    return df.dropna()

OZELLIKLER = [
    'RSI_Norm','RSI_f_Norm','MACD_Norm','MACD_hist',
    'BB_Konum','BB_genislik','MA5_Fark','MA20_Fark',
    'MA50_Fark','MA200_Fark','Trend_Guc',
    'Getiri_1g','Getiri_3g','Getiri_5g','Getiri_10g','Getiri_20g',
    'Hacim_Oran','ATR','Volatilite','Kanat','Govde','Yon',
    '52H_Yuzde','RSI_Trend','Hacim_Fiyat'
]

def model_egit(sembol):
    df = yf.Ticker(sembol).history(period="5y", interval="1d")
    df = df[['Open','High','Low','Close','Volume']]
    df.index = df.index.tz_localize(None)
    df = ozellikler_ekle(df)
    df['Gelecek'] = df['Close'].shift(-3) / df['Close'] - 1
    df['Hedef']   = df['Gelecek'].apply(
        lambda g: 2 if g >= 0.015 else (0 if g <= -0.015 else 1))
    df = df.dropna()
    X = df[OZELLIKLER].values
    y = df['Hedef'].values
    bolme  = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_e    = scaler.fit_transform(X[:bolme])
    model  = VotingClassifier(estimators=[
        ('rf',  RandomForestClassifier(n_estimators=100, max_depth=6,
                 random_state=42, class_weight='balanced')),
        ('xgb', XGBClassifier(n_estimators=100, max_depth=5,
                 learning_rate=0.05, random_state=42,
                 eval_metric='mlogloss', verbosity=0)),
        ('lgbm',LGBMClassifier(n_estimators=100, max_depth=5,
                 learning_rate=0.05, random_state=42,
                 class_weight='balanced', verbose=-1))
    ], voting='soft')
    model.fit(X_e, y[:bolme])
    return model, scaler, df

def sinyal_uret(sembol, model, scaler, df, carpani=1.0):
    try:
        ticker    = yf.Ticker(sembol)
        son_fiyat = ticker.fast_info.last_price
        onceki    = ticker.fast_info.regular_market_previous_close
        degisim   = ((son_fiyat - onceki) / onceki * 100) if onceki else 0
        son_X     = scaler.transform([df[OZELLIKLER].iloc[-1].values])
        tahmin    = model.predict(son_X)[0]
        olas      = model.predict_proba(son_X)[0]
        guven     = float(max(olas))
        esik      = 0.38 / carpani
        karar     = {2:"AL", 0:"SAT", 1:"BEKLE"}[tahmin]
        if guven < esik:
            karar = "BEKLE"
        atr = guvenli_sayi(df['ATR'].iloc[-1])
        return {
    'sembol' : sembol,
    'fiyat'  : round(guvenli_sayi(son_fiyat), 2),
    'degisim': round(guvenli_sayi(degisim), 2),
    'rsi'    : round(guvenli_sayi(df['RSI'].iloc[-1]), 1),
    'karar'  : karar,
    'guven'  : round(guvenli_sayi(guven), 3),
    'hedef'  : round(guvenli_sayi(son_fiyat + atr*2.5), 2) if karar=="AL" else None,
    'stop'   : round(guvenli_sayi(son_fiyat - atr*1.5), 2) if karar=="AL" else None,
}
    except:
        return None

def grafik_verisi_hazirla(sembol, df):
    s = df.tail(90)
    return {
        'tarihler': [str(t)[:10] for t in s.index],
        'open' : s['Open'].round(2).tolist(),
        'high' : s['High'].round(2).tolist(),
        'low'  : s['Low'].round(2).tolist(),
        'close': s['Close'].round(2).tolist(),
        'ma20' : s['MA20'].round(2).tolist(),
        'ma50' : s['MA50'].round(2).tolist(),
    }

def track_record_oku():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "track_record.csv"), encoding='utf-8-sig')
        tamamlanan = df[df['sonuc'].isin(['KAZANDI','KAYBETTI'])]
        if len(tamamlanan) == 0:
            return {'toplam':len(df),'tamamlanan':0,'kazanan':0,
                    'basari':0,'ort_kar':0,
                    'son_sinyaller':df.tail(10).to_dict('records')}
        kazanan = len(tamamlanan[tamamlanan['sonuc']=='KAZANDI'])
        basari  = kazanan / len(tamamlanan) * 100
        ort_kar = tamamlanan['kar_zarar'].astype(float).mean()
        return {
            'toplam'      : len(df),
            'tamamlanan'  : len(tamamlanan),
            'kazanan'     : kazanan,
            'basari'      : round(basari, 1),
            'ort_kar'     : round(float(ort_kar), 2),
            'son_sinyaller': df.tail(10).iloc[::-1].to_dict('records'),
        }
    except:
        return {'toplam':0,'tamamlanan':0,'kazanan':0,
                'basari':0,'ort_kar':0,'son_sinyaller':[]}

def sistem_baslat():
    print("\nModeller eğitiliyor...")
    for s in HISSELER:
        for deneme in range(3):
            try:
                print(f"  {s} eğitiliyor... (deneme {deneme+1})")
                model, scaler, df = model_egit(s)
                SISTEM_VERISI['modeller'][s] = (model, scaler, df)
                print(f"  {s} ✅")
                break
            except Exception as e:
                print(f"  {s} ❌ deneme {deneme+1}: {e}")
                if deneme < 2:
                    time.sleep(15)
        time.sleep(3)

    print("Modeller hazır!\n")
    SISTEM_VERISI['hazir'] = True

    while True:
        try:
            piyasa  = piyasa_bilgisi_cek()
            carpani = piyasa.get('carpani', 1.0)
            sinyaller, grafik_v = [], {}

            for s, (model, scaler, df) in SISTEM_VERISI['modeller'].items():
                sinyal = sinyal_uret(s, model, scaler, df, carpani)
                if sinyal:
                    sinyaller.append(sinyal)
                grafik_v[s] = grafik_verisi_hazirla(s, df)

            SISTEM_VERISI['sinyaller']      = sinyaller
            SISTEM_VERISI['piyasa']         = piyasa
            SISTEM_VERISI['grafik_verisi']  = grafik_v
            SISTEM_VERISI['son_guncelleme'] = datetime.now().strftime("%H:%M:%S")

            print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(sinyaller)} sinyal güncellendi.")
        except Exception as e:
            print(f"Güncelleme hatası: {e}")

        time.sleep(GUNCELLEME)

@app.route('/')
def index():
    return render_template_string(HTML, hisseler=HISSELER)

def guvenli_sayi(x, default=0):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except:
        return default
    
def json_temizle(data):
    if isinstance(data, dict):
        return {k: json_temizle(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_temizle(i) for i in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        return data
_egitim_basladi = False

@app.route('/api/veri')
def api_veri():
    global _egitim_basladi
    if not _egitim_basladi:
        _egitim_basladi = True
        threading.Thread(target=sistem_baslat, daemon=True).start()
        print("İlk istek alındı — model eğitimi başlatıldı.")

    tr_data = track_record_oku()
    data = {
        'hazir': SISTEM_VERISI['hazir'],
        'sinyaller': SISTEM_VERISI['sinyaller'],
        'piyasa': SISTEM_VERISI['piyasa'],
        'grafik_verisi': SISTEM_VERISI['grafik_verisi'],
        'borsa_acik': borsa_acik_mi(),
        'son_guncelleme': SISTEM_VERISI['son_guncelleme'],
        'track_record': tr_data,
    }
    return jsonify(json_temizle(data))

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  BIST TRADING DASHBOARD BAŞLATILIYOR")
    print(f"  Tarayıcıda aç: http://localhost:{PORT}")
    print("  Durdurmak için CTRL+C")
    print("="*55)

    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)