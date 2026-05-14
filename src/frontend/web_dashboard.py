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
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:#0f172a; color:#e2e8f0; min-height:100vh; }
  .header { background:#1e293b; border-bottom:1px solid #334155;
            padding:13px 20px; display:flex; align-items:center;
            justify-content:space-between; position:sticky; top:0; z-index:100; }
  .header h1 { font-size:17px; font-weight:700; color:#f1f5f9; }
  .badge { background:#22c55e22; border:1px solid #22c55e44; color:#22c55e;
           padding:4px 10px; border-radius:20px; font-size:12px; }
  .badge.kapali { background:#ef444422; border-color:#ef444444; color:#ef4444; }
  .tab-nav { background:#1e293b; border-bottom:1px solid #334155;
             display:flex; padding:0 20px; overflow-x:auto; }
  .tab-nav::-webkit-scrollbar { display:none; }
  .tab-btn { padding:13px 18px; border:none; background:none; color:#64748b;
             cursor:pointer; font-size:13px; font-weight:500;
             border-bottom:2px solid transparent; white-space:nowrap; transition:all .2s; }
  .tab-btn:hover { color:#94a3b8; }
  .tab-btn.active { color:#3b82f6; border-bottom-color:#3b82f6; }
  .tab-pane { display:none; }
  .tab-pane.active { display:block; }
  .container { padding:18px 20px; max-width:1400px; margin:0 auto; }
  .card { background:#1e293b; border:1px solid #334155; border-radius:12px;
          padding:18px; margin-bottom:18px; }
  .card-title { font-size:12px; color:#94a3b8; text-transform:uppercase;
                letter-spacing:.05em; margin-bottom:14px; }
  .stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:18px; }
  .stat-card { background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px; }
  .stat-head { display:flex; justify-content:space-between; align-items:flex-start;
               margin-bottom:10px; }
  .stat-label { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.04em; }
  .stat-icon { font-size:18px; }
  .stat-value { font-size:24px; font-weight:700; color:#f1f5f9; line-height:1.1; }
  .stat-sub { font-size:11px; color:#64748b; margin-top:5px; }
  .pill { display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
  .pill.al   { background:#22c55e22; color:#22c55e; border:1px solid #22c55e44; }
  .pill.sat  { background:#ef444422; color:#ef4444; border:1px solid #ef444444; }
  .pill.bekle{ background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b44; }
  .rejim-badge { display:inline-block; padding:5px 14px; border-radius:8px; font-weight:600; font-size:14px; }
  .rejim-boga  { background:#22c55e22; color:#22c55e; }
  .rejim-ayi   { background:#ef444422; color:#ef4444; }
  .rejim-yatay { background:#f59e0b22; color:#f59e0b; }
  .sektor-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; }
  .sektor-kart { background:#0f172a; border:1px solid #334155; border-radius:10px;
                 padding:14px; text-align:center; transition:border-color .2s; }
  .sektor-kart:hover { border-color:#3b82f655; }
  .sektor-icon { font-size:22px; margin-bottom:6px; }
  .sektor-name { font-size:11px; color:#64748b; margin-bottom:4px; }
  .sektor-val  { font-size:17px; font-weight:700; }
  .sektor-durum{ font-size:11px; color:#64748b; margin-top:3px; }
  .tablo { width:100%; border-collapse:collapse; font-size:13px; }
  .tablo th { text-align:left; padding:9px 11px; color:#64748b; font-size:11px;
              text-transform:uppercase; border-bottom:1px solid #334155; white-space:nowrap; }
  .tablo td { padding:10px 11px; border-bottom:1px solid #1e293b; }
  .tablo tr.al-row td { background:#22c55e08; }
  .tablo tr.sat-row td { background:#ef444408; }
  .tablo tr.bekle-row td { background:#f59e0b08; }
  .tablo tr:hover td { filter:brightness(1.2); }
  .guven-bar { background:#334155; border-radius:3px; height:4px; margin-top:4px; width:72px; }
  .guven-fill { height:100%; border-radius:3px; background:linear-gradient(90deg,#3b82f6,#22c55e); }
  .chart-ctrl { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
  .sel { background:#0f172a; color:#e2e8f0; border:1px solid #334155;
         padding:6px 12px; border-radius:6px; font-size:13px; cursor:pointer; outline:none; }
  .period-btn { background:#0f172a; border:1px solid #334155; color:#94a3b8;
                padding:5px 13px; border-radius:6px; cursor:pointer; font-size:12px; transition:all .2s; }
  .period-btn.active { background:#3b82f622; border-color:#3b82f6; color:#3b82f6; }
  .period-btn:hover { border-color:#3b82f677; }
  .tr-stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:18px; }
  .tr-stat { background:#0f172a; border:1px solid #334155; border-radius:10px;
             padding:14px; text-align:center; }
  .tr-stat-label { font-size:11px; color:#64748b; text-transform:uppercase; margin-bottom:7px; }
  .tr-stat-value { font-size:22px; font-weight:700; }
  .p-form { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px; }
  .p-input { background:#0f172a; border:1px solid #334155; color:#e2e8f0;
             padding:8px 12px; border-radius:8px; font-size:13px; outline:none;
             flex:1; min-width:110px; }
  .p-input:focus { border-color:#3b82f6; }
  .btn-add { background:#3b82f6; color:#fff; border:none; padding:8px 18px;
             border-radius:8px; cursor:pointer; font-size:13px; font-weight:600; white-space:nowrap; }
  .btn-add:hover { background:#2563eb; }
  .btn-del { background:#ef444422; color:#ef4444; border:1px solid #ef444444;
             padding:4px 10px; border-radius:6px; cursor:pointer; font-size:11px; }
  .btn-del:hover { background:#ef444433; }
  .alarm-badge { display:inline-block; padding:3px 8px; border-radius:12px; font-size:11px; font-weight:600;
                 background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b33; }
  .alarm-hit   { background:#22c55e22; color:#22c55e; border-color:#22c55e33; }
  .empty-state { text-align:center; padding:32px; color:#475569; font-size:13px; }
  .green { color:#22c55e; } .red { color:#ef4444; } .yellow { color:#f59e0b; } .blue { color:#3b82f6; }
  @media (max-width:768px) {
    .stat-grid, .tr-stat-grid { grid-template-columns:repeat(2,1fr); }
    .sektor-grid { grid-template-columns:repeat(2,1fr); }
    .tablo th:nth-child(n+6), .tablo td:nth-child(n+6) { display:none; }
    .container { padding:12px; }
  }
  @media (max-width:480px) {
    .stat-value { font-size:20px; }
    .tr-stat-value { font-size:18px; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>📊 BIST Trading</h1>
  <div style="display:flex;gap:10px;align-items:center">
    <span id="borsa-durum" class="badge">● YÜKLENIYOR</span>
    <span id="son-guncelleme" style="font-size:11px;color:#475569"></span>
  </div>
</div>

<nav class="tab-nav">
  <button class="tab-btn active" onclick="tabAc('sinyaller',this)">📈 Sinyaller</button>
  <button class="tab-btn" onclick="tabAc('grafik',this)">📉 Grafik</button>
  <button class="tab-btn" onclick="tabAc('trackrecord',this)">🏆 Track Record</button>
  <button class="tab-btn" onclick="tabAc('portfoy',this)">💼 Portföy</button>
</nav>

<!-- TAB 1: SİNYALLER -->
<div id="tab-sinyaller" class="tab-pane active">
<div class="container">
  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-head">
        <div class="stat-label">Piyasa Rejimi</div>
        <div class="stat-icon">🌐</div>
      </div>
      <div id="rejim-badge" class="rejim-badge rejim-yatay">—</div>
      <div class="stat-sub" id="rejim-aciklama">Yükleniyor...</div>
    </div>
    <div class="stat-card">
      <div class="stat-head">
        <div class="stat-label">Aktif Sinyaller</div>
        <div class="stat-icon">🎯</div>
      </div>
      <div class="stat-value" id="sinyal-sayisi">—</div>
      <div class="stat-sub" id="sinyal-ozet">AL / SAT / BEKLE</div>
    </div>
    <div class="stat-card">
      <div class="stat-head">
        <div class="stat-label">BIST100</div>
        <div class="stat-icon">📊</div>
      </div>
      <div class="stat-value" id="bist-deger">—</div>
      <div class="stat-sub" id="bist-sub">USD/TRY: —</div>
    </div>
    <div class="stat-card">
      <div class="stat-head">
        <div class="stat-label">Başarı Oranı</div>
        <div class="stat-icon">🏆</div>
      </div>
      <div class="stat-value" id="basari-deger">—</div>
      <div class="stat-sub" id="basari-sub">Tamamlanan: —</div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">🏭 Sektör Analizi</div>
    <div class="sektor-grid" id="sektor-grid">
      <div class="empty-state">Yükleniyor...</div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">📋 Hisse Sinyalleri</div>
    <div id="sinyal-tablo-alani">
      <div class="empty-state">Modeller eğitiliyor, lütfen bekle...</div>
    </div>
  </div>
</div>
</div>

<!-- TAB 2: GRAFİK -->
<div id="tab-grafik" class="tab-pane">
<div class="container">
  <div class="card">
    <div class="chart-ctrl">
      <select id="hisse-sec" class="sel" onchange="grafikGuncelle()">
        {% for h in hisseler %}
        <option value="{{ h }}">{{ h.replace('.IS','') }}</option>
        {% endfor %}
      </select>
      <button class="period-btn" onclick="periodSec(30,this)">1A</button>
      <button class="period-btn active" onclick="periodSec(90,this)">3A</button>
      <button class="period-btn" onclick="periodSec(180,this)">6A</button>
      <button class="period-btn" onclick="periodSec(252,this)">1Y</button>
    </div>
    <div id="grafik-alan" style="height:380px"></div>
    <div id="hacim-alan" style="height:110px;margin-top:2px"></div>
    <div id="rsi-alan" style="height:110px;margin-top:2px"></div>
  </div>
</div>
</div>

<!-- TAB 3: TRACK RECORD -->
<div id="tab-trackrecord" class="tab-pane">
<div class="container">
  <div class="tr-stat-grid">
    <div class="tr-stat">
      <div class="tr-stat-label">📊 Toplam Sinyal</div>
      <div class="tr-stat-value" id="tr-toplam">—</div>
    </div>
    <div class="tr-stat">
      <div class="tr-stat-label">✅ Tamamlanan</div>
      <div class="tr-stat-value" id="tr-tamamlanan">—</div>
    </div>
    <div class="tr-stat">
      <div class="tr-stat-label">🎯 Başarı Oranı</div>
      <div class="tr-stat-value" id="tr-basari">—</div>
    </div>
    <div class="tr-stat">
      <div class="tr-stat-label">💰 Ort. Kar/Zarar</div>
      <div class="tr-stat-value" id="tr-ort-kar">—</div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">📈 Kümülatif Performans</div>
    <div id="perf-grafik" style="height:240px"></div>
  </div>
  <div class="card">
    <div class="card-title">📋 Sinyal Geçmişi</div>
    <div id="track-record-alani">
      <div class="empty-state">Yükleniyor...</div>
    </div>
  </div>
</div>
</div>

<!-- TAB 4: PORTFÖY -->
<div id="tab-portfoy" class="tab-pane">
<div class="container">
  <div class="card">
    <div class="card-title">💼 Portföy Takibi</div>
    <div class="p-form">
      <input class="p-input" id="p-sembol" placeholder="Sembol (örn: AKBNK)" list="h-list" style="max-width:140px">
      <datalist id="h-list">{% for h in hisseler %}<option value="{{ h.replace('.IS','') }}">{% endfor %}</datalist>
      <input class="p-input" id="p-adet" placeholder="Adet" type="number" style="max-width:100px">
      <input class="p-input" id="p-maliyet" placeholder="Ort. Maliyet ₺" type="number" style="max-width:150px">
      <button class="btn-add" onclick="portfoyEkle()">+ Ekle</button>
    </div>
    <div id="portfoy-tablo"><div class="empty-state">Portföy boş.</div></div>
    <div id="portfoy-ozet" style="display:none;margin-top:14px;padding:14px;background:#0f172a;border-radius:10px;display:flex;gap:20px;flex-wrap:wrap;font-size:13px"></div>
  </div>
  <div class="card">
    <div class="card-title">🔔 Fiyat Alarmları</div>
    <div class="p-form">
      <input class="p-input" id="a-sembol" placeholder="Sembol" list="h-list2" style="max-width:140px">
      <datalist id="h-list2">{% for h in hisseler %}<option value="{{ h.replace('.IS','') }}">{% endfor %}</datalist>
      <select class="sel" id="a-yon">
        <option value="above">Üstüne çıkınca</option>
        <option value="below">Altına düşünce</option>
      </select>
      <input class="p-input" id="a-fiyat" placeholder="Hedef Fiyat ₺" type="number" style="max-width:150px">
      <button class="btn-add" onclick="alarmEkle()">+ Alarm Kur</button>
    </div>
    <div id="alarm-listesi"><div class="empty-state">Henüz alarm yok.</div></div>
  </div>
</div>
</div>

<script>
let grafikVerisi = {};
let trackData = null;
let currentPeriod = 90;
let guncelFiyatlar = {};

function tabAc(id, btn) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  btn.classList.add('active');
  if (id === 'grafik') grafikGuncelle();
  if (id === 'trackrecord') perfGrafikCiz();
  if (id === 'portfoy') { portfoyGuncelle(); alarmListeGuncelle(); }
}

function veriCek() {
  fetch('/api/veri')
    .then(r => r.json())
    .then(data => {
      grafikVerisi = data.grafik_verisi || {};
      trackData = data.track_record;
      (data.sinyaller || []).forEach(s => {
        guncelFiyatlar[s.sembol.replace('.IS','')] = s.fiyat;
      });
      sayfaGuncelle(data);
      alarmKontrol();
    })
    .catch(e => console.log('Hata:', e));
}

function sayfaGuncelle(data) {
  const borsa = data.borsa_acik;
  const d = document.getElementById('borsa-durum');
  d.textContent = borsa ? '● BORSA AÇIK' : '● BORSA KAPALI';
  d.className = 'badge' + (borsa ? '' : ' kapali');
  const sg = document.getElementById('son-guncelleme');
  if (sg) sg.textContent = data.son_guncelleme ? 'Son: ' + data.son_guncelleme : '';

  if (!data.hazir) return;

  const rejim = (data.piyasa && data.piyasa.rejim) || 'YATAY';
  const re = document.getElementById('rejim-badge');
  re.textContent = ((data.piyasa && data.piyasa.emoji) || '') + ' ' + rejim;
  re.className = 'rejim-badge rejim-' + (rejim.includes('BOGA') ? 'boga' : rejim.includes('AYI') ? 'ayi' : 'yatay');
  document.getElementById('rejim-aciklama').textContent = (data.piyasa && data.piyasa.aciklama) || '';

  const sinyaller = data.sinyaller || [];
  const al = sinyaller.filter(s => s.karar === 'AL').length;
  const sat = sinyaller.filter(s => s.karar === 'SAT').length;
  const bekle = sinyaller.filter(s => s.karar === 'BEKLE').length;
  document.getElementById('sinyal-sayisi').textContent = sinyaller.length;
  document.getElementById('sinyal-ozet').innerHTML =
    '<span class="green">' + al + ' AL</span> / ' +
    '<span class="red">' + sat + ' SAT</span> / ' +
    '<span class="yellow">' + bekle + ' BEKLE</span>';

  if (data.piyasa && data.piyasa.bist_son) {
    const p = data.piyasa;
    const gr = p.bist_getiri_1ay >= 0 ? 'green' : 'red';
    document.getElementById('bist-deger').textContent =
      Number(p.bist_son).toLocaleString('tr-TR', {maximumFractionDigits:0});
    document.getElementById('bist-sub').innerHTML =
      '<span class="' + gr + '">1A: %' + Number(p.bist_getiri_1ay).toFixed(1) + '</span>' +
      ' &nbsp;|&nbsp; USD: ' + Number(p.usdtry).toFixed(2);
  }

  const tr = data.track_record;
  if (tr && tr.tamamlanan > 0) {
    const br = tr.basari >= 55 ? 'green' : tr.basari >= 45 ? 'yellow' : 'red';
    const kr = tr.ort_kar >= 0 ? 'green' : 'red';
    const bd = document.getElementById('basari-deger');
    bd.className = 'stat-value ' + br;
    bd.textContent = '%' + tr.basari;
    document.getElementById('basari-sub').innerHTML =
      'Tamamlanan: ' + tr.tamamlanan + ' &nbsp;|&nbsp; Ort: <span class="' + kr + '">%' +
      (tr.ort_kar > 0 ? '+' : '') + tr.ort_kar + '</span>';
  } else {
    document.getElementById('basari-deger').textContent = '—';
    document.getElementById('basari-sub').textContent = 'Henüz tamamlanan yok';
  }

  sektorGuncelle(sinyaller);
  sinyalTabloGuncelle(sinyaller);
  trackTabGuncelle(tr);
}

const SEKTOR_MAP = {
  AKBNK:'Bankacılık', GARAN:'Bankacılık', YKBNK:'Bankacılık',
  EKGYO:'Gayrimenkul', PGSUS:'Havacılık', THYAO:'Havacılık',
  TCELL:'Telekom', SISE:'Cam & Kimya', FROTO:'Otomotiv',
  EREGL:'Demir & Çelik', ASELS:'Savunma', TUPRS:'Petrol & Enerji'
};
const SEKTOR_ICON = {
  'Bankacılık':'🏦', 'Gayrimenkul':'🏢', 'Havacılık':'✈️',
  'Telekom':'📡', 'Cam & Kimya':'🧪', 'Otomotiv':'🚗',
  'Demir & Çelik':'⚙️', 'Savunma':'🛡️', 'Petrol & Enerji':'⛽'
};

function sektorGuncelle(sinyaller) {
  const s = {};
  sinyaller.forEach(x => {
    const kod = x.sembol.replace('.IS','');
    const sek = SEKTOR_MAP[kod] || 'Diğer';
    if (!s[sek]) s[sek] = {al:0, sat:0, bekle:0, deg:[]};
    if (x.karar==='AL') s[sek].al++;
    else if (x.karar==='SAT') s[sek].sat++;
    else s[sek].bekle++;
    s[sek].deg.push(x.degisim || 0);
  });
  let html = '';
  Object.entries(s).forEach(([sek, b]) => {
    const ort = b.deg.length ? b.deg.reduce((a,v) => a+v, 0) / b.deg.length : 0;
    const r = ort > 0 ? 'green' : ort < 0 ? 'red' : 'yellow';
    const durum = b.al > b.sat ? '🟢 AL ağırlıklı' : b.sat > b.al ? '🔴 SAT ağırlıklı' : '🟡 Karışık';
    html += '<div class="sektor-kart">' +
      '<div class="sektor-icon">' + (SEKTOR_ICON[sek]||'📦') + '</div>' +
      '<div class="sektor-name">' + sek + '</div>' +
      '<div class="sektor-val ' + r + '">' + (ort>=0?'+':'') + ort.toFixed(1) + '%</div>' +
      '<div class="sektor-durum">' + durum + '</div>' +
      '</div>';
  });
  document.getElementById('sektor-grid').innerHTML = html || '<div class="empty-state">Yükleniyor...</div>';
}

function sinyalTabloGuncelle(sinyaller) {
  if (!sinyaller || !sinyaller.length) {
    document.getElementById('sinyal-tablo-alani').innerHTML = '<div class="empty-state">Modeller eğitiliyor...</div>';
    return;
  }
  let h = '<table class="tablo"><thead><tr><th>Hisse</th><th>Fiyat</th><th>Değişim</th><th>RSI</th><th>Karar</th><th>Güven</th><th>Hedef</th><th>Stop</th></tr></thead><tbody>';
  sinyaller.forEach(s => {
    const dr = s.degisim >= 0 ? 'green' : 'red';
    const di = s.degisim >= 0 ? '+' : '';
    const kc = s.karar==='AL' ? 'al' : s.karar==='SAT' ? 'sat' : 'bekle';
    const gp = (s.guven * 100).toFixed(0);
    const rr = s.rsi < 40 ? 'green' : s.rsi > 60 ? 'red' : 'yellow';
    h += '<tr class="' + kc + '-row">' +
      '<td style="font-weight:700">' + s.sembol.replace('.IS','') + '</td>' +
      '<td>' + Number(s.fiyat).toFixed(2) + ' ₺</td>' +
      '<td class="' + dr + '">' + di + Number(s.degisim).toFixed(2) + '%</td>' +
      '<td class="' + rr + '">' + Number(s.rsi).toFixed(1) + '</td>' +
      '<td><span class="pill ' + kc + '">' + s.karar + '</span></td>' +
      '<td>%' + gp + '<div class="guven-bar"><div class="guven-fill" style="width:' + gp + '%"></div></div></td>' +
      '<td class="green">' + (s.hedef ? Number(s.hedef).toFixed(2) + ' ₺' : '—') + '</td>' +
      '<td class="' + (s.karar==='SAT'?'green':'red') + '">' + (s.stop ? Number(s.stop).toFixed(2) + ' ₺' : '—') + '</td>' +
      '</tr>';
  });
  document.getElementById('sinyal-tablo-alani').innerHTML = h + '</tbody></table>';
}

function periodSec(gun, btn) {
  currentPeriod = gun;
  document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  grafikGuncelle();
}

function grafikGuncelle() {
  const hisse = document.getElementById('hisse-sec') && document.getElementById('hisse-sec').value;
  const veri = grafikVerisi[hisse];
  if (!veri || !veri.tarihler) return;
  const n = Math.min(currentPeriod, veri.tarihler.length);
  const sl = arr => (arr || []).slice(-n);
  const tar = sl(veri.tarihler), op = sl(veri.open), hi = sl(veri.high);
  const lo = sl(veri.low), cl = sl(veri.close), m20 = sl(veri.ma20);
  const m50 = sl(veri.ma50), vol = sl(veri.volume), rsi = sl(veri.rsi);

  Plotly.newPlot('grafik-alan', [
    {type:'candlestick', x:tar, open:op, high:hi, low:lo, close:cl, name:hisse.replace('.IS',''),
     increasing:{line:{color:'#22c55e'},fillcolor:'#22c55e22'},
     decreasing:{line:{color:'#ef4444'},fillcolor:'#ef444422'}},
    {type:'scatter', x:tar, y:m20, name:'MA20', line:{color:'#f59e0b',width:1.5}},
    {type:'scatter', x:tar, y:m50, name:'MA50', line:{color:'#8b5cf6',width:1.5}}
  ], {
    paper_bgcolor:'#1e293b', plot_bgcolor:'#1e293b',
    font:{color:'#94a3b8', size:11},
    xaxis:{gridcolor:'#334155', rangeslider:{visible:false}, type:'date'},
    yaxis:{gridcolor:'#334155', ticksuffix:' ₺'},
    margin:{t:8, r:16, b:28, l:64},
    legend:{bgcolor:'#0f172a33', font:{size:11}},
    showlegend:true
  }, {responsive:true, displayModeBar:false});

  if (vol && vol.length) {
    const vc = cl.map((c,i) => c >= (op[i]||c) ? '#22c55e55' : '#ef444455');
    Plotly.newPlot('hacim-alan', [{type:'bar', x:tar, y:vol, marker:{color:vc}, name:'Hacim'}], {
      paper_bgcolor:'#1e293b', plot_bgcolor:'#1e293b',
      font:{color:'#94a3b8', size:10},
      xaxis:{gridcolor:'#334155', type:'date'},
      yaxis:{gridcolor:'#334155', tickformat:'.2s'},
      margin:{t:4, r:16, b:28, l:64}, showlegend:false
    }, {responsive:true, displayModeBar:false});
  }

  if (rsi && rsi.length) {
    const x0 = tar[0], x1 = tar[tar.length-1];
    Plotly.newPlot('rsi-alan', [
      {type:'scatter', x:tar, y:rsi, line:{color:'#3b82f6', width:1.5}, name:'RSI'},
      {type:'scatter', x:[x0,x1], y:[70,70], line:{color:'#ef444455', width:1, dash:'dot'}, showlegend:false},
      {type:'scatter', x:[x0,x1], y:[30,30], line:{color:'#22c55e55', width:1, dash:'dot'}, showlegend:false}
    ], {
      paper_bgcolor:'#1e293b', plot_bgcolor:'#1e293b',
      font:{color:'#94a3b8', size:10},
      xaxis:{gridcolor:'#334155', type:'date'},
      yaxis:{gridcolor:'#334155', range:[0,100], tickvals:[30,50,70]},
      margin:{t:4, r:16, b:28, l:64}, showlegend:false
    }, {responsive:true, displayModeBar:false});
  }
}

function trackTabGuncelle(tr) {
  if (!tr) return;
  const br = tr.basari >= 55 ? 'green' : tr.basari >= 45 ? 'yellow' : 'red';
  const kr = tr.ort_kar >= 0 ? 'green' : 'red';
  document.getElementById('tr-toplam').textContent = tr.toplam;
  document.getElementById('tr-tamamlanan').textContent = tr.tamamlanan;
  const tb = document.getElementById('tr-basari');
  tb.className = 'tr-stat-value ' + br;
  tb.textContent = tr.tamamlanan > 0 ? '%' + tr.basari : '—';
  const tk = document.getElementById('tr-ort-kar');
  tk.className = 'tr-stat-value ' + kr;
  tk.textContent = tr.tamamlanan > 0 ? '%' + (tr.ort_kar>0?'+':'') + tr.ort_kar : '—';

  if (!tr.son_sinyaller || !tr.son_sinyaller.length) {
    document.getElementById('track-record-alani').innerHTML = '<div class="empty-state">Henüz tamamlanan sinyal yok.</div>';
    return;
  }
  let h = '<table class="tablo"><thead><tr><th>Tarih</th><th>Hisse</th><th>Karar</th><th>Giriş</th><th>Hedef</th><th>Stop</th><th>Çıkış</th><th>K/Z</th><th>Sonuç</th></tr></thead><tbody>';
  tr.son_sinyaller.forEach(s => {
    const kzv = s.kar_zarar ? parseFloat(s.kar_zarar) : null;
    const sr = s.sonuc==='KAZANDI' ? 'green' : s.sonuc==='KAYBETTI' ? 'red' : 'yellow';
    const kc = s.karar==='AL' ? 'al' : s.karar==='SAT' ? 'sat' : 'bekle';
    const kzs = kzv != null ? '<span class="' + (kzv>=0?'green':'red') + '">%' + (kzv>0?'+':'') + kzv.toFixed(1) + '</span>' : '—';
    h += '<tr class="' + kc + '-row">' +
      '<td style="font-size:11px;white-space:nowrap">' + (s.zaman||'—') + '</td>' +
      '<td style="font-weight:700">' + (s.sembol||'').replace('.IS','') + '</td>' +
      '<td><span class="pill ' + kc + '">' + s.karar + '</span></td>' +
      '<td>' + (s.fiyat_giris ? parseFloat(s.fiyat_giris).toFixed(2)+' ₺' : '—') + '</td>' +
      '<td class="green">' + (s.hedef ? parseFloat(s.hedef).toFixed(2)+' ₺' : '—') + '</td>' +
      '<td class="' + (s.karar==='SAT'?'green':'red') + '">' + (s.stop ? parseFloat(s.stop).toFixed(2)+' ₺' : '—') + '</td>' +
      '<td>' + (s.fiyat_cikis ? parseFloat(s.fiyat_cikis).toFixed(2)+' ₺' : '—') + '</td>' +
      '<td>' + kzs + '</td>' +
      '<td class="' + sr + '" style="font-weight:600">' + (s.sonuc||'Bekliyor') + '</td>' +
      '</tr>';
  });
  document.getElementById('track-record-alani').innerHTML = h + '</tbody></table>';
}

function perfGrafikCiz() {
  if (!trackData || !trackData.tamamlanan_liste || !trackData.tamamlanan_liste.length) return;
  const liste = trackData.tamamlanan_liste;
  let cum = 0;
  const xs = [], ys = [], txts = [];
  liste.forEach((item, i) => {
    cum += parseFloat(item.kar_zarar || 0);
    xs.push(i + 1);
    ys.push(parseFloat(cum.toFixed(2)));
    txts.push((item.sembol||'').replace('.IS','') + ' ' + (item.sonuc||''));
  });
  const last = ys[ys.length-1] || 0;
  Plotly.newPlot('perf-grafik', [{
    type:'scatter', x:xs, y:ys, mode:'lines+markers',
    line:{color:'#3b82f6', width:2},
    marker:{color:ys.map(v => v>=0?'#22c55e':'#ef4444'), size:5},
    fill:'tozeroy', fillcolor:last>=0?'#22c55e0d':'#ef44440d',
    text:txts,
    hovertemplate:'%{text}<br>Kümülatif: %{y:.1f}%<extra></extra>'
  }], {
    paper_bgcolor:'#1e293b', plot_bgcolor:'#1e293b',
    font:{color:'#94a3b8', size:11},
    xaxis:{gridcolor:'#334155', title:'Sinyal #'},
    yaxis:{gridcolor:'#334155', ticksuffix:'%', zeroline:true, zerolinecolor:'#475569'},
    margin:{t:8, r:16, b:40, l:56}, showlegend:false
  }, {responsive:true, displayModeBar:false});
}

// Portföy
function portfoyEkle() {
  const sembol = (document.getElementById('p-sembol').value||'').toUpperCase().trim();
  const adet = parseFloat(document.getElementById('p-adet').value);
  const maliyet = parseFloat(document.getElementById('p-maliyet').value);
  if (!sembol || !adet || !maliyet) { alert('Tüm alanları doldurun.'); return; }
  const p = JSON.parse(localStorage.getItem('portfoy')||'[]');
  const idx = p.findIndex(x => x.sembol === sembol);
  if (idx >= 0) p[idx] = {sembol,adet,maliyet};
  else p.push({sembol,adet,maliyet});
  localStorage.setItem('portfoy', JSON.stringify(p));
  document.getElementById('p-sembol').value = '';
  document.getElementById('p-adet').value = '';
  document.getElementById('p-maliyet').value = '';
  portfoyGuncelle();
}

function portfoySil(s) {
  const p = JSON.parse(localStorage.getItem('portfoy')||'[]').filter(x => x.sembol !== s);
  localStorage.setItem('portfoy', JSON.stringify(p));
  portfoyGuncelle();
}

function portfoyGuncelle() {
  const p = JSON.parse(localStorage.getItem('portfoy')||'[]');
  const tbl = document.getElementById('portfoy-tablo');
  const ozet = document.getElementById('portfoy-ozet');
  if (!p.length) {
    tbl.innerHTML = '<div class="empty-state">Portföy boş.</div>';
    if (ozet) ozet.style.display = 'none';
    return;
  }
  let h = '<table class="tablo"><thead><tr><th>Hisse</th><th>Adet</th><th>Maliyet</th><th>Güncel</th><th>Piyasa D.</th><th>K/Z</th><th>K/Z %</th><th></th></tr></thead><tbody>';
  let totM = 0, totD = 0;
  p.forEach(x => {
    const gun = guncelFiyatlar[x.sembol] || null;
    const pd = gun !== null ? gun * x.adet : null;
    const md = x.maliyet * x.adet;
    const kz = pd !== null ? pd - md : null;
    const kzp = gun !== null ? (gun - x.maliyet) / x.maliyet * 100 : null;
    const r = kz !== null ? (kz>=0?'green':'red') : '';
    totM += md;
    if (pd !== null) totD += pd;
    h += '<tr>' +
      '<td style="font-weight:700">' + x.sembol + '</td>' +
      '<td>' + x.adet + '</td>' +
      '<td>' + x.maliyet.toFixed(2) + ' ₺</td>' +
      '<td>' + (gun !== null ? gun.toFixed(2)+' ₺' : '<span style="color:#475569">—</span>') + '</td>' +
      '<td>' + (pd !== null ? pd.toLocaleString('tr-TR',{maximumFractionDigits:0})+' ₺' : '—') + '</td>' +
      '<td class="' + r + '">' + (kz !== null ? (kz>=0?'+':'')+kz.toLocaleString('tr-TR',{maximumFractionDigits:0})+' ₺' : '—') + '</td>' +
      '<td class="' + r + '">' + (kzp !== null ? (kzp>=0?'+':'')+kzp.toFixed(1)+'%' : '—') + '</td>' +
      '<td><button class="btn-del" onclick="portfoySil(' + "'" + x.sembol + "'" + ')">Sil</button></td>' +
      '</tr>';
  });
  tbl.innerHTML = h + '</tbody></table>';
  const netKZ = totD - totM;
  const netKZP = totM > 0 ? netKZ / totM * 100 : 0;
  const nr = netKZ >= 0 ? 'green' : 'red';
  if (ozet) {
    ozet.style.display = 'flex';
    ozet.innerHTML =
      '<span>Maliyet: <strong>' + totM.toLocaleString('tr-TR',{maximumFractionDigits:0}) + ' ₺</strong></span>' +
      '<span>Piyasa D.: <strong>' + totD.toLocaleString('tr-TR',{maximumFractionDigits:0}) + ' ₺</strong></span>' +
      '<span>Net K/Z: <strong class="' + nr + '">' + (netKZ>=0?'+':'') + netKZ.toLocaleString('tr-TR',{maximumFractionDigits:0}) + ' ₺ (%' + (netKZP>=0?'+':'') + netKZP.toFixed(1) + ')</strong></span>';
  }
}

// Alarmlar
function alarmEkle() {
  const sembol = (document.getElementById('a-sembol').value||'').toUpperCase().trim();
  const yon = document.getElementById('a-yon').value;
  const fiyat = parseFloat(document.getElementById('a-fiyat').value);
  if (!sembol || !fiyat) { alert('Sembol ve fiyat zorunlu.'); return; }
  const a = JSON.parse(localStorage.getItem('alarmlar')||'[]');
  a.push({sembol, yon, fiyat, tetiklendi:false});
  localStorage.setItem('alarmlar', JSON.stringify(a));
  document.getElementById('a-sembol').value = '';
  document.getElementById('a-fiyat').value = '';
  alarmListeGuncelle();
}

function alarmSil(i) {
  const a = JSON.parse(localStorage.getItem('alarmlar')||'[]');
  a.splice(i, 1);
  localStorage.setItem('alarmlar', JSON.stringify(a));
  alarmListeGuncelle();
}

function alarmKontrol() {
  const a = JSON.parse(localStorage.getItem('alarmlar')||'[]');
  let changed = false;
  a.forEach((alarm, i) => {
    if (alarm.tetiklendi) return;
    const gun = guncelFiyatlar[alarm.sembol];
    if (gun === undefined) return;
    const hit = alarm.yon === 'above' ? gun >= alarm.fiyat : gun <= alarm.fiyat;
    if (hit) {
      a[i].tetiklendi = true;
      changed = true;
      const msg = alarm.sembol + ' ' + gun.toFixed(2) + ' ₺ fiyatına ' + (alarm.yon==='above'?'ulaştı':'indi');
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('Fiyat Alarmı', {body: msg});
      }
    }
  });
  if (changed) { localStorage.setItem('alarmlar', JSON.stringify(a)); alarmListeGuncelle(); }
}

function alarmListeGuncelle() {
  const a = JSON.parse(localStorage.getItem('alarmlar')||'[]');
  const el = document.getElementById('alarm-listesi');
  if (!el) return;
  if (!a.length) { el.innerHTML = '<div class="empty-state">Henüz alarm yok.</div>'; return; }
  let h = '<table class="tablo"><thead><tr><th>Hisse</th><th>Koşul</th><th>Hedef</th><th>Güncel</th><th>Durum</th><th></th></tr></thead><tbody>';
  a.forEach((x, i) => {
    const gun = guncelFiyatlar[x.sembol];
    const durum = x.tetiklendi ?
      '<span class="alarm-badge alarm-hit">Tetiklendi</span>' :
      '<span class="alarm-badge">Bekliyor</span>';
    h += '<tr>' +
      '<td style="font-weight:700">' + x.sembol + '</td>' +
      '<td>' + (x.yon==='above'?'↑ Üstüne':'↓ Altına') + '</td>' +
      '<td>' + x.fiyat.toFixed(2) + ' ₺</td>' +
      '<td>' + (gun !== undefined ? gun.toFixed(2)+' ₺' : '—') + '</td>' +
      '<td>' + durum + '</td>' +
      '<td><button class="btn-del" onclick="alarmSil(' + i + ')">Sil</button></td>' +
      '</tr>';
  });
  el.innerHTML = h + '</tbody></table>';
  if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission();
}

alarmListeGuncelle();
portfoyGuncelle();
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
        if karar == "AL":
            hedef = round(guvenli_sayi(son_fiyat + atr * 2.5), 2)
            stop  = round(guvenli_sayi(son_fiyat - atr * 1.5), 2)
        elif karar == "SAT":
            hedef = round(guvenli_sayi(son_fiyat - atr * 2.5), 2)
            stop  = round(guvenli_sayi(son_fiyat + atr * 1.5), 2)
        else:
            hedef = round(guvenli_sayi(son_fiyat + atr * 2.5), 2)
            stop  = round(guvenli_sayi(son_fiyat - atr * 1.5), 2)
        return {
    'sembol' : sembol,
    'fiyat'  : round(guvenli_sayi(son_fiyat), 2),
    'degisim': round(guvenli_sayi(degisim), 2),
    'rsi'    : round(guvenli_sayi(df['RSI'].iloc[-1]), 1),
    'karar'  : karar,
    'guven'  : round(guvenli_sayi(guven), 3),
    'hedef'  : hedef,
    'stop'   : stop,
}
    except:
        return None

def grafik_verisi_hazirla(sembol, df):
    s = df.tail(252).copy()
    rsi_vals = ta.rsi(s['Close'], length=14)
    return {
        'tarihler': [str(t)[:10] for t in s.index],
        'open'   : s['Open'].round(2).tolist(),
        'high'   : s['High'].round(2).tolist(),
        'low'    : s['Low'].round(2).tolist(),
        'close'  : s['Close'].round(2).tolist(),
        'ma20'   : s['MA20'].round(2).tolist(),
        'ma50'   : s['MA50'].round(2).tolist(),
        'volume' : s['Volume'].tolist(),
        'rsi'    : rsi_vals.round(1).tolist(),
    }

def track_record_oku():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "track_record.csv"), encoding='utf-8-sig')
        df['_dt'] = pd.to_datetime(df['zaman'], dayfirst=True)
        tamamlanan = df[df['sonuc'].isin(['KAZANDI','KAYBETTI'])].copy()
        son_sinyaller = (df.sort_values('_dt')
                           .groupby('sembol', as_index=False).last()
                           .sort_values('_dt', ascending=False)
                           .drop(columns=['_dt'])
                           .to_dict('records'))
        if len(tamamlanan) == 0:
            return {'toplam':len(df),'tamamlanan':0,'kazanan':0,
                    'basari':0,'ort_kar':0,'tamamlanan_liste':[],
                    'son_sinyaller':son_sinyaller}
        kazanan = len(tamamlanan[tamamlanan['sonuc']=='KAZANDI'])
        basari  = kazanan / len(tamamlanan) * 100
        ort_kar = tamamlanan['kar_zarar'].astype(float).mean()
        tamamlanan_liste = (tamamlanan.sort_values('_dt')
                              [['sembol','kar_zarar','sonuc']]
                              .to_dict('records'))
        return {
            'toplam'           : len(df),
            'tamamlanan'       : len(tamamlanan),
            'kazanan'          : kazanan,
            'basari'           : round(basari, 1),
            'ort_kar'          : round(float(ort_kar), 2),
            'son_sinyaller'    : son_sinyaller,
            'tamamlanan_liste' : tamamlanan_liste,
        }
    except:
        return {'toplam':0,'tamamlanan':0,'kazanan':0,
                'basari':0,'ort_kar':0,'son_sinyaller':[],'tamamlanan_liste':[]}

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
                try:
                    df_c = yf.Ticker(s).history(period='1y', interval='1d')
                    df_c = df_c[['Open','High','Low','Close','Volume']]
                    df_c.index = df_c.index.tz_localize(None)
                    df_c['MA20'] = ta.sma(df_c['Close'], length=20)
                    df_c['MA50'] = ta.sma(df_c['Close'], length=50)
                    grafik_v[s] = grafik_verisi_hazirla(s, df_c)
                except:
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