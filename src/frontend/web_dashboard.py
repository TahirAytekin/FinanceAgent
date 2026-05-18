from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import math
import os
import pickle
import zlib
import threading
import time
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
try:
    import feedparser as _feedparser
except ImportError:
    _feedparser = None
try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_OK = True
except ImportError:
    _PSYCOPG2_OK = False

# ─── Ayarlar ───────────────────────────────────────────
HISSELER   = ["AKBNK.IS", "GARAN.IS", "YKBNK.IS",
              "EKGYO.IS", "PGSUS.IS", "TCELL.IS",
              "SISE.IS",  "FROTO.IS"]
GUNCELLEME = 300
PORT       = int(os.environ.get('PORT', 5000))
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ───────────────────────────────────────────────────────

# ─── Veritabani ────────────────────────────────────────
DB_URL = os.environ.get('DATABASE_URL')

def db_baglan():
    if not DB_URL or not _PSYCOPG2_OK:
        return None
    try:
        return psycopg2.connect(DB_URL, connect_timeout=5)
    except Exception as e:
        print(f"[DB] Baglanti hatasi: {e}")
        return None

def db_tablolari_olustur():
    conn = db_baglan()
    if conn is None:
        print("[DB] Veritabani yok - bellek modu aktif.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS lidya_modeller (
                sembol VARCHAR(20) PRIMARY KEY,
                model_data BYTEA NOT NULL,
                scaler_data BYTEA NOT NULL,
                ozellik_sayisi INTEGER,
                egitim_tarihi TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS lidya_sinyaller (
                id BIGSERIAL PRIMARY KEY,
                sembol VARCHAR(20), fiyat FLOAT, degisim FLOAT,
                rsi FLOAT, karar VARCHAR(10), guven FLOAT,
                hedef FLOAT, stop FLOAT,
                zaman TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS lidya_track_record (
                id BIGSERIAL PRIMARY KEY,
                sembol VARCHAR(20), karar VARCHAR(10),
                fiyat_giris FLOAT, fiyat_cikis FLOAT,
                hedef FLOAT, stop FLOAT, kar_zarar FLOAT,
                sonuc VARCHAR(20) DEFAULT 'Bekliyor',
                zaman TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS lidya_portfoy (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(64), sembol VARCHAR(20),
                adet FLOAT, maliyet FLOAT,
                guncelleme TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(session_id, sembol)
            );
            CREATE TABLE IF NOT EXISTS lidya_alarmlar (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(64), sembol VARCHAR(20),
                yon VARCHAR(10), fiyat FLOAT,
                tetiklendi BOOLEAN DEFAULT FALSE,
                olusturma TIMESTAMPTZ DEFAULT NOW()
            );
            """)
        conn.commit()
        print("[DB] Tablolar hazir.")
    except Exception as e:
        print(f"[DB] Tablo olusturma hatasi: {e}")
    finally:
        conn.close()

def model_db_kaydet(sembol, model, scaler):
    conn = db_baglan()
    if conn is None:
        return
    try:
        mb = zlib.compress(pickle.dumps(model), level=6)
        sb = zlib.compress(pickle.dumps(scaler), level=6)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lidya_modeller
                    (sembol, model_data, scaler_data, ozellik_sayisi, egitim_tarihi)
                VALUES (%s,%s,%s,%s,NOW())
                ON CONFLICT (sembol) DO UPDATE SET
                    model_data=EXCLUDED.model_data,
                    scaler_data=EXCLUDED.scaler_data,
                    ozellik_sayisi=EXCLUDED.ozellik_sayisi,
                    egitim_tarihi=NOW()
            """, (sembol, psycopg2.Binary(mb), psycopg2.Binary(sb), len(OZELLIKLER)))
        conn.commit()
        print(f"[DB] {sembol} modeli kaydedildi.")
    except Exception as e:
        print(f"[DB] Model kayit hatasi {sembol}: {e}")
    finally:
        conn.close()

def model_db_yukle(sembol, max_gun=7):
    conn = db_baglan()
    if conn is None:
        return None, None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_data, scaler_data, ozellik_sayisi, egitim_tarihi
                FROM lidya_modeller WHERE sembol=%s
            """, (sembol,))
            row = cur.fetchone()
        if row is None:
            return None, None
        model_data, scaler_data, ozellik_sayisi, egitim_tarihi = row
        yas = (datetime.now(timezone.utc) - egitim_tarihi.replace(tzinfo=timezone.utc)
               if egitim_tarihi.tzinfo is None
               else datetime.now(timezone.utc) - egitim_tarihi).days
        if yas > max_gun:
            print(f"[DB] {sembol} modeli eski ({yas} gun), yeniden egitilecek.")
            return None, None
        if ozellik_sayisi != len(OZELLIKLER):
            print(f"[DB] {sembol} ozellik sayisi uyusmuyor ({ozellik_sayisi} vs {len(OZELLIKLER)}), yeniden egitilecek.")
            return None, None
        model  = pickle.loads(zlib.decompress(bytes(model_data)))
        scaler = pickle.loads(zlib.decompress(bytes(scaler_data)))
        print(f"[DB] {sembol} modeli yuklendi ({yas} gun onceki egitim) ✅")
        return model, scaler
    except Exception as e:
        print(f"[DB] Model yukleme hatasi {sembol}: {e}")
        return None, None
    finally:
        conn.close()

def sinyalleri_db_kaydet(sinyaller):
    conn = db_baglan()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            for s in sinyaller:
                cur.execute("""
                    INSERT INTO lidya_sinyaller
                        (sembol,fiyat,degisim,rsi,karar,guven,hedef,stop)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (s['sembol'],s.get('fiyat'),s.get('degisim'),s.get('rsi'),
                      s.get('karar'),s.get('guven'),s.get('hedef'),s.get('stop')))
        conn.commit()
    except Exception as e:
        print(f"[DB] Sinyal kayit hatasi: {e}")
    finally:
        conn.close()

def track_record_db_ekle(sinyaller):
    """AL/SAT sinyallerini lidya_track_record'a Bekliyor olarak ekler (24 saatte bir tekrar)."""
    conn = db_baglan()
    if conn is None:
        return
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            for s in sinyaller:
                if s.get('karar') not in ('AL', 'SAT'):
                    continue
                cur.execute("""
                    SELECT id FROM lidya_track_record
                    WHERE sembol=%s AND karar=%s AND sonuc='Bekliyor'
                    AND zaman > NOW() - INTERVAL '24 hours'
                    LIMIT 1
                """, (s['sembol'], s['karar']))
                if cur.fetchone():
                    continue
                cur.execute("""
                    INSERT INTO lidya_track_record
                        (sembol, karar, fiyat_giris, hedef, stop, sonuc)
                    VALUES (%s,%s,%s,%s,%s,'Bekliyor')
                """, (s['sembol'], s['karar'],
                      s.get('fiyat'), s.get('hedef'), s.get('stop')))
        conn.commit()
    except Exception as e:
        print(f"[DB] Track record ekleme hatasi: {e}")
    finally:
        conn.close()

def track_record_db_oku():
    conn = db_baglan()
    if conn is None:
        return None
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT sembol, karar, fiyat_giris, fiyat_cikis,
                       hedef, stop, kar_zarar, sonuc,
                       TO_CHAR(zaman,'DD.MM.YYYY HH24:MI') as zaman
                FROM lidya_track_record ORDER BY zaman DESC
            """)
            rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return None
        tamamlanan = [r for r in rows if r['sonuc'] in ('KAZANDI','KAYBETTI')]
        kazanan = sum(1 for r in tamamlanan if r['sonuc']=='KAZANDI')
        basari  = kazanan/len(tamamlanan)*100 if tamamlanan else 0
        ort_kar = (sum(float(r['kar_zarar'] or 0) for r in tamamlanan)/len(tamamlanan)
                   if tamamlanan else 0)
        return {
            'toplam':len(rows),'tamamlanan':len(tamamlanan),'kazanan':kazanan,
            'basari':round(basari,1),'ort_kar':round(ort_kar,2),
            'son_sinyaller':rows[:20],
            'tamamlanan_liste':[{'sembol':r['sembol'],'kar_zarar':r['kar_zarar'],
                                  'sonuc':r['sonuc']} for r in tamamlanan],
        }
    except Exception as e:
        print(f"[DB] Track record okuma hatasi: {e}")
        return None
    finally:
        conn.close()

def sinyallerden_track_record_guncelle():
    conn = db_baglan()
    if conn is None:
        return
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, sembol, karar, fiyat_giris
                FROM lidya_track_record
                WHERE sonuc='Bekliyor' AND zaman < NOW() - INTERVAL '3 days'
            """)
            bekleyenler = cur.fetchall()
            for b in bekleyenler:
                cur.execute("""
                    SELECT fiyat FROM lidya_sinyaller
                    WHERE sembol=%s ORDER BY zaman DESC LIMIT 1
                """, (b['sembol'],))
                son = cur.fetchone()
                if not son or not b['fiyat_giris']:
                    continue
                cikis = float(son['fiyat'])
                giris = float(b['fiyat_giris'])
                if giris == 0:
                    continue
                kz = (cikis - giris) / giris * 100
                if b['karar'] == 'SAT':
                    kz = -kz
                sonuc = 'KAZANDI' if kz > 0 else 'KAYBETTI'
                cur.execute("""
                    UPDATE lidya_track_record
                    SET fiyat_cikis=%s, kar_zarar=%s, sonuc=%s WHERE id=%s
                """, (cikis, round(kz,2), sonuc, b['id']))
        conn.commit()
    except Exception as e:
        print(f"[DB] Track record guncelleme hatasi: {e}")
    finally:
        conn.close()

def portfoy_db_oku(session_id):
    conn = db_baglan()
    if conn is None:
        return None
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT sembol,adet,maliyet FROM lidya_portfoy WHERE session_id=%s",
                        (session_id,))
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        return None
    finally:
        conn.close()

def portfoy_db_kaydet(session_id, sembol, adet, maliyet):
    conn = db_baglan()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lidya_portfoy (session_id,sembol,adet,maliyet,guncelleme)
                VALUES (%s,%s,%s,%s,NOW())
                ON CONFLICT (session_id,sembol) DO UPDATE SET
                    adet=EXCLUDED.adet, maliyet=EXCLUDED.maliyet, guncelleme=NOW()
            """, (session_id, sembol, adet, maliyet))
        conn.commit()
        return True
    except Exception as e:
        return False
    finally:
        conn.close()

def portfoy_db_sil(session_id, sembol):
    conn = db_baglan()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM lidya_portfoy WHERE session_id=%s AND sembol=%s",
                        (session_id, sembol))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def alarmlar_db_oku(session_id):
    conn = db_baglan()
    if conn is None:
        return None
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""SELECT id,sembol,yon,fiyat,tetiklendi FROM lidya_alarmlar
                           WHERE session_id=%s ORDER BY olusturma DESC""", (session_id,))
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        return None
    finally:
        conn.close()

def alarm_db_ekle(session_id, sembol, yon, fiyat):
    conn = db_baglan()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO lidya_alarmlar (session_id,sembol,yon,fiyat)
                           VALUES (%s,%s,%s,%s) RETURNING id""",
                        (session_id, sembol, yon, fiyat))
            new_id = cur.fetchone()[0]
        conn.commit()
        return new_id
    except Exception:
        return None
    finally:
        conn.close()

def alarm_db_sil(session_id, alarm_id):
    conn = db_baglan()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM lidya_alarmlar WHERE id=%s AND session_id=%s",
                        (alarm_id, session_id))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def alarmlar_db_kontrol(sinyaller):
    conn = db_baglan()
    if conn is None:
        return
    try:
        fm = {s['sembol'].replace('.IS',''): s['fiyat'] for s in sinyaller}
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id,sembol,yon,fiyat FROM lidya_alarmlar WHERE tetiklendi=FALSE")
            for alarm in cur.fetchall():
                g = fm.get(alarm['sembol'])
                if g is None:
                    continue
                hit = ((alarm['yon']=='above' and g>=alarm['fiyat']) or
                       (alarm['yon']=='below' and g<=alarm['fiyat']))
                if hit:
                    cur.execute("UPDATE lidya_alarmlar SET tetiklendi=TRUE WHERE id=%s",
                                (alarm['id'],))
        conn.commit()
    except Exception as e:
        print(f"[DB] Alarm kontrol hatasi: {e}")
    finally:
        conn.close()
# ───────────────────────────────────────────────────────

app = Flask(__name__)

SISTEM_VERISI = {
    'sinyaller'     : [],
    'piyasa'        : {},
    'son_guncelleme': None,
    'modeller'      : {},
    'grafik_verisi' : {},
    'hazir'         : False,
    'kenar'         : {'kripto':[], 'doviz':[], 'emtia':[]},
    'haberler'      : [],
}

HTML = '''<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LIDYA — Borsa Analiz</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root{--bg:#0c0a17;--sf:#110e1f;--card:#17142b;--bd:#2e2a48;--tx:#ddd6f3;--mu:#6b6488;--ac:#a78bfa;--gr:#22c55e;--re:#ef4444;--ye:#f59e0b;--ch:#07050e;--cg:#1c1830;--cf:#5e5a7a;}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;}
.hdr{background:var(--sf);border-bottom:1px solid var(--bd);padding:11px 20px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:200;}
.brand{font-family:'Cinzel',serif;font-size:22px;font-weight:900;letter-spacing:5px;text-transform:uppercase;background:linear-gradient(90deg,#ede9ff 0%,#a78bfa 55%,#ec4899 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 0 18px rgba(167,139,250,.45));}
.brand-sub{font-size:9px;color:var(--mu);letter-spacing:2px;text-transform:uppercase;margin-top:2px;}
.badge{background:#22c55e14;border:1px solid #22c55e28;color:var(--gr);padding:3px 10px;border-radius:20px;font-size:11px;}
.badge.kapali{background:#ef444414;border-color:#ef444428;color:var(--re);}
.tnav{background:var(--sf);border-bottom:1px solid var(--bd);display:flex;padding:0 20px;overflow-x:auto;position:sticky;top:52px;z-index:100;}
.tnav::-webkit-scrollbar{display:none;}
.tb{padding:11px 18px;border:none;background:none;color:var(--mu);cursor:pointer;font-size:13px;font-weight:500;border-bottom:2px solid transparent;white-space:nowrap;transition:all .2s;}
.tb:hover{color:var(--tx);}
.tb.active{color:var(--ac);border-bottom-color:var(--ac);}
.tp{display:none;} .tp.active{display:block;}
.lay{display:flex;min-height:calc(100vh - 88px);}
.sb{width:228px;min-width:228px;background:var(--sf);overflow-y:auto;height:calc(100vh - 88px);position:sticky;top:88px;flex-shrink:0;}
.sb-l{border-right:1px solid var(--bd);}
.sb-r{border-left:1px solid var(--bd);}
.sb::-webkit-scrollbar{width:3px;}.sb::-webkit-scrollbar-thumb{background:var(--bd);}
.main{flex:1;min-width:0;}
.sbt{font-size:11px;font-weight:700;color:var(--tx);text-transform:uppercase;letter-spacing:.08em;padding:11px 12px 7px;border-bottom:1px solid var(--bd);}
.sbi{padding:8px 12px;border-bottom:1px solid rgba(46,42,72,.35);transition:background .15s;}
.sbi:hover{background:var(--card);}
.prow{display:flex;justify-content:space-between;align-items:center;}
.pname{font-size:11px;color:var(--mu);}
.pval{font-size:13px;font-weight:600;color:var(--tx);}
.pchg{font-size:11px;margin-top:1px;text-align:right;}
.na{text-decoration:none;color:var(--tx);display:block;}
.na:hover .nt{color:var(--ac);}
.nt{font-size:11px;line-height:1.4;margin-bottom:2px;transition:color .2s;}
.ns{font-size:9px;color:var(--mu);}
.sbnote{padding:9px 12px;font-size:9px;color:var(--mu);border-top:1px solid var(--bd);}
.con{padding:14px 18px;}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px;margin-bottom:14px;}
.ctit{font-size:12px;font-weight:700;color:var(--tx);text-transform:uppercase;letter-spacing:.07em;margin-bottom:11px;}
.sg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px;}
.sc{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:13px;}
.sh{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:7px;}
.sl{font-size:11px;font-weight:700;color:var(--tx);text-transform:uppercase;letter-spacing:.04em;}
.sv{font-size:22px;font-weight:700;color:var(--tx);line-height:1.1;}
.ss{font-size:11px;color:var(--mu);margin-top:4px;}
.rb{display:inline-block;padding:4px 11px;border-radius:6px;font-weight:600;font-size:13px;}
.rb-b{background:#22c55e14;color:var(--gr);}
.rb-a{background:#ef444414;color:var(--re);}
.rb-y{background:#f59e0b14;color:var(--ye);}
.pill{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600;}
.pill.al{background:#22c55e14;color:var(--gr);border:1px solid #22c55e28;}
.pill.sat{background:#ef444414;color:var(--re);border:1px solid #ef444428;}
.pill.bekle{background:#f59e0b14;color:var(--ye);border:1px solid #f59e0b28;}
.ekgd{display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:8px;}
.ek{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:11px;text-align:center;transition:border-color .2s;}
.ek:hover{border-color:rgba(167,139,250,.35);}
.eik{font-size:19px;margin-bottom:5px;}
.enm{font-size:11px;color:var(--mu);margin-bottom:3px;}
.ev{font-size:15px;font-weight:700;}
.ed{font-size:10px;color:var(--mu);margin-top:2px;}
.t{width:100%;border-collapse:collapse;font-size:12px;}
.t th{text-align:left;padding:7px 9px;color:var(--mu);font-size:10px;text-transform:uppercase;border-bottom:1px solid var(--bd);white-space:nowrap;}
.t td{padding:8px 9px;border-bottom:1px solid rgba(46,42,72,.3);}
.t tr.al-r td{background:rgba(34,197,94,.04);}
.t tr.sat-r td{background:rgba(239,68,68,.04);}
.t tr.bk-r td{background:rgba(245,158,11,.04);}
.t tr:hover td{background:var(--sf);}
.gb{background:var(--bd);border-radius:3px;height:3px;margin-top:4px;width:54px;}
.gf{height:100%;border-radius:3px;background:linear-gradient(90deg,#6366f1,var(--ac));}
.cc{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-bottom:11px;}
.sel{background:var(--sf);color:var(--tx);border:1px solid var(--bd);padding:5px 10px;border-radius:6px;font-size:12px;outline:none;}
.pb{background:var(--sf);border:1px solid var(--bd);color:var(--mu);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px;transition:all .2s;}
.pb.active{background:rgba(167,139,250,.12);border-color:var(--ac);color:var(--ac);}
.pb:hover{border-color:rgba(167,139,250,.4);}
.trg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px;}
.trs{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px;text-align:center;}
.trl{font-size:11px;font-weight:700;color:var(--tx);text-transform:uppercase;margin-bottom:6px;}
.trv{font-size:20px;font-weight:700;}
.pf{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:12px;}
.pi{background:var(--sf);border:1px solid var(--bd);color:var(--tx);padding:6px 10px;border-radius:7px;font-size:12px;outline:none;flex:1;min-width:100px;}
.pi:focus{border-color:var(--ac);}
.ba{background:var(--ac);color:#0c0a17;border:none;padding:6px 15px;border-radius:7px;cursor:pointer;font-size:12px;font-weight:700;white-space:nowrap;}
.ba:hover{opacity:.9;}
.bd2{background:var(--sf);color:var(--re);border:1px solid rgba(239,68,68,.2);padding:3px 8px;border-radius:5px;cursor:pointer;font-size:11px;}
.alb{display:inline-block;padding:2px 7px;border-radius:12px;font-size:10px;font-weight:600;background:rgba(245,158,11,.12);color:var(--ye);border:1px solid rgba(245,158,11,.2);}
.alh{background:rgba(34,197,94,.12);color:var(--gr);border-color:rgba(34,197,94,.2);}
.es{text-align:center;padding:22px;color:var(--mu);font-size:12px;}
.gr{color:var(--gr);}.re{color:var(--re);}.ye{color:var(--ye);}
.sirket-kart{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:14px;text-align:center;cursor:pointer;transition:all .2s;}
.sirket-kart:hover{border-color:rgba(167,139,250,.5);transform:translateY(-2px);}
.sk-logo{width:44px;height:44px;border-radius:11px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:14px;margin:0 auto 8px;}
.sk-ad{font-size:12px;font-weight:700;color:var(--tx);margin-bottom:6px;}
.modal-bg{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(7,5,14,.88);z-index:500;display:flex;align-items:flex-start;justify-content:center;overflow-y:auto;padding:30px 15px;}
.modal-kart{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:22px;width:100%;max-width:680px;margin:auto;}
.met-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:14px 0;}
.met{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:10px;text-align:center;}
.met-l{font-size:10px;color:var(--mu);margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em;}
.met-v{font-size:13px;font-weight:700;color:var(--tx);}
@media(max-width:1100px){.sb{width:195px;min-width:195px;}}
@media(max-width:880px){.sb{display:none;}.sg,.trg{grid-template-columns:repeat(2,1fr);}.ekgd{grid-template-columns:repeat(2,1fr);}}
@media(max-width:560px){.con{padding:10px;}.t th:nth-child(n+6),.t td:nth-child(n+6){display:none;}.met-grid{grid-template-columns:repeat(2,1fr);}}
.mob-menu-btn{display:none;background:none;border:1px solid var(--bd);color:var(--tx);padding:5px 10px;border-radius:7px;cursor:pointer;font-size:17px;line-height:1;}
.bnav{display:none;position:fixed;bottom:0;left:0;right:0;height:57px;background:var(--sf);border-top:1px solid var(--bd);z-index:300;align-items:stretch;}
.bni{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;border:none;background:none;color:var(--mu);cursor:pointer;font-size:9px;font-weight:600;padding:5px 0;transition:color .2s;}
.bni.active{color:var(--ac);}.bni svg{width:18px;height:18px;stroke:currentColor;fill:none;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round;}
.drawer-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(7,5,14,.72);z-index:400;display:none;}
.drawer-overlay.open{display:block;}
.drawer{position:fixed;top:0;left:-285px;width:275px;height:100%;background:var(--sf);z-index:401;transition:left .28s cubic-bezier(.4,0,.2,1);overflow-y:auto;border-right:1px solid var(--bd);}
.drawer.open{left:0;}
.drawer-hdr{padding:13px 14px;border-bottom:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:var(--sf);}
.drawer-kapat{background:none;border:none;color:var(--mu);font-size:22px;cursor:pointer;line-height:1;}
.pf-grafik-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}
.ind-bar{display:flex;gap:5px;flex-wrap:wrap;padding:8px 0 10px;border-bottom:1px solid var(--bd);margin-bottom:10px;align-items:center;}
.ind-btn{padding:3px 11px;border-radius:5px;border:1px solid var(--bd);background:none;color:var(--mu);font-size:11px;cursor:pointer;transition:all .18s;}
.ind-btn:hover{border-color:var(--ac);color:var(--tx);}
.ind-btn.active{background:rgba(167,139,250,.14);border-color:var(--ac);color:var(--ac);}
.ind-sep{width:1px;height:16px;background:var(--bd);margin:0 3px;}
.cizim-aktif{background:rgba(167,139,250,.28)!important;border-color:var(--ac)!important;color:var(--ac)!important;}
@media(max-width:640px){
  .mob-menu-btn{display:flex!important;align-items:center;justify-content:center;}
  .tnav{display:none!important;}
  .bnav{display:flex!important;}
  .lay{padding-bottom:60px;}
  .pf-grafik-grid{grid-template-columns:1fr;}
  #sirket-kartlar{grid-template-columns:repeat(2,1fr)!important;}
  .sg{grid-template-columns:repeat(2,1fr);}
  .trg{grid-template-columns:repeat(2,1fr);}
}
</style>
</head>
<body>

<div class="hdr">
  <div style="display:flex;align-items:center;gap:13px">
    <svg width="40" height="40" viewBox="0 0 42 42" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="bg1" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0%" stop-color="#a78bfa"/>
          <stop offset="100%" stop-color="#ec4899"/>
        </linearGradient>
      </defs>
      <rect x="1"  y="27" width="11" height="14" rx="3" fill="url(#bg1)" opacity="0.55"/>
      <rect x="16" y="18" width="11" height="23" rx="3" fill="url(#bg1)" opacity="0.78"/>
      <rect x="31" y="7"  width="11" height="34" rx="3" fill="url(#bg1)"/>
    </svg>
    <div>
      <div class="brand">LIDYA</div>
      <div class="brand-sub">Borsa Analiz Platformu</div>
    </div>
  </div>
  <div style="display:flex;gap:10px;align-items:center">
    <button class="mob-menu-btn" onclick="drawerAc()" title="Haberler">&#9776;</button>
    <span id="borsa-durum" class="badge">● YÜKLENIYOR</span>
    <span id="son-guncelleme" style="font-size:10px;color:var(--mu)"></span>
  </div>
</div>

<nav class="tnav">
  <button class="tb active" onclick="tabAc('sinyaller',this)">Sinyaller</button>
  <button class="tb" onclick="tabAc('grafik',this)">Grafik</button>
  <button class="tb" onclick="tabAc('trackrecord',this)">Track Record</button>
  <button class="tb" onclick="tabAc('portfoy',this)">Portföy</button>
</nav>

<div class="lay">

<!-- LEFT: Haberler -->
<aside class="sb sb-l">
  <div class="sbt">Güncel Haberler</div>
  <div id="haber-listesi"><div class="es">Yükleniyor...</div></div>
</aside>

<!-- CENTER -->
<main class="main">

<!-- TAB: Sinyaller -->
<div id="tab-sinyaller" class="tp active">
<div class="con">
  <div class="sg">
    <div class="sc">
      <div class="sh"><div class="sl">Piyasa Rejimi</div></div>
      <div id="rejim-badge" class="rb rb-y">—</div>
      <div class="ss" id="rejim-aciklama">Yükleniyor...</div>
    </div>
    <div class="sc">
      <div class="sh"><div class="sl">Aktif Sinyaller</div></div>
      <div class="sv" id="sinyal-sayisi">—</div>
      <div class="ss" id="sinyal-ozet">AL / SAT / BEKLE</div>
    </div>
    <div class="sc">
      <div class="sh"><div class="sl">BIST100</div></div>
      <div class="sv" id="bist-deger">—</div>
      <div class="ss" id="bist-sub">USD/TRY: —</div>
    </div>
    <div class="sc">
      <div class="sh"><div class="sl">Başarı Oranı</div></div>
      <div class="sv" id="basari-deger">—</div>
      <div class="ss" id="basari-sub">Tamamlanan: —</div>
    </div>
  </div>
  <div class="card">
    <div class="ctit">Sektör Analizi</div>
    <div class="ekgd" id="sektor-grid"><div class="es">Yükleniyor...</div></div>
  </div>
  <div class="card">
    <div class="ctit">Hisse Sinyalleri</div>
    <div id="sinyal-tablo-alani"><div class="es">Modeller eğitiliyor...</div></div>
  </div>
  <div class="card">
    <div class="ctit">Sirket Profilleri</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px" id="sirket-kartlar"><div class="es">Sinyaller yukleniyor...</div></div>
  </div>
</div>
</div>

<!-- TAB: Grafik -->
<div id="tab-grafik" class="tp">
<div class="con">
  <div class="card">
    <div class="cc">
      <select id="hisse-sec" class="sel" onchange="grafikGuncelle()">
        {% for h in hisseler %}
        <option value="{{ h }}">{{ h.replace('.IS','') }}</option>
        {% endfor %}
      </select>
      <button class="pb" onclick="periodSec(30,this)">1A</button>
      <button class="pb active" onclick="periodSec(90,this)">3A</button>
      <button class="pb" onclick="periodSec(180,this)">6A</button>
      <button class="pb" onclick="periodSec(252,this)">1Y</button>
    </div>
    <div class="ind-bar">
      <button class="ind-btn" id="ind-ma200" onclick="indToggle('ma200',this)">MA200</button>
      <button class="ind-btn" id="ind-bb" onclick="indToggle('bb',this)">Bollinger</button>
      <button class="ind-btn" id="ind-macd" onclick="indToggle('macd',this)">MACD</button>
      <button class="ind-btn" id="ind-stoch" onclick="indToggle('stoch',this)">Stoch</button>
      <button class="ind-btn" id="ind-pivotlar" onclick="indToggle('pivotlar',this)">S/R</button>
      <button class="ind-btn active" id="ind-sinyaller" onclick="indToggle('sinyaller',this)">Sinyaller</button>
      <div class="ind-sep"></div>
      <button class="ind-btn" id="btn-h-isin" onclick="aracSec('h-isin',this)" title="Yatay ışın — destek/direnç çizgisi">— H. Işın</button>
      <button class="ind-btn" id="btn-v-isin" onclick="aracSec('v-isin',this)" title="Dikey ışın — tarih çizgisi">| V. Işın</button>
      <button class="ind-btn" onclick="isinTemizle()" title="Tüm çizilen ışınları temizle" style="color:var(--re);border-color:rgba(239,68,68,.3)">✕ Temizle</button>
      <div class="ind-sep"></div>
      <button class="ind-btn" id="btn-cizim" onclick="cizimToggle(this)" title="Serbest çizim (trend çizgisi, dikdörtgen, daire)">✏ Serbest</button>
    </div>
    <div id="grafik-alan" style="height:370px"></div>
    <div id="hacim-alan" style="height:85px;margin-top:2px"></div>
    <div id="rsi-alan" style="height:85px;margin-top:2px"></div>
    <div id="macd-alan" style="display:none;height:85px;margin-top:2px"></div>
    <div id="stoch-alan" style="display:none;height:85px;margin-top:2px"></div>
  </div>
</div>
</div>

<!-- TAB: Track Record -->
<div id="tab-trackrecord" class="tp">
<div class="con">
  <div class="trg">
    <div class="trs"><div class="trl">Toplam Sinyal</div><div class="trv" id="tr-toplam">—</div></div>
    <div class="trs"><div class="trl">Tamamlanan</div><div class="trv" id="tr-tamamlanan">—</div></div>
    <div class="trs"><div class="trl">Başarı Oranı</div><div class="trv" id="tr-basari">—</div></div>
    <div class="trs"><div class="trl">Ort. Kar/Zarar</div><div class="trv" id="tr-ort-kar">—</div></div>
  </div>
  <div class="card">
    <div class="ctit">Kümülatif Performans</div>
    <div id="perf-grafik" style="height:220px"></div>
  </div>
  <div class="card">
    <div class="ctit">Sinyal Geçmişi</div>
    <div id="track-record-alani"><div class="es">Yükleniyor...</div></div>
  </div>
</div>
</div>

<!-- TAB: Portföy -->
<div id="tab-portfoy" class="tp">
<div class="con">
  <div class="card">
    <div class="ctit">Portföy Takibi</div>
    <div class="pf">
      <input class="pi" id="p-sembol" placeholder="Sembol (örn: AKBNK)" list="hl" style="max-width:140px">
      <datalist id="hl">{% for h in hisseler %}<option value="{{ h.replace('.IS','') }}">{% endfor %}</datalist>
      <input class="pi" id="p-adet" placeholder="Adet" type="number" style="max-width:90px">
      <input class="pi" id="p-maliyet" placeholder="Ort. Maliyet ₺" type="number" style="max-width:145px">
      <button class="ba" onclick="portfoyEkle()">+ Ekle</button>
    </div>
    <div id="portfoy-tablo"><div class="es">Portföy boş.</div></div>
    <div id="portfoy-ozet" style="display:none;margin-top:12px;padding:12px;background:var(--sf);border-radius:8px;font-size:12px;display:flex;gap:18px;flex-wrap:wrap"></div>
  </div>
  <div class="pf-grafik-grid" id="portfoy-grafik-satir" style="display:none">
    <div class="card" style="margin:0">
      <div class="ctit">Hisse Dağılımı</div>
      <div id="portfoy-pasta" style="height:230px"></div>
    </div>
    <div class="card" style="margin:0">
      <div class="ctit">Kar / Zarar Durumu</div>
      <div id="portfoy-bar" style="height:230px"></div>
    </div>
  </div>
  <div class="card">
    <div class="ctit">Fiyat Alarmları</div>
    <div class="pf">
      <input class="pi" id="a-sembol" placeholder="Sembol" list="hl2" style="max-width:140px">
      <datalist id="hl2">{% for h in hisseler %}<option value="{{ h.replace('.IS','') }}">{% endfor %}</datalist>
      <select class="sel" id="a-yon">
        <option value="above">Üstüne çıkınca</option>
        <option value="below">Altına düşünce</option>
      </select>
      <input class="pi" id="a-fiyat" placeholder="Hedef Fiyat ₺" type="number" style="max-width:145px">
      <button class="ba" onclick="alarmEkle()">+ Alarm</button>
    </div>
    <div id="alarm-listesi"><div class="es">Henüz alarm yok.</div></div>
  </div>
  <div class="card">
    <div class="ctit">Cihazlar Arasi Sync</div>
    <p style="font-size:12px;color:var(--mu);margin-bottom:10px">Asagidaki kodu baska cihaza yapistir, portfoy ve alarmlar oraya tasinir.</p>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <input class="pi" id="sync-kod" readonly style="flex:1;min-width:200px;font-size:11px;font-family:monospace">
      <button class="ba" onclick="syncKopyala()">Kopyala</button>
      <button class="ba" style="background:var(--sf);color:var(--tx);border:1px solid var(--bd)" onclick="syncUygula()">Uygula</button>
    </div>
    <input class="pi" id="sync-giris" placeholder="Baska cihazin kodunu buraya yapistir..." style="width:100%;margin-top:8px;font-size:11px;font-family:monospace">
  </div>
</div>
</div>

</main>

<!-- RIGHT: Fiyatlar -->
<aside class="sb sb-r">
  <div class="sbt">Kripto</div>
  <div id="kripto-listesi"><div class="es">Yükleniyor...</div></div>
  <div style="height:1px;background:var(--bd)"></div>
  <div class="sbt">Döviz</div>
  <div id="doviz-listesi"><div class="es">Yükleniyor...</div></div>
  <div style="height:1px;background:var(--bd)"></div>
  <div class="sbt">Emtia</div>
  <div id="emtia-listesi"><div class="es">Yükleniyor...</div></div>
  <div class="sbnote">5 dk. güncelleme · Yahoo Finance</div>
</aside>

</div><!-- /lay -->

<!-- Haberler Drawer (mobil) -->
<div class="drawer-overlay" id="drawer-overlay" onclick="drawerKapat()"></div>
<div class="drawer" id="haberler-drawer">
  <div class="drawer-hdr">
    <span style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em">Güncel Haberler</span>
    <button class="drawer-kapat" onclick="drawerKapat()">&#xD7;</button>
  </div>
  <div id="haber-listesi-mob"><div class="es">Yükleniyor...</div></div>
</div>

<!-- Alt Navigasyon (mobil) -->
<nav class="bnav" id="bnav">
  <button class="bni" id="bni-haber" onclick="drawerAc()">
    <svg viewBox="0 0 24 24"><path d="M4 6h16M4 10h12M4 14h10M4 18h8"/></svg>
    Haberler
  </button>
  <button class="bni active" id="bni-sinyaller" onclick="tabMob('sinyaller',this)">
    <svg viewBox="0 0 24 24"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>
    Sinyaller
  </button>
  <button class="bni" id="bni-grafik" onclick="tabMob('grafik',this)">
    <svg viewBox="0 0 24 24"><rect x="3" y="12" width="4" height="9" rx="1"/><rect x="10" y="7" width="4" height="14" rx="1"/><rect x="17" y="3" width="4" height="18" rx="1"/></svg>
    Grafik
  </button>
  <button class="bni" id="bni-trackrecord" onclick="tabMob('trackrecord',this)">
    <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
    Analiz
  </button>
  <button class="bni" id="bni-portfoy" onclick="tabMob('portfoy',this)">
    <svg viewBox="0 0 24 24"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/></svg>
    Portföy
  </button>
</nav>

<div id="sirket-modal" class="modal-bg" style="display:none" onclick="if(event.target===this)sirketKapat()">
  <div class="modal-kart" id="modal-icerik"><div class="es">Yukleniyor...</div></div>
</div>

<script>
const CBG='#07050e', CGR='#1c1830', CFN='#5e5a7a';
let grafikVerisi={}, trackData=null, period=90, fiyatlar={}, sinyalVerisi=[];
let indAktif={ma200:false,bb:false,macd:false,stoch:false,pivotlar:false,sinyaller:true};
let cizimModu=false, aktifArac=null, isinlar=[];

function getSid(){
  let sid=localStorage.getItem('lidya_sid');
  if(!sid){
    const a=new Uint8Array(16);
    crypto.getRandomValues(a);
    sid=[...a].map((b,i)=>(i===4||i===6||i===8||i===10?'-':'')+b.toString(16).padStart(2,'0')).join('');
    localStorage.setItem('lidya_sid',sid);
  }
  return sid;
}
const SID=getSid();

function tabAc(id,btn){
  document.querySelectorAll('.tp').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.tb').forEach(e=>e.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  if(btn) btn.classList.add('active');
  document.querySelectorAll('.bni[id^="bni-"]').forEach(e=>{
    if(e.id==='bni-'+id) e.classList.add('active');
    else if(e.id!=='bni-haber') e.classList.remove('active');
  });
  if(id==='grafik') grafikGuncelle();
  if(id==='trackrecord') perfCiz();
  if(id==='portfoy'){portfoyGun();alarmGun();const sk=document.getElementById('sync-kod');if(sk) sk.value=SID;}
}

function tabMob(id,btn){
  document.querySelectorAll('.bni').forEach(e=>e.classList.remove('active'));
  btn.classList.add('active');
  tabAc(id, null);
}

function drawerAc(){
  document.getElementById('haberler-drawer').classList.add('open');
  document.getElementById('drawer-overlay').classList.add('open');
  document.body.style.overflow='hidden';
}

function drawerKapat(){
  document.getElementById('haberler-drawer').classList.remove('open');
  document.getElementById('drawer-overlay').classList.remove('open');
  document.body.style.overflow='';
}

function veriCek(){
  fetch('/api/veri').then(r=>r.json()).then(data=>{
    grafikVerisi=data.grafik_verisi||{};
    trackData=data.track_record;
    sinyalVerisi=data.sinyaller||[];
    sinyalVerisi.forEach(s=>{fiyatlar[s.sembol.replace('.IS','')]=s.fiyat;});
    sayfaGun(data);
    alarmKontrol();
  }).catch(e=>console.log('Hata:',e));
}

function sayfaGun(data){
  const b=data.borsa_acik, d=document.getElementById('borsa-durum');
  d.textContent=b?'BORSA ACIK':'BORSA KAPALI';
  d.className='badge'+(b?'':' kapali');
  // saat saatGuncelle() tarafindan her saniye guncelleniyor
  if(data.kenar) kenarGun(data.kenar);
  if(data.haberler&&data.haberler.length) haberGun(data.haberler);
  if(!data.hazir) return;

  const rejim=(data.piyasa&&data.piyasa.rejim)||'YATAY';
  const re=document.getElementById('rejim-badge');
  re.textContent=((data.piyasa&&data.piyasa.emoji)||'')+' '+rejim;
  re.className='rb '+(rejim.includes('BOGA')?'rb-b':rejim.includes('AYI')?'rb-a':'rb-y');
  document.getElementById('rejim-aciklama').textContent=(data.piyasa&&data.piyasa.aciklama)||'';

  const sn=data.sinyaller||[];
  const al=sn.filter(s=>s.karar==='AL').length, sat=sn.filter(s=>s.karar==='SAT').length, bk=sn.filter(s=>s.karar==='BEKLE').length;
  document.getElementById('sinyal-sayisi').textContent=sn.length;
  document.getElementById('sinyal-ozet').innerHTML='<span class="gr">'+al+' AL</span> / <span class="re">'+sat+' SAT</span> / <span class="ye">'+bk+' BEKLE</span>';

  if(data.piyasa&&data.piyasa.bist_son){
    const p=data.piyasa, gr=p.bist_getiri_1ay>=0?'gr':'re';
    document.getElementById('bist-deger').textContent=Number(p.bist_son).toLocaleString('tr-TR',{maximumFractionDigits:0});
    document.getElementById('bist-sub').innerHTML='<span class="'+gr+'">1A: %'+Number(p.bist_getiri_1ay).toFixed(1)+'</span> | USD: '+Number(p.usdtry).toFixed(2);
  }

  const tr=data.track_record;
  if(tr&&tr.tamamlanan>0){
    const br=tr.basari>=55?'gr':tr.basari>=45?'ye':'re', kr=tr.ort_kar>=0?'gr':'re';
    const bd=document.getElementById('basari-deger');
    bd.className='sv '+br; bd.textContent='%'+tr.basari;
    document.getElementById('basari-sub').innerHTML='Tamamlanan: '+tr.tamamlanan+' | <span class="'+kr+'">%'+(tr.ort_kar>0?'+':'')+tr.ort_kar+'</span>';
  }else{
    document.getElementById('basari-deger').textContent='--';
    document.getElementById('basari-sub').textContent='Tamamlanan yok';
  }

  sektorGun(sn);
  tabloGun(sn);
  trTabGun(tr);
  sirketKartlariGun(sn);
}

const SMAP={AKBNK:'Bankacilik',GARAN:'Bankacilik',YKBNK:'Bankacilik',EKGYO:'Gayrimenkul',PGSUS:'Havacilik',THYAO:'Havacilik',TCELL:'Telekom',SISE:'Cam & Kimya',FROTO:'Otomotiv',EREGL:'Demir & Celik',ASELS:'Savunma',TUPRS:'Petrol & Enerji'};

function sektorGun(sn){
  const s={};
  sn.forEach(x=>{const k=x.sembol.replace('.IS',''),sek=SMAP[k]||'Diger';
    if(!s[sek]) s[sek]={al:0,sat:0,bk:0,d:[]};
    if(x.karar==='AL') s[sek].al++; else if(x.karar==='SAT') s[sek].sat++; else s[sek].bk++;
    s[sek].d.push(x.degisim||0);
  });
  let h='';
  Object.entries(s).forEach(([sek,b])=>{
    const ort=b.d.length?b.d.reduce((a,v)=>a+v,0)/b.d.length:0;
    const r=ort>0?'gr':ort<0?'re':'ye';
    const du=b.al>b.sat?'AL agirlikli':b.sat>b.al?'SAT agirlikli':'Karisik';
    h+='<div class="ek"><div class="enm">'+sek+'</div><div class="ev '+r+'">'+(ort>=0?'+':'')+ort.toFixed(1)+'%</div><div class="ed">'+du+'</div></div>';
  });
  document.getElementById('sektor-grid').innerHTML=h||'<div class="es">Yukleniyor...</div>';
}

function tabloGun(sn){
  if(!sn||!sn.length){document.getElementById('sinyal-tablo-alani').innerHTML='<div class="es">Modeller egitiliyor...</div>';return;}
  let h='<table class="t"><thead><tr><th>Hisse</th><th>Fiyat</th><th>Degisim</th><th>RSI</th><th>Karar</th><th>Guven</th><th>Hedef</th><th>Stop</th></tr></thead><tbody>';
  sn.forEach(s=>{
    const dr=s.degisim>=0?'gr':'re', di=s.degisim>=0?'+':'';
    const kc=s.karar==='AL'?'al':s.karar==='SAT'?'sat':'bekle';
    const rc=s.karar==='AL'?'al-r':s.karar==='SAT'?'sat-r':'bk-r';
    const gp=(s.guven*100).toFixed(0), rr=s.rsi<40?'gr':s.rsi>60?'re':'ye';
    h+='<tr class="'+rc+'"><td style="font-weight:700">'+s.sembol.replace('.IS','')+'</td>'+
      '<td>'+Number(s.fiyat).toFixed(2)+' TL</td>'+
      '<td class="'+dr+'">'+di+Number(s.degisim).toFixed(2)+'%</td>'+
      '<td class="'+rr+'">'+Number(s.rsi).toFixed(1)+'</td>'+
      '<td><span class="pill '+kc+'">'+s.karar+'</span></td>'+
      '<td>%'+gp+'<div class="gb"><div class="gf" style="width:'+gp+'%"></div></div></td>'+
      '<td class="gr">'+(s.hedef?Number(s.hedef).toFixed(2)+' TL':'-')+'</td>'+
      '<td class="'+(s.karar==='SAT'?'gr':'re')+'">'+(s.stop?Number(s.stop).toFixed(2)+' TL':'-')+'</td></tr>';
  });
  document.getElementById('sinyal-tablo-alani').innerHTML=h+'</tbody></table>';
}

function periodSec(g,btn){
  period=g;
  document.querySelectorAll('.pb').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  grafikGuncelle();
}

function indToggle(id,btn){
  indAktif[id]=!indAktif[id];
  btn.classList.toggle('active');
  const ma=document.getElementById('macd-alan'), sa=document.getElementById('stoch-alan');
  if(ma) ma.style.display=indAktif.macd?'':'none';
  if(sa) sa.style.display=indAktif.stoch?'':'none';
  grafikGuncelle();
}

function cizimToggle(btn){
  cizimModu=!cizimModu;
  btn.classList.toggle('cizim-aktif');
  btn.textContent=cizimModu?'✏ Serbest ON':'✏ Serbest';
  if(cizimModu) _aracKapat();
  grafikGuncelle();
}

function _aracKapat(){
  aktifArac=null;
  ['btn-h-isin','btn-v-isin'].forEach(id=>{
    const b=document.getElementById(id);
    if(b) b.classList.remove('cizim-aktif');
  });
  const g=document.getElementById('grafik-alan');
  if(g) g.style.cursor='';
}

function aracSec(arac,btn){
  if(aktifArac===arac){_aracKapat();return;}
  aktifArac=arac;
  ['btn-h-isin','btn-v-isin'].forEach(id=>{
    const b=document.getElementById(id);
    if(b) b.classList.remove('cizim-aktif');
  });
  btn.classList.add('cizim-aktif');
  // Serbest çizim modunu kapat
  if(cizimModu){
    cizimModu=false;
    const cb=document.getElementById('btn-cizim');
    if(cb){cb.classList.remove('cizim-aktif');cb.textContent='✏ Serbest';}
  }
  const g=document.getElementById('grafik-alan');
  if(g) g.style.cursor='crosshair';
  grafikGuncelle();
}

function isinTemizle(){
  isinlar=[];
  grafikGuncelle();
}

function grafikEventleriAktar(){
  const el=document.getElementById('grafik-alan');
  if(!el||typeof el.removeAllListeners!=='function') return;
  el.removeAllListeners('plotly_click');
  el.on('plotly_click',function(data){
    if(!aktifArac||!data||!data.points||!data.points.length) return;
    const pt=data.points[0];
    const x=pt.x;
    const y=pt.y!=null?pt.y:(pt.close??pt.high??pt.low??pt.open);
    if(y==null) return;
    if(aktifArac==='h-isin'){
      const fiyat=parseFloat(parseFloat(y).toFixed(2));
      isinlar.push({
        type:'line',xref:'paper',yref:'y',
        x0:0,x1:1,y0:fiyat,y1:fiyat,
        line:{color:'#a78bfa',width:1.5},
        _lbl:fiyat.toFixed(2)+' TL',_tip:'h'
      });
    } else if(aktifArac==='v-isin'){
      isinlar.push({
        type:'line',xref:'x',yref:'paper',
        x0:x,x1:x,y0:0,y1:1,
        line:{color:'#f59e0b',width:1.5,dash:'dash'},
        _lbl:String(x).slice(0,10),_tip:'v'
      });
    }
    grafikGuncelle();
  });
}

function _maHesapla(arr,p){
  return arr.map((v,i)=>{
    if(i<p-1) return null;
    let s=0,c=0;
    for(let j=i-p+1;j<=i;j++){if(arr[j]!=null){s+=arr[j];c++;}}
    return c===p?parseFloat((s/p).toFixed(2)):null;
  });
}

function _pivotHesapla(hi,lo,tar,bak=8){
  const n=hi.length, res=[];
  const cur=lo[n-1]||0;
  const seen=new Set();
  for(let i=bak;i<n-bak;i++){
    let isH=true,isL=true;
    for(let j=i-bak;j<=i+bak;j++){
      if(j===i) continue;
      if(hi[j]>=hi[i]) isH=false;
      if(lo[j]<=lo[i]) isL=false;
    }
    if(isH){const v=Math.round(hi[i]*100)/100;if(!seen.has('R'+v)){res.push({f:v,t:'R',x:tar[i]});seen.add('R'+v);}}
    if(isL){const v=Math.round(lo[i]*100)/100;if(!seen.has('S'+v)){res.push({f:v,t:'S',x:tar[i]});seen.add('S'+v);}}
  }
  const direnc=res.filter(p=>p.t==='R'&&p.f>cur).sort((a,b)=>a.f-b.f).slice(0,3);
  const destek=res.filter(p=>p.t==='S'&&p.f<cur*1.03).sort((a,b)=>b.f-a.f).slice(0,3);
  return [...direnc,...destek];
}

function grafikGuncelle(){
  const el=document.getElementById('hisse-sec');
  if(!el) return;
  const hisse=el.value, veri=grafikVerisi[hisse];
  if(!veri||!veri.tarihler) return;
  const n=Math.min(period,veri.tarihler.length);
  const sl=arr=>(arr||[]).slice(-n);
  const tar=sl(veri.tarihler),op=sl(veri.open),hi=sl(veri.high),lo=sl(veri.low),cl=sl(veri.close);
  const m20=sl(veri.ma20),m50=sl(veri.ma50),vol=sl(veri.volume),rsi=sl(veri.rsi);
  const BL={paper_bgcolor:CBG,plot_bgcolor:CBG,font:{color:CFN,size:10},margin:{t:6,r:10,b:26,l:60}};

  // Ana grafik trace'leri
  const traces=[
    {type:'candlestick',x:tar,open:op,high:hi,low:lo,close:cl,name:hisse.replace('.IS',''),
     increasing:{line:{color:'#22c55e',width:1},fillcolor:'rgba(34,197,94,.18)'},
     decreasing:{line:{color:'#ef4444',width:1},fillcolor:'rgba(239,68,68,.18)'}},
    {type:'scatter',x:tar,y:m20,name:'MA20',line:{color:'#f59e0b',width:1.5},opacity:.9},
    {type:'scatter',x:tar,y:m50,name:'MA50',line:{color:'#8b5cf6',width:1.5},opacity:.9}
  ];

  // MA200
  if(indAktif.ma200&&veri.ma200){
    traces.push({type:'scatter',x:tar,y:sl(veri.ma200),name:'MA200',
      line:{color:'#ec4899',width:1.5,dash:'dash'},opacity:.85});
  }

  // Bollinger Bands
  if(indAktif.bb&&veri.bb_ust){
    const bu=sl(veri.bb_ust),ba=sl(veri.bb_alt);
    traces.push({type:'scatter',x:tar,y:bu,name:'BB Üst',
      line:{color:'rgba(99,102,241,.55)',width:1,dash:'dot'},showlegend:false});
    traces.push({type:'scatter',x:tar,y:ba,name:'BB Alt',
      line:{color:'rgba(99,102,241,.55)',width:1,dash:'dot'},
      fill:'tonexty',fillcolor:'rgba(99,102,241,.05)',showlegend:false});
  }

  // Pivot / Destek-Direnç
  const shapes=[],anns=[];
  if(indAktif.pivotlar){
    _pivotHesapla(hi,lo,tar).forEach(p=>{
      const col=p.t==='R'?'rgba(239,68,68,.45)':'rgba(34,197,94,.45)';
      shapes.push({type:'line',x0:tar[0],x1:tar[tar.length-1],y0:p.f,y1:p.f,
        line:{color:col,width:1,dash:'dot'},xref:'x',yref:'y'});
      anns.push({x:tar[tar.length-1],y:p.f,text:(p.t==='R'?'D ':'S ')+p.f.toFixed(2),
        xanchor:'right',showarrow:false,
        font:{size:9,color:p.t==='R'?'#ef4444':'#22c55e'},bgcolor:'transparent'});
    });
  }

  // Sinyal işaretçileri
  if(indAktif.sinyaller){
    const sin=sinyalVerisi.find(s=>s.sembol===hisse);
    if(sin&&sin.karar!=='BEKLE'){
      const isAl=sin.karar==='AL';
      const lx=tar[tar.length-1];
      const ly=isAl?lo[lo.length-1]:hi[hi.length-1];
      const offset=(hi[hi.length-1]-lo[lo.length-1])*0.6||1;
      anns.push({x:lx,y:isAl?ly-offset:ly+offset,
        text:sin.karar,showarrow:true,arrowhead:2,arrowsize:1.2,
        arrowcolor:isAl?'#22c55e':'#ef4444',ax:0,ay:isAl?30:-30,
        font:{size:11,color:isAl?'#22c55e':'#ef4444',family:'monospace'},
        bgcolor:'transparent'});
    }
  }

  // Kullanıcı çizilen ışınları birleştir
  isinlar.forEach(r=>{
    shapes.push({type:r.type,xref:r.xref,yref:r.yref,
      x0:r.x0,x1:r.x1,y0:r.y0,y1:r.y1,line:r.line});
    if(r._tip==='h'){
      anns.push({x:0.995,y:r.y0,xref:'paper',yref:'y',
        text:r._lbl,xanchor:'right',showarrow:false,
        font:{size:9,color:'#a78bfa'},
        bgcolor:'rgba(7,5,14,.75)',borderpad:2,bordercolor:'rgba(167,139,250,.3)',borderwidth:1});
    } else if(r._tip==='v'){
      anns.push({x:r.x0,y:1,xref:'x',yref:'paper',
        text:r._lbl,xanchor:'left',yanchor:'top',showarrow:false,
        font:{size:9,color:'#f59e0b'},
        bgcolor:'rgba(7,5,14,.75)',borderpad:2,bordercolor:'rgba(245,158,11,.3)',borderwidth:1});
    }
  });

  const cfg=cizimModu
    ?{responsive:true,displayModeBar:true,
      modeBarButtonsToAdd:['drawline','drawopenpath','drawrect','drawcircle','eraseshape'],
      modeBarButtonsToRemove:['autoScale2d','lasso2d','select2d','toImage'],
      displaylogo:false}
    :{responsive:true,displayModeBar:false};

  Plotly.newPlot('grafik-alan',traces,
    {...BL,xaxis:{gridcolor:CGR,rangeslider:{visible:false},type:'date'},
     yaxis:{gridcolor:CGR,ticksuffix:' TL'},
     shapes:shapes,annotations:anns,
     newshape:{line:{color:'#a78bfa',width:2}},
     legend:{bgcolor:CBG+'aa',font:{size:10}},showlegend:true},
  cfg);

  // Hacim + MA20
  if(vol&&vol.length){
    const volMa=_maHesapla(vol,20);
    Plotly.newPlot('hacim-alan',[
      {type:'bar',x:tar,y:vol,name:'Hacim',
        marker:{color:cl.map((c,i)=>c>=(op[i]||c)?'rgba(34,197,94,.35)':'rgba(239,68,68,.35)')}},
      {type:'scatter',x:tar,y:volMa,name:'Hac.MA20',
        line:{color:'#f59e0b',width:1.5},opacity:.9}
    ],{...BL,xaxis:{gridcolor:CGR,type:'date'},yaxis:{gridcolor:CGR,tickformat:'.2s'},showlegend:false},
    {responsive:true,displayModeBar:false});
  }

  // RSI
  if(rsi&&rsi.length){
    const x0=tar[0],x1=tar[tar.length-1];
    Plotly.newPlot('rsi-alan',[
      {type:'scatter',x:tar,y:rsi,line:{color:'#6366f1',width:1.5},name:'RSI'},
      {type:'scatter',x:[x0,x1],y:[70,70],line:{color:'rgba(239,68,68,.3)',width:1,dash:'dot'},showlegend:false},
      {type:'scatter',x:[x0,x1],y:[30,30],line:{color:'rgba(34,197,94,.3)',width:1,dash:'dot'},showlegend:false}
    ],{...BL,xaxis:{gridcolor:CGR,type:'date'},yaxis:{gridcolor:CGR,range:[0,100],tickvals:[30,50,70]},showlegend:false},
    {responsive:true,displayModeBar:false});
  }

  // MACD
  if(indAktif.macd&&veri.macd){
    const mc=sl(veri.macd),ms=sl(veri.macd_s),mh=sl(veri.macd_h);
    Plotly.newPlot('macd-alan',[
      {type:'bar',x:tar,y:mh,name:'Hist',
        marker:{color:mh.map(v=>v>=0?'rgba(34,197,94,.5)':'rgba(239,68,68,.5)')}},
      {type:'scatter',x:tar,y:mc,name:'MACD',line:{color:'#6366f1',width:1.5}},
      {type:'scatter',x:tar,y:ms,name:'Signal',line:{color:'#f59e0b',width:1.5}}
    ],{...BL,xaxis:{gridcolor:CGR,type:'date'},yaxis:{gridcolor:CGR},showlegend:false},
    {responsive:true,displayModeBar:false});
  }

  // Stochastic
  if(indAktif.stoch&&veri.stoch_k){
    const sk=sl(veri.stoch_k),sd=sl(veri.stoch_d);
    const x0=tar[0],x1=tar[tar.length-1];
    Plotly.newPlot('stoch-alan',[
      {type:'scatter',x:tar,y:sk,name:'%K',line:{color:'#6366f1',width:1.5}},
      {type:'scatter',x:tar,y:sd,name:'%D',line:{color:'#f59e0b',width:1.5}},
      {type:'scatter',x:[x0,x1],y:[80,80],line:{color:'rgba(239,68,68,.3)',width:1,dash:'dot'},showlegend:false},
      {type:'scatter',x:[x0,x1],y:[20,20],line:{color:'rgba(34,197,94,.3)',width:1,dash:'dot'},showlegend:false}
    ],{...BL,xaxis:{gridcolor:CGR,type:'date'},yaxis:{gridcolor:CGR,range:[0,100],tickvals:[20,50,80]},showlegend:false},
    {responsive:true,displayModeBar:false});
  }

  grafikEventleriAktar();
}

function trTabGun(tr){
  if(!tr) return;
  const br=tr.basari>=55?'gr':tr.basari>=45?'ye':'re', kr=tr.ort_kar>=0?'gr':'re';
  document.getElementById('tr-toplam').textContent=tr.toplam;
  document.getElementById('tr-tamamlanan').textContent=tr.tamamlanan;
  const tb=document.getElementById('tr-basari'); tb.className='trv '+br;
  tb.textContent=tr.tamamlanan>0?'%'+tr.basari:'-';
  const tk=document.getElementById('tr-ort-kar'); tk.className='trv '+kr;
  tk.textContent=tr.tamamlanan>0?'%'+(tr.ort_kar>0?'+':'')+tr.ort_kar:'-';
  if(!tr.son_sinyaller||!tr.son_sinyaller.length){
    document.getElementById('track-record-alani').innerHTML='<div class="es">Henuz tamamlanan sinyal yok.</div>';return;
  }
  let h='<table class="t"><thead><tr><th>Tarih</th><th>Hisse</th><th>Karar</th><th>Giris</th><th>Hedef</th><th>Stop</th><th>Cikis</th><th>K/Z</th><th>Sonuc</th></tr></thead><tbody>';
  tr.son_sinyaller.forEach(s=>{
    const kzv=s.kar_zarar?parseFloat(s.kar_zarar):null;
    const sr=s.sonuc==='KAZANDI'?'gr':s.sonuc==='KAYBETTI'?'re':'ye';
    const kc=s.karar==='AL'?'al':s.karar==='SAT'?'sat':'bekle';
    const rc=s.karar==='AL'?'al-r':s.karar==='SAT'?'sat-r':'bk-r';
    const kzs=kzv!=null?'<span class="'+(kzv>=0?'gr':'re')+'">%'+(kzv>0?'+':'')+kzv.toFixed(1)+'</span>':'-';
    h+='<tr class="'+rc+'"><td style="font-size:10px;white-space:nowrap">'+(s.zaman||'-')+'</td>'+
      '<td style="font-weight:700">'+(s.sembol||'').replace('.IS','')+'</td>'+
      '<td><span class="pill '+kc+'">'+s.karar+'</span></td>'+
      '<td>'+(s.fiyat_giris?parseFloat(s.fiyat_giris).toFixed(2)+' TL':'-')+'</td>'+
      '<td class="gr">'+(s.hedef?parseFloat(s.hedef).toFixed(2)+' TL':'-')+'</td>'+
      '<td class="'+(s.karar==='SAT'?'gr':'re')+'">'+(s.stop?parseFloat(s.stop).toFixed(2)+' TL':'-')+'</td>'+
      '<td>'+(s.fiyat_cikis?parseFloat(s.fiyat_cikis).toFixed(2)+' TL':'-')+'</td>'+
      '<td>'+kzs+'</td>'+
      '<td class="'+sr+'" style="font-weight:600">'+(s.sonuc||'Bekliyor')+'</td></tr>';
  });
  document.getElementById('track-record-alani').innerHTML=h+'</tbody></table>';
}

function perfCiz(){
  if(!trackData||!trackData.tamamlanan_liste||!trackData.tamamlanan_liste.length) return;
  const liste=trackData.tamamlanan_liste;
  let cum=0; const xs=[],ys=[],txts=[];
  liste.forEach((item,i)=>{cum+=parseFloat(item.kar_zarar||0);xs.push(i+1);ys.push(parseFloat(cum.toFixed(2)));txts.push((item.sembol||'').replace('.IS','')+' '+(item.sonuc||''));});
  const last=ys[ys.length-1]||0;
  Plotly.newPlot('perf-grafik',[{type:'scatter',x:xs,y:ys,mode:'lines+markers',
    line:{color:'#6366f1',width:2},marker:{color:ys.map(v=>v>=0?'#22c55e':'#ef4444'),size:5},
    fill:'tozeroy',fillcolor:last>=0?'rgba(34,197,94,.06)':'rgba(239,68,68,.06)',
    text:txts,hovertemplate:'%{text}<br>Kumulatif: %{y:.1f}%<extra></extra>'}],
  {paper_bgcolor:CBG,plot_bgcolor:CBG,font:{color:CFN,size:11},
   xaxis:{gridcolor:CGR,title:'Sinyal #'},yaxis:{gridcolor:CGR,ticksuffix:'%',zeroline:true,zerolinecolor:'#2e2a48'},
   margin:{t:6,r:10,b:36,l:50},showlegend:false},
  {responsive:true,displayModeBar:false});
}

function fp(v,d){return v==null?'-':Number(v).toLocaleString('tr-TR',{minimumFractionDigits:d,maximumFractionDigits:d});}

function kenarGun(kenar){
  ['kripto','doviz','emtia'].forEach(grp=>{
    if(!kenar[grp]||!kenar[grp].length) return;
    const pfx=grp==='kripto'?'$':grp==='doviz'?'TL':'$';
    const dec=grp==='emtia'?1:2;
    let h='';
    kenar[grp].forEach(x=>{
      const r=(x.degisim||0)>=0?'gr':'re', d=(x.degisim||0)>=0?'+':'';
      h+='<div class="sbi"><div class="prow">'+
        '<div><div class="pname">'+x.name+'</div><div class="pval">'+pfx+fp(x.fiyat,dec)+'</div></div>'+
        '<div class="pchg '+r+'">'+d+Number(x.degisim||0).toFixed(1)+'%</div>'+
        '</div></div>';
    });
    document.getElementById(grp+'-listesi').innerHTML=h;
  });
}

function haberGun(haberler){
  let h='';
  haberler.forEach(x=>{
    h+='<div class="sbi"><a class="na" href="'+(x.link||'#')+'" target="_blank" rel="noopener">'+
      '<div class="nt">'+(x.baslik||'')+'</div>'+
      '<div class="ns">'+(x.kaynak||'')+'</div>'+
      '</a></div>';
  });
  const c=h||'<div class="es">Haber yok.</div>';
  document.getElementById('haber-listesi').innerHTML=c;
  const mob=document.getElementById('haber-listesi-mob');
  if(mob) mob.innerHTML=c;
}

function portfoyGrafikleriCiz(p){
  const sat=document.getElementById('portfoy-grafik-satir');
  if(!p||!p.length){if(sat) sat.style.display='none';return;}
  if(sat) sat.style.display='';
  const lbl=[],vals=[],kzArr=[],clrs=[];
  const rnk=['#a78bfa','#22c55e','#f59e0b','#6366f1','#ef4444','#06b6d4','#ec4899','#84cc16'];
  p.forEach((x,i)=>{
    const g=fiyatlar[x.sembol]||x.maliyet;
    const val=parseFloat((g*x.adet).toFixed(0));
    const kzp=parseFloat(((g-x.maliyet)/x.maliyet*100).toFixed(2));
    lbl.push(x.sembol); vals.push(val); kzArr.push(kzp); clrs.push(rnk[i%rnk.length]);
  });
  const lay0={paper_bgcolor:CBG,plot_bgcolor:CBG,font:{color:'#ddd6f3',size:11},margin:{t:8,b:8,l:8,r:8}};
  const cfg={responsive:true,displayModeBar:false};
  Plotly.newPlot('portfoy-pasta',[{
    type:'pie',labels:lbl,values:vals,hole:0.42,
    marker:{colors:clrs,line:{color:CBG,width:2}},
    textinfo:'label+percent',textfont:{size:10},
    hovertemplate:'%{label}<br>%{value:,.0f} TL<extra></extra>'
  }],{...lay0,showlegend:false,margin:{t:8,b:8,l:8,r:8}},cfg);
  const barClr=kzArr.map(v=>v>=0?'rgba(34,197,94,.75)':'rgba(239,68,68,.75)');
  Plotly.newPlot('portfoy-bar',[{
    type:'bar',x:lbl,y:kzArr,marker:{color:barClr,line:{width:0}},
    text:kzArr.map(v=>(v>=0?'+':'')+v+'%'),textposition:'outside',
    hovertemplate:'%{x}: %{y:.2f}%<extra></extra>'
  }],{...lay0,margin:{t:22,b:30,l:36,r:8},
    yaxis:{gridcolor:'rgba(46,42,72,.4)',zerolinecolor:'rgba(46,42,72,.8)',ticksuffix:'%'},
    xaxis:{tickfont:{size:10}}},cfg);
}

function _portfoyRender(p){
  const tbl=document.getElementById('portfoy-tablo'), oz=document.getElementById('portfoy-ozet');
  if(!p.length){tbl.innerHTML='<div class="es">Portfoy bos.</div>';if(oz) oz.style.display='none';return;}
  let h='<table class="t"><thead><tr><th>Hisse</th><th>Adet</th><th>Maliyet</th><th>Guncel</th><th>Piyasa D.</th><th>K/Z</th><th>K/Z %</th><th></th></tr></thead><tbody>';
  let totM=0,totD=0;
  p.forEach(x=>{
    const g=fiyatlar[x.sembol]||null,pd=g!==null?g*x.adet:null,md=x.maliyet*x.adet;
    const kz=pd!==null?pd-md:null,kzp=g!==null?(g-x.maliyet)/x.maliyet*100:null;
    const r=kz!==null?(kz>=0?'gr':'re'):'';
    totM+=md; if(pd!==null) totD+=pd;
    h+='<tr><td style="font-weight:700">'+x.sembol+'</td><td>'+x.adet+'</td>'+
      '<td>'+x.maliyet.toFixed(2)+' TL</td>'+
      '<td>'+(g!==null?g.toFixed(2)+' TL':'<span style="color:var(--mu)">-</span>')+'</td>'+
      '<td>'+(pd!==null?pd.toLocaleString(\'tr-TR\',{maximumFractionDigits:0})+\' TL\':\'-\')+'</td>'+
      '<td class="'+r+'">'+(kz!==null?(kz>=0?\'+\':\'\')+kz.toLocaleString(\'tr-TR\',{maximumFractionDigits:0})+\' TL\':\'-\')+'</td>'+
      '<td class="'+r+'">'+(kzp!==null?(kzp>=0?\'+\':\'\')+kzp.toFixed(1)+\'%\':\'-\')+'</td>'+
      '<td><button class="bd2" onclick="portfoySil(\\\''+x.sembol+'\\\')">Sil</button></td></tr>';
  });
  tbl.innerHTML=h+'</tbody></table>';
  const nkz=totD-totM,nkzp=totM>0?nkz/totM*100:0,nr=nkz>=0?'gr':'re';
  if(oz){oz.style.display='flex';oz.innerHTML='<span>Maliyet: <strong>'+totM.toLocaleString('tr-TR',{maximumFractionDigits:0})+' TL</strong></span><span>Piyasa D.: <strong>'+totD.toLocaleString('tr-TR',{maximumFractionDigits:0})+' TL</strong></span><span>Net K/Z: <strong class="'+nr+'">'+(nkz>=0?'+':'')+nkz.toLocaleString('tr-TR',{maximumFractionDigits:0})+' TL (%'+(nkzp>=0?'+':'')+nkzp.toFixed(1)+')</strong></span>';}
  portfoyGrafikleriCiz(p);
}

function _lsPortfoy(){return JSON.parse(localStorage.getItem('portfoy')||'[]');}
function _lsPortfoyKaydet(p){localStorage.setItem('portfoy',JSON.stringify(p));}

function portfoyEkle(){
  const s=(document.getElementById('p-sembol').value||'').toUpperCase().trim();
  const a=parseFloat(document.getElementById('p-adet').value);
  const m=parseFloat(document.getElementById('p-maliyet').value);
  if(!s||!a||!m){alert('Tum alanlari doldurun.');return;}
  document.getElementById('p-sembol').value='';
  document.getElementById('p-adet').value='';
  document.getElementById('p-maliyet').value='';
  const p=_lsPortfoy(); const i=p.findIndex(x=>x.sembol===s);
  if(i>=0) p[i]={sembol:s,adet:a,maliyet:m}; else p.push({sembol:s,adet:a,maliyet:m});
  _lsPortfoyKaydet(p);
  _portfoyRender(p);
  fetch('/api/portfoy/'+SID,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sembol:s,adet:a,maliyet:m})}).catch(()=>{});
}

function portfoySil(s){
  const p=_lsPortfoy().filter(x=>x.sembol!==s);
  _lsPortfoyKaydet(p);
  _portfoyRender(p);
  fetch('/api/portfoy/'+SID+'/'+s,{method:'DELETE'}).catch(()=>{});
}

function portfoyGun(){
  const local=_lsPortfoy();
  _portfoyRender(local);
  fetch('/api/portfoy/'+SID)
    .then(r=>r.json())
    .then(srv=>{
      if(srv&&srv.length>0){_lsPortfoyKaydet(srv);_portfoyRender(srv);}
    })
    .catch(()=>{});
}

function _alarmRender(a){
  const el=document.getElementById('alarm-listesi');
  if(!el) return;
  if(!a.length){el.innerHTML='<div class="es">Henuz alarm yok.</div>';return;}
  let h='<table class="t"><thead><tr><th>Hisse</th><th>Kosul</th><th>Hedef</th><th>Guncel</th><th>Durum</th><th></th></tr></thead><tbody>';
  a.forEach((x)=>{
    const g=fiyatlar[x.sembol];
    const du=x.tetiklendi?'<span class="alb alh">Tetiklendi</span>':'<span class="alb">Bekliyor</span>';
    const delKey=x.id!==undefined?x.id:JSON.stringify(x);
    h+='<tr><td style="font-weight:700">'+x.sembol+'</td>'+
      '<td>'+(x.yon==='above'?'Yukari':'Asagi')+'</td>'+
      '<td>'+parseFloat(x.fiyat).toFixed(2)+' TL</td>'+
      '<td>'+(g!==undefined?g.toFixed(2)+' TL':'-')+'</td>'+
      '<td>'+du+'</td>'+
      '<td><button class="bd2" onclick="alarmSil('+JSON.stringify(delKey)+')">Sil</button></td></tr>';
  });
  el.innerHTML=h+'</tbody></table>';
  if('Notification' in window&&Notification.permission==='default') Notification.requestPermission();
}

function _lsAlarmlar(){return JSON.parse(localStorage.getItem('alarmlar')||'[]');}
function _lsAlarmlarKaydet(a){localStorage.setItem('alarmlar',JSON.stringify(a));}

function alarmEkle(){
  const s=(document.getElementById('a-sembol').value||'').toUpperCase().trim();
  const y=document.getElementById('a-yon').value;
  const f=parseFloat(document.getElementById('a-fiyat').value);
  if(!s||!f){alert('Sembol ve fiyat zorunlu.');return;}
  document.getElementById('a-sembol').value='';
  document.getElementById('a-fiyat').value='';
  const a=_lsAlarmlar();
  a.push({sembol:s,yon:y,fiyat:f,tetiklendi:false});
  _lsAlarmlarKaydet(a);
  _alarmRender(a);
  fetch('/api/alarmlar/'+SID,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sembol:s,yon:y,fiyat:f})})
    .then(r=>r.json())
    .then(res=>{if(res.id){alarmGun();}})
    .catch(()=>{});
}

function alarmSil(key){
  if(typeof key==='number'){
    fetch('/api/alarmlar/'+SID+'/'+key,{method:'DELETE'}).catch(()=>{});
    const a=_lsAlarmlar().filter(x=>x.id!==key);
    _lsAlarmlarKaydet(a); _alarmRender(a);
  } else {
    const a=_lsAlarmlar();
    const idx=a.findIndex(x=>JSON.stringify(x)===key);
    if(idx>=0) a.splice(idx,1);
    _lsAlarmlarKaydet(a); _alarmRender(a);
  }
  alarmGun();
}

function alarmKontrol(){
  const a=_lsAlarmlar(); let ch=false;
  a.forEach((alarm,i)=>{
    if(alarm.tetiklendi) return;
    const g=fiyatlar[alarm.sembol]; if(g===undefined) return;
    const hit=alarm.yon==='above'?g>=alarm.fiyat:g<=alarm.fiyat;
    if(hit){a[i].tetiklendi=true;ch=true;
      if('Notification' in window&&Notification.permission==='granted')
        new Notification('Fiyat Alarmi',{body:alarm.sembol+' -> '+g.toFixed(2)+' TL'});
    }
  });
  if(ch){_lsAlarmlarKaydet(a); _alarmRender(a);}
}

function alarmGun(){
  const local=_lsAlarmlar();
  _alarmRender(local);
  fetch('/api/alarmlar/'+SID)
    .then(r=>r.json())
    .then(srv=>{
      if(srv&&srv.length>0){_lsAlarmlarKaydet(srv);_alarmRender(srv);}
    })
    .catch(()=>{});
}

function syncKopyala(){
  const el=document.getElementById('sync-kod');
  if(el){navigator.clipboard.writeText(el.value).then(()=>alert('Sync kodu kopyalandi!'));}
}

function syncUygula(){
  const giris=(document.getElementById('sync-giris').value||'').trim();
  if(!giris){alert('Lutfen bir sync kodu girin.');return;}
  localStorage.setItem('lidya_sid',giris);
  alert('Sync kodu uygulandı! Sayfa yenileniyor...');
  location.reload();
}

function sirketKartlariGun(sn){
  const el=document.getElementById('sirket-kartlar');
  if(!el||!sn||!sn.length) return;
  const rnk=['#a78bfa','#22c55e','#f59e0b','#6366f1','#ef4444','#06b6d4','#ec4899','#84cc16'];
  let h='';
  sn.forEach((s,i)=>{
    const k=s.sembol.replace('.IS','');
    const r=rnk[i%rnk.length];
    const kc=s.karar==='AL'?'al':s.karar==='SAT'?'sat':'bekle';
    h+='<div class="sirket-kart" data-sembol="'+k+'" onclick="sirketAc(this.dataset.sembol)">'+
      '<div class="sk-logo" style="background:'+r+'22;border:1px solid '+r+'44;color:'+r+'">'+k.substring(0,2)+'</div>'+
      '<div class="sk-ad">'+k+'</div>'+
      '<span class="pill '+kc+'">'+s.karar+'</span></div>';
  });
  el.innerHTML=h;
}

function sirketAc(sembol){
  const modal=document.getElementById('sirket-modal');
  modal.style.display='flex';
  document.getElementById('modal-icerik').innerHTML='<div class="es">Yukleniyor...</div>';
  fetch('/api/hisse/'+sembol).then(r=>r.json()).then(d=>{
    const s=d.sinyal,f=d.finans||{};
    const kc=s?(s.karar==='AL'?'al':s.karar==='SAT'?'sat':'bekle'):'bekle';
    const fmt=(v,dec,suf)=>v==null?'-':(Number(v).toLocaleString('tr-TR',{maximumFractionDigits:dec||0}))+(suf||'');
    const fmtP=v=>v==null?'-':v>=1e12?(v/1e12).toFixed(1)+' T TL':v>=1e9?(v/1e9).toFixed(1)+' Mr TL':v>=1e6?(v/1e6).toFixed(0)+' Mn TL':fmt(v,0,' TL');
    const metriks=[
      ['Piyasa Degeri',fmtP(f.piyasaDegeri)],
      ['F/K Orani',fmt(f.fk,1)],
      ['PD/DD',fmt(f.pd_dd,2)],
      ['52H Yuksek',fmt(f.yuksek52,2,' TL')],
      ['52H Dusuk',fmt(f.dusuk52,2,' TL')],
      ['Temettu',f.temettu!=null?'%'+Number(f.temettu*100).toFixed(1):'-'],
      ['Beta',fmt(f.beta,2)],
      ['Calisan',f.calisanSayisi?Number(f.calisanSayisi).toLocaleString('tr-TR'):'-'],
    ];
    let h='<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px">'+
      '<div style="display:flex;gap:12px;align-items:center">'+
      '<div style="width:50px;height:50px;border-radius:12px;background:rgba(167,139,250,.12);border:1px solid rgba(167,139,250,.3);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:15px;color:var(--ac)">'+sembol.substring(0,2)+'</div>'+
      '<div><div style="font-size:17px;font-weight:700;color:var(--tx)">'+d.ad+'</div>'+
      '<div style="font-size:11px;color:var(--mu);margin-top:2px">'+sembol+' - '+d.sektor+'</div></div></div>'+
      '<div style="display:flex;gap:8px;align-items:center">'+
      (s?'<span class="pill '+kc+'">'+s.karar+'</span>':'')+
      '<button onclick="sirketKapat()" style="background:var(--sf);border:1px solid var(--bd);color:var(--mu);width:30px;height:30px;border-radius:6px;cursor:pointer;font-size:16px;line-height:1">x</button>'+
      '</div></div>';
    if(d.aciklama) h+='<div style="font-size:12px;color:var(--mu);line-height:1.65;padding:12px;background:var(--sf);border-radius:8px;margin-bottom:2px">'+d.aciklama+'</div>';
    h+='<div class="met-grid">';
    metriks.forEach(function(m){h+='<div class="met"><div class="met-l">'+m[0]+'</div><div class="met-v">'+m[1]+'</div></div>';});
    h+='</div>';
    if(s){
      const dr=s.degisim>=0?'gr':'re';
      h+='<div style="background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:11px;display:flex;gap:16px;flex-wrap:wrap;font-size:12px">'+
        '<span>Fiyat: <strong>'+Number(s.fiyat).toFixed(2)+' TL</strong></span>'+
        '<span class="'+dr+'">Degisim: '+(s.degisim>=0?'+':'')+Number(s.degisim).toFixed(2)+'%</span>'+
        '<span>RSI: <strong>'+Number(s.rsi).toFixed(1)+'</strong></span>'+
        '<span class="gr">Hedef: <strong>'+Number(s.hedef).toFixed(2)+' TL</strong></span>'+
        '<span class="re">Stop: <strong>'+Number(s.stop).toFixed(2)+' TL</strong></span>'+
        '<span>Guven: <strong>%'+Number(s.guven*100).toFixed(0)+'</strong></span></div>';
    }
    document.getElementById('modal-icerik').innerHTML=h;
  }).catch(function(){document.getElementById('modal-icerik').innerHTML='<div class="es">Veri alinamadi.</div>';});
}

function sirketKapat(){document.getElementById('sirket-modal').style.display='none';}
document.addEventListener('keydown',function(e){if(e.key==='Escape')sirketKapat();});

function saatGuncelle(){
  try{
    const tr=new Date(new Date().toLocaleString('en-US',{timeZone:'Europe/Istanbul'}));
    const p=n=>n.toString().padStart(2,'0');
    const saat=p(tr.getHours())+':'+p(tr.getMinutes())+':'+p(tr.getSeconds());
    const sg=document.getElementById('son-guncelleme');
    if(sg) sg.textContent=saat;
    const gun=tr.getDay(), dak=tr.getHours()*60+tr.getMinutes();
    const acik=gun>=1&&gun<=5&&dak>=580&&dak<1080;
    const d=document.getElementById('borsa-durum');
    if(d&&!d.textContent.includes('YUKLEN')){
      const yeni=acik?'BORSA ACIK':'BORSA KAPALI';
      if(d.textContent!==yeni){d.textContent=yeni;d.className='badge'+(acik?'':' kapali');}
    }
  }catch(e){}
}
saatGuncelle();
setInterval(saatGuncelle,1000);

alarmGun(); portfoyGun(); veriCek(); setInterval(veriCek,10000);
</script>
</body>
</html>'''

# ── BACKEND ────────────────────────────────────────────

def borsa_acik_mi():
    from datetime import time as dtime
    now = datetime.now()
    if now.weekday() >= 5:   # Cumartesi=5, Pazar=6
        return False
    return dtime(10, 0) <= now.time() <= dtime(18, 10)

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

def kenar_verileri_cek():
    gruplar = {
        'kripto': {
            'BTC-USD':'Bitcoin','ETH-USD':'Ethereum',
            'BNB-USD':'BNB','XRP-USD':'XRP','SOL-USD':'Solana'
        },
        'doviz': {
            'USDTRY=X':'USD/TRY','EURTRY=X':'EUR/TRY',
            'GBPTRY=X':'GBP/TRY','JPYTRY=X':'JPY/TRY (x100)'
        },
        'emtia': {
            'GC=F':'Altın (oz)','SI=F':'Gümüş (oz)',
            'PL=F':'Platin (oz)','PA=F':'Paladyum (oz)'
        }
    }
    sonuc = {}
    for grup, isimler in gruplar.items():
        liste = []
        for sembol, isim in isimler.items():
            try:
                df = yf.Ticker(sembol).history(period='5d', interval='1d').dropna()
                if len(df) >= 2:
                    son     = float(df['Close'].iloc[-1])
                    onceki  = float(df['Close'].iloc[-2])
                    if grup == 'doviz' and 'JPY' in sembol:
                        son *= 100
                        onceki *= 100
                    degisim = (son - onceki) / onceki * 100 if onceki else 0.0
                elif len(df) == 1:
                    son = float(df['Close'].iloc[-1])
                    if grup == 'doviz' and 'JPY' in sembol:
                        son *= 100
                    degisim = 0.0
                else:
                    continue
                liste.append({'name': isim, 'symbol': sembol,
                              'fiyat': round(son, 2), 'degisim': round(degisim, 2)})
            except:
                pass
        sonuc[grup] = liste
    return sonuc

def haber_cek():
    if _feedparser is None:
        return []
    kaynaklar = [
        ("https://www.bloomberght.com/rss",                          "Bloomberg HT"),
        ("https://www.haberturk.com/rss/ekonomi.xml",                "Haberturk"),
        ("https://borsagundem.com/feed",                             "Borsa Gundem"),
        ("https://paraanaliz.com/feed/",                             "Para Analiz"),
        ("https://www.sabah.com.tr/rss/ekonomi.xml",                 "Sabah Ekonomi"),
        ("https://www.dunya.com/feeds/rss",                          "Dunya Gazetesi"),
        ("https://www.finansgundem.com/rss/haberler.xml",            "Finans Gundem"),
        ("https://feeds.reuters.com/reuters/businessNews",           "Reuters"),
    ]
    gruplar = []
    for url, kaynak in kaynaklar:
        try:
            feed = _feedparser.parse(url)
            for entry in (feed.entries or [])[:3]:
                baslik = (entry.get('title') or '').strip()
                if baslik:
                    gruplar.append({
                        'baslik': baslik[:90],
                        'kaynak': kaynak,
                        'link'  : entry.get('link', '#')
                    })
        except:
            pass
    return gruplar[:20]

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
    try:
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.iloc[:, 0] / 100
        df['Stoch_D'] = stoch.iloc[:, 1] / 100
    except Exception:
        df['Stoch_K'] = 0.5; df['Stoch_D'] = 0.5
    try:
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20) / 200
    except Exception:
        df['CCI'] = 0.0
    try:
        df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'], length=14) / 100
    except Exception:
        df['Williams_R'] = -0.5
    try:
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx_df.iloc[:, 0] / 100
    except Exception:
        df['ADX'] = 0.25
    try:
        obv = ta.obv(df['Close'], df['Volume'])
        df['OBV_Oran'] = obv.pct_change(5).fillna(0)
    except Exception:
        df['OBV_Oran'] = 0.0
    try:
        df['Getiri_30g'] = df['Close'].pct_change(30)
        df['Getiri_60g'] = df['Close'].pct_change(60)
    except Exception:
        df['Getiri_30g'] = 0.0; df['Getiri_60g'] = 0.0
    return df.dropna()

OZELLIKLER = [
    'RSI_Norm','RSI_f_Norm','MACD_Norm','MACD_hist',
    'BB_Konum','BB_genislik','MA5_Fark','MA20_Fark',
    'MA50_Fark','MA200_Fark','Trend_Guc',
    'Getiri_1g','Getiri_3g','Getiri_5g','Getiri_10g','Getiri_20g',
    'Hacim_Oran','ATR','Volatilite','Kanat','Govde','Yon',
    '52H_Yuzde','RSI_Trend','Hacim_Fiyat',
    'Stoch_K','Stoch_D','CCI','Williams_R','ADX','OBV_Oran',
    'Getiri_30g','Getiri_60g',
]

SIRKET_BILGI = {
    'AKBNK': {
        'ad': 'Akbank T.A.S.',
        'sektor': 'Bankacilik',
        'aciklama': "Akbank, 1948 yilinda kurulan Turkiye'nin en buyuk ozel bankalarindan biridir. Bireysel, kurumsal ve ticari bankacilik, kredi kartlari, yatirim bankaciligi ve sigortacilik alanlarinda hizmet vermektedir. 2024 sonu itibarita 17 milyondan fazla musteriye ulasmakatdir.",
    },
    'GARAN': {
        'ad': 'Garanti BBVA',
        'sektor': 'Bankacilik',
        'aciklama': "Garanti BBVA, 1946 yilinda kurulan ve Ispanyol BBVA'nin istiraki olan Turkiye'nin onde gelen ozel bankalarindan biridir. Bireysel ve kurumsal bankacilik ile dijital hizmetler alaninda faaliyet gostermektedir.",
    },
    'YKBNK': {
        'ad': 'Yapi ve Kredi Bankasi A.S.',
        'sektor': 'Bankacilik',
        'aciklama': "Yapi Kredi, 1944 yilinda kurulan Turkiye'nin ilk ozel bankalarindan biridir. Koc Holding ve UniCredit ortakligiyla faaliyet gosteren banka, bireysel, ticari ve kurumsal bankacilik hizmetleri sunmaktadir.",
    },
    'EKGYO': {
        'ad': 'Emlak Konut GYO A.S.',
        'sektor': 'Gayrimenkul',
        'aciklama': "Emlak Konut GYO, Turkiye'nin en buyuk gayrimenkul yatirim ortakligidir. TOKI istirakidir; konut projeleri gelistirme ve arsa satisi alanlarinda faaliyet gostermekte olup yuzlerce teslim edilmis projesi bulunmaktadir.",
    },
    'PGSUS': {
        'ad': 'Pegasus Hava Tasimaciligi A.S.',
        'sektor': 'Havacilik',
        'aciklama': "Pegasus Airlines, 1990 yilinda kurulan Turkiye merkezli dusuk maliyetli havayolu sirketidir. Ic hat ve uluslararasi hatlar dahil 100'den fazla destinasyona ucus gerceklestirmektedir.",
    },
    'TCELL': {
        'ad': 'Turkcell Iletisim Hizmetleri A.S.',
        'sektor': 'Telekomunikasyon',
        'aciklama': "Turkcell, Turkiye'nin lider mobil iletisim sirketidir. 1994 yilinda kurulan sirket, mobil, fiber internet, dijital servisler ve teknoloji cozumleri alanlarinda faaliyet gostermektedir.",
    },
    'SISE': {
        'ad': 'Turkiye Sise ve Cam Fabrikalari A.S.',
        'sektor': 'Cam & Kimya',
        'aciklama': "Sisecam, 1935 yilinda kurulan dunya genelinde faaliyet gosteren cam ureticisidir. Duzcam, ambalaj cami, cam ev esyasi ve kimyasallar alanlarinda Turkiye'nin en buyuk sanayi kuruluslarindan biridir.",
    },
    'FROTO': {
        'ad': 'Ford Otomotiv Sanayi A.S.',
        'sektor': 'Otomotiv',
        'aciklama': "Ford Otosan, Ford Motor Company ile Koc Holding'in ortakligiyla 1959 yilinda kurulan Turkiye'nin onde gelen otomotiv ureticisidir. Ticari arac, binek arac ve elektrikli arac uretimi gerceklestirmektedir.",
    },
}

def veri_hazirla(sembol):
    df = yf.Ticker(sembol).history(period="5y", interval="1d")
    df = df[['Open','High','Low','Close','Volume']]
    df.index = df.index.tz_localize(None)
    return ozellikler_ekle(df)

def model_egit(sembol):
    df = veri_hazirla(sembol)
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
                 class_weight='balanced', verbose=-1)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                 early_stopping=True, random_state=42)),
    ], voting='soft')
    model.fit(X_e, y[:bolme])
    return model, scaler, df

def guvenli_sayi(x, default=0):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except:
        return default

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

    def safe_list(series, dec=2):
        return [None if pd.isna(v) else round(float(v), dec) for v in series]

    rsi_vals = ta.rsi(s['Close'], length=14)

    # MA200 — hesapla (tum df'den, son 252 bar al)
    try:
        ma200_full = ta.sma(df['Close'], length=200)
        ma200 = ma200_full.iloc[-252:]
    except Exception:
        ma200 = pd.Series([None]*len(s))

    # Bollinger Bands (20,2)
    try:
        bb = ta.bbands(s['Close'], length=20)
        bb_ust = safe_list(bb.iloc[:, 2])   # BBU
        bb_alt = safe_list(bb.iloc[:, 0])   # BBL
    except Exception:
        bb_ust = bb_alt = [None]*len(s)

    # MACD (12,26,9)
    try:
        macd_df = ta.macd(s['Close'])
        macd_l = safe_list(macd_df.iloc[:, 0], 4)   # MACD line
        macd_h = safe_list(macd_df.iloc[:, 1], 4)   # Histogram
        macd_s = safe_list(macd_df.iloc[:, 2], 4)   # Signal
    except Exception:
        macd_l = macd_h = macd_s = [None]*len(s)

    # Stochastic (14,3,3)
    try:
        stoch_df = ta.stoch(s['High'], s['Low'], s['Close'])
        stoch_k = safe_list(stoch_df.iloc[:, 0])
        stoch_d = safe_list(stoch_df.iloc[:, 1])
    except Exception:
        stoch_k = stoch_d = [None]*len(s)

    return {
        'tarihler': [str(t)[:10] for t in s.index],
        'open'    : s['Open'].round(2).tolist(),
        'high'    : s['High'].round(2).tolist(),
        'low'     : s['Low'].round(2).tolist(),
        'close'   : s['Close'].round(2).tolist(),
        'ma20'    : safe_list(s['MA20']),
        'ma50'    : safe_list(s['MA50']),
        'ma200'   : safe_list(ma200),
        'bb_ust'  : bb_ust,
        'bb_alt'  : bb_alt,
        'macd'    : macd_l,
        'macd_h'  : macd_h,
        'macd_s'  : macd_s,
        'stoch_k' : stoch_k,
        'stoch_d' : stoch_d,
        'volume'  : s['Volume'].tolist(),
        'rsi'     : safe_list(rsi_vals, 1),
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
    db_tablolari_olustur()

    try:
        SISTEM_VERISI['kenar'] = kenar_verileri_cek()
        print("Kenar veriler hazir.")
    except Exception as e:
        print(f"Kenar veri hatasi: {e}")
    try:
        SISTEM_VERISI['haberler'] = haber_cek()
        print(f"Haberler hazir: {len(SISTEM_VERISI['haberler'])} haber")
    except Exception as e:
        print(f"Haber hatasi: {e}")

    print("\nModeller yukleniyor / egitiliyor...")
    for s in HISSELER:
        model, scaler = model_db_yukle(s)
        if model is not None:
            try:
                df = veri_hazirla(s)
                SISTEM_VERISI['modeller'][s] = (model, scaler, df)
                print(f"  {s} DB'den yuklendi ✅")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"  {s} veri hazirlama hatasi: {e}")
        for deneme in range(3):
            try:
                print(f"  {s} egitiliyor... (deneme {deneme+1})")
                model, scaler, df = model_egit(s)
                SISTEM_VERISI['modeller'][s] = (model, scaler, df)
                model_db_kaydet(s, model, scaler)
                print(f"  {s} ✅")
                break
            except Exception as e:
                print(f"  {s} ❌ deneme {deneme+1}: {e}")
                if deneme < 2:
                    time.sleep(15)
        time.sleep(3)

    print("Modeller hazir!\n")
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

            sinyalleri_db_kaydet(sinyaller)
            track_record_db_ekle(sinyaller)
            alarmlar_db_kontrol(sinyaller)
            sinyallerden_track_record_guncelle()

            try:
                SISTEM_VERISI['kenar'] = kenar_verileri_cek()
            except Exception as e:
                print(f"Kenar veri hatasi: {e}")

            try:
                SISTEM_VERISI['haberler'] = haber_cek()
            except Exception as e:
                print(f"Haber hatasi: {e}")

            print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(sinyaller)} sinyal guncellendi.")
        except Exception as e:
            print(f"Guncelleme hatasi: {e}")

        time.sleep(GUNCELLEME)

@app.route('/')
def index():
    return render_template_string(HTML, hisseler=HISSELER)

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

    tr_data = track_record_db_oku() or track_record_oku()
    data = {
        'hazir'         : SISTEM_VERISI['hazir'],
        'sinyaller'     : SISTEM_VERISI['sinyaller'],
        'piyasa'        : SISTEM_VERISI['piyasa'],
        'grafik_verisi' : SISTEM_VERISI['grafik_verisi'],
        'borsa_acik'    : borsa_acik_mi(),
        'son_guncelleme': SISTEM_VERISI['son_guncelleme'],
        'track_record'  : tr_data,
        'kenar'         : SISTEM_VERISI['kenar'],
        'haberler'      : SISTEM_VERISI['haberler'],
    }
    return jsonify(json_temizle(data))

@app.route('/api/hisse/<sembol>')
def api_hisse(sembol):
    sembol_is = sembol + '.IS' if not sembol.endswith('.IS') else sembol
    bilgi = SIRKET_BILGI.get(sembol, {'ad': sembol, 'sektor': '-', 'aciklama': ''})
    fin = {}
    try:
        info = yf.Ticker(sembol_is).info or {}
        fin = {
            'piyasaDegeri' : info.get('marketCap'),
            'fk'           : info.get('trailingPE'),
            'pd_dd'        : info.get('priceToBook'),
            'yuksek52'     : info.get('fiftyTwoWeekHigh'),
            'dusuk52'      : info.get('fiftyTwoWeekLow'),
            'temettu'      : info.get('dividendYield'),
            'calisanSayisi': info.get('fullTimeEmployees'),
            'beta'         : info.get('beta'),
        }
    except:
        pass
    sinyal = next((s for s in SISTEM_VERISI['sinyaller']
                   if s['sembol'].replace('.IS', '') == sembol), None)
    return jsonify(json_temizle({
        'sembol'  : sembol,
        'ad'      : bilgi['ad'],
        'sektor'  : bilgi['sektor'],
        'aciklama': bilgi['aciklama'],
        'finans'  : fin,
        'sinyal'  : sinyal,
    }))

@app.route('/api/portfoy/<sid>', methods=['GET'])
def api_portfoy_oku(sid):
    data = portfoy_db_oku(sid)
    return jsonify(data if data is not None else [])

@app.route('/api/portfoy/<sid>', methods=['POST'])
def api_portfoy_kaydet(sid):
    d = request.get_json() or {}
    ok = portfoy_db_kaydet(sid, d.get('sembol', ''),
                            float(d.get('adet', 0)), float(d.get('maliyet', 0)))
    return jsonify({'ok': ok})

@app.route('/api/portfoy/<sid>/<sembol>', methods=['DELETE'])
def api_portfoy_sil(sid, sembol):
    ok = portfoy_db_sil(sid, sembol)
    return jsonify({'ok': ok})

@app.route('/api/alarmlar/<sid>', methods=['GET'])
def api_alarmlar_oku(sid):
    data = alarmlar_db_oku(sid)
    return jsonify(data if data is not None else [])

@app.route('/api/alarmlar/<sid>', methods=['POST'])
def api_alarm_ekle(sid):
    d = request.get_json() or {}
    new_id = alarm_db_ekle(sid, d.get('sembol', ''), d.get('yon', 'above'),
                            float(d.get('fiyat', 0)))
    return jsonify({'ok': new_id is not None, 'id': new_id})

@app.route('/api/alarmlar/<sid>/<int:alarm_id>', methods=['DELETE'])
def api_alarm_sil(sid, alarm_id):
    ok = alarm_db_sil(sid, alarm_id)
    return jsonify({'ok': ok})

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  LIDYA BORSA PLATFORMU BAŞLATILIYOR")
    print(f"  Tarayıcıda aç: http://localhost:{PORT}")
    print("  Durdurmak için CTRL+C")
    print("="*55)

    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
