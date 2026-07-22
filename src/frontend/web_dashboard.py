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
from datetime import datetime, timezone, timedelta
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
try:
    import jwt as _pyjwt
    _PYJWT_OK = True
except ImportError:
    _PYJWT_OK = False
try:
    from src.data_ingestion.haber_analizi import google_news_rss as _google_news_rss
    from src.data_ingestion.haber_analizi import yahoo_news_cek as _yahoo_news_cek
    _HABER_ANALIZI_OK = True
except Exception:
    _HABER_ANALIZI_OK = False

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

# ─── Supabase (kullanici hesabi) ────────────────────────
SUPABASE_URL        = os.environ.get('SUPABASE_URL', '')
SUPABASE_ANON_KEY    = os.environ.get('SUPABASE_ANON_KEY', '')
SUPABASE_JWT_SECRET  = os.environ.get('SUPABASE_JWT_SECRET')  # legacy HS256 yedek yol icin
_jwks_client = None

def _jwks_al():
    global _jwks_client
    if _jwks_client is None and SUPABASE_URL:
        _jwks_client = _pyjwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    return _jwks_client

def dogrulanan_kullanici_id():
    """Authorization: Bearer <supabase_jwt> header'i varsa dogrulayip user_id (sub) doner.
    Header yoksa None doner (anonim akis icin normal durum). Gecersiz token'da da None
    doner, ama cagiran route bunu ayirt edebilsin diye ikinci bir bayrak dondurulur.
    Supabase yeni projelerde ES256 (JWKS ile) imzali oturum token'lari verir; JWKS
    ucu once denenir, olmazsa (eski projeler icin) HS256+SUPABASE_JWT_SECRET denenir."""
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return None, False
    if not _PYJWT_OK:
        return None, True
    token = auth[7:]
    try:
        client = _jwks_al()
        if client is not None:
            signing_key = client.get_signing_key_from_jwt(token)
            payload = _pyjwt.decode(
                token, signing_key.key, algorithms=['ES256', 'RS256'],
                audience='authenticated'
            )
            return payload.get('sub'), True
    except Exception as e:
        print(f"[Auth] JWKS dogrulama hatasi, HS256 deneniyor: {e}")
    if SUPABASE_JWT_SECRET:
        try:
            payload = _pyjwt.decode(
                token, SUPABASE_JWT_SECRET, algorithms=['HS256'],
                audience='authenticated'
            )
            return payload.get('sub'), True
        except Exception as e:
            print(f"[Auth] Token dogrulama hatasi: {e}")
    return None, True

def sid_yetki_hatasi(sid):
    """Portfoy/alarm route'larinin basinda cagrilir. Authorization header'i yoksa
    None doner (anonim akisa devam et, mevcut davranis). Header varsa dogrular;
    gecersizse veya sid ile eslesmiyorsa (jsonify(...), status) doner — route bunu
    direkt 'return' etmeli. Uyusursa None doner (devam et)."""
    user_id, header_vardi = dogrulanan_kullanici_id()
    if not header_vardi:
        return None
    if user_id is None:
        return jsonify({'hata': 'Gecersiz veya suresi dolmus oturum'}), 401
    if user_id != sid:
        return jsonify({'hata': 'Bu veriye erisim yetkiniz yok'}), 403
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
        fm = {s['sembol'].replace('.IS',''): s for s in sinyaller}
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id,sembol,yon,fiyat FROM lidya_alarmlar WHERE tetiklendi=FALSE")
            for alarm in cur.fetchall():
                sinyal = fm.get(alarm['sembol'])
                if sinyal is None:
                    continue
                g     = sinyal.get('fiyat')
                rsi   = sinyal.get('rsi')
                karar = sinyal.get('karar')
                yon   = alarm['yon']
                hit = False
                if yon == 'above' and g is not None and alarm['fiyat'] is not None:
                    hit = g >= float(alarm['fiyat'])
                elif yon == 'below' and g is not None and alarm['fiyat'] is not None:
                    hit = g <= float(alarm['fiyat'])
                elif yon == 'rsi_ob' and rsi is not None:
                    hit = rsi > 70
                elif yon == 'rsi_os' and rsi is not None:
                    hit = rsi < 30
                elif yon == 'sinyal_al':
                    hit = karar == 'AL'
                elif yon == 'sinyal_sat':
                    hit = karar == 'SAT'
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
<meta name="description" content="LIDYA, BIST hisseleri icin teknik gosterge ve haber duyarlılığı verilerini şeffaf şekilde sunan deneysel bir bilgi panelidir. Yatırım tavsiyesi degildir.">
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
.hesap-cta{background:linear-gradient(90deg,#ede9ff 0%,#a78bfa 55%,#ec4899 100%);color:#0c0a17;border:none;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:800;letter-spacing:.3px;cursor:pointer;box-shadow:0 0 14px rgba(167,139,250,.45);transition:transform .18s,filter .18s;white-space:nowrap;}
.hesap-cta:hover{transform:scale(1.05);filter:brightness(1.1);}
.tnav{background:var(--sf);border-bottom:1px solid var(--bd);display:flex;padding:0 20px;overflow-x:auto;position:sticky;top:52px;z-index:100;}
.tnav::-webkit-scrollbar{display:none;}
.tb{padding:11px 18px;border:none;background:none;color:var(--mu);cursor:pointer;font-size:13px;font-weight:500;border-bottom:2px solid transparent;white-space:nowrap;transition:all .2s;}
.tb:hover{color:var(--tx);}
.tb.active{color:var(--ac);border-bottom-color:var(--ac);}
.tp{display:none;} .tp.active{display:block;}
.uyari{position:sticky;top:88px;z-index:90;background:#f59e0b14;border-bottom:1px solid #f59e0b28;color:var(--ye);font-size:11px;line-height:16px;padding:7px 20px;text-align:center;}
.uyari b{color:var(--tx);}
.lay{display:flex;min-height:calc(100vh - 118px);}
.sb{width:228px;min-width:228px;background:var(--sf);overflow-y:auto;height:calc(100vh - 118px);position:sticky;top:118px;flex-shrink:0;}
.sb-l{border-right:1px solid var(--bd);}
.sb-r{border-left:1px solid var(--bd);}
.sb::-webkit-scrollbar{width:3px;}.sb::-webkit-scrollbar-thumb{background:var(--bd);}
.main{flex:1;min-width:0;}
.sbt{font-size:13px;font-weight:800;color:var(--tx);text-transform:uppercase;letter-spacing:.1em;padding:11px 12px 7px;border-bottom:1px solid var(--bd);text-shadow:0 1px 0 rgba(255,255,255,.2),0 2px 1px rgba(0,0,0,.5),0 3px 4px rgba(0,0,0,.55),0 0 14px rgba(167,139,250,.35);}
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
.modal-sekmeler{display:flex;gap:0;margin-bottom:16px;border-bottom:1px solid var(--bd);}
.modal-sekme{padding:7px 18px;border:none;background:none;color:var(--mu);font-size:12px;font-weight:600;cursor:pointer;border-bottom:2px solid transparent;transition:all .18s;}
.modal-sekme:hover{color:var(--tx);}
.modal-sekme.active{color:var(--ac);border-bottom-color:var(--ac);}
.fin-tablo{width:100%;border-collapse:collapse;font-size:11px;}
.fin-tablo th{color:var(--mu);font-weight:600;padding:5px 7px;text-align:right;border-bottom:1px solid var(--bd);white-space:nowrap;}
.fin-tablo th:first-child{text-align:left;}
.fin-tablo td{padding:5px 7px;text-align:right;border-bottom:1px solid rgba(46,42,72,.35);white-space:nowrap;}
.fin-tablo td:first-child{text-align:left;color:var(--mu);}
.fin-tablo tbody tr:last-child td{font-weight:700;color:var(--tx);border-top:1px solid var(--bd);}
.fin-2kol{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}
.fin-grafik-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:4px;}
.fin-grafik-baslik{font-size:10px;color:var(--mu);margin-bottom:4px;text-align:center;}
.oran-serit{display:flex;gap:0;flex-wrap:wrap;background:var(--sf);border-radius:8px;margin-bottom:14px;overflow:hidden;border:1px solid var(--bd);}
.oran-item{flex:1;min-width:80px;padding:9px 8px;text-align:center;border-right:1px solid var(--bd);}
.oran-item:last-child{border-right:none;}
.oran-l{font-size:9px;color:var(--mu);text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px;}
.oran-v{font-size:13px;font-weight:700;color:var(--tx);}
@media(max-width:640px){.fin-2kol{grid-template-columns:1fr;}.fin-grafik-grid{grid-template-columns:1fr;}.modal-kart{padding:14px;}}
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
  .hesap-cta{padding:5px 11px;font-size:11px;}
}
@media(max-width:420px){
  #son-guncelleme{display:none;}
}
/* Ekonomik Takvim */
.tak-filtre{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;}
.tak-chip{padding:3px 11px;border-radius:20px;font-size:11px;border:1px solid var(--bd);background:none;color:var(--mu);cursor:pointer;transition:all .15s;}
.tak-chip:hover{border-color:var(--ac);color:var(--tx);}
.tak-chip.active{background:rgba(167,139,250,.14);border-color:var(--ac);color:var(--ac);}
.tak-item{display:flex;gap:12px;align-items:flex-start;padding:11px 0;border-bottom:1px solid rgba(46,42,72,.3);}
.tak-item:last-child{border-bottom:none;}
.tak-tarih-blok{min-width:52px;text-align:center;padding-top:2px;}
.tak-gun{font-size:22px;font-weight:700;line-height:1.05;}
.tak-ay{font-size:9px;color:var(--mu);text-transform:uppercase;letter-spacing:.06em;}
.tak-badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px;}
.tak-kart-ic{flex:1;}
.tak-basl{font-size:12px;font-weight:600;color:var(--tx);line-height:1.4;}
.tak-alt{font-size:10px;color:var(--mu);margin-top:3px;line-height:1.45;}
.tak-gecti .tak-basl{color:var(--cf);}
.tak-gecti .tak-gun{color:var(--mu)!important;}
.tak-gecti{opacity:.5;}
.tak-aktif .tak-kart-ic{border-left:2px solid var(--ac);padding-left:9px;}
.tak-geri-sayim{font-size:10px;font-weight:600;margin-top:5px;}
/* İndikatör Rehberi Kartları */
.ind-kart-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:9px;margin-top:14px;}
.ind-kart{background:var(--ch);border:1px solid var(--bd);border-radius:9px;padding:12px;transition:border-color .2s;}
.ind-kart:hover{border-color:rgba(167,139,250,.3);}
.ind-k-kat{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;padding:2px 7px;border-radius:3px;display:inline-block;margin-bottom:7px;}
.ind-k-isim{font-size:12px;font-weight:700;color:var(--tx);margin-bottom:5px;}
.ind-k-deger{font-size:18px;font-weight:700;line-height:1.2;margin-bottom:6px;min-height:22px;}
.ind-k-durum{display:inline-block;padding:2px 9px;border-radius:12px;font-size:10px;font-weight:600;margin-bottom:8px;}
.ind-k-acik{font-size:10px;color:var(--mu);line-height:1.5;margin-bottom:5px;}
.ind-k-hint{font-size:9px;color:var(--cf);border-top:1px solid rgba(46,42,72,.4);padding-top:5px;line-height:1.5;}
/* Alarm Merkezi */
.alarm-tur-btn{padding:3px 10px;border-radius:5px;border:1px solid var(--bd);background:none;color:var(--mu);font-size:11px;cursor:pointer;transition:all .18s;}
.alarm-tur-btn:hover{border-color:var(--ac);color:var(--tx);}
.alarm-tur-btn.active{background:rgba(167,139,250,.14);border-color:var(--ac);color:var(--ac);}
.notif-item{padding:9px 12px;border-bottom:1px solid rgba(46,42,72,.3);display:flex;gap:10px;align-items:flex-start;}
.notif-item:last-child{border-bottom:none;}
.notif-dot{width:8px;height:8px;border-radius:50%;margin-top:4px;flex-shrink:0;}
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
    <button class="hesap-cta" id="hesap-cta" onclick="tabAc('hesap', null)">Kayıt Ol</button>
    <span id="son-guncelleme" style="font-size:10px;color:var(--mu)"></span>
  </div>
</div>

<nav class="tnav">
  <button class="tb active" onclick="tabAc('sinyaller',this)">Sinyaller</button>
  <button class="tb" onclick="tabAc('grafik',this)">Grafik</button>
  <button class="tb" onclick="tabAc('trackrecord',this)">Track Record</button>
  <button class="tb" onclick="tabAc('portfoy',this)">Portföy</button>
  <button class="tb" onclick="tabAc('takvim',this)">Takvim</button>
  <button class="tb" onclick="tabAc('alarmlar',this)">Alarmlar</button>
</nav>

<div class="uyari">⚠ Bu platform <b>deneyseldir</b>. Gösterilen veriler <b>yatırım tavsiyesi değildir</b> — model çıktılarının geçmiş gerçek başarı oranını <b>Track Record</b> sekmesinde görebilirsiniz.</div>

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
      <div class="ss" id="sinyal-ozet">Pozitif / Negatif / Nötr</div>
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
      <div class="ss" style="opacity:.7">3 sınıflı rastgele tahmin: ~%33</div>
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
      <button class="ind-btn" id="btn-cetvel" onclick="aracSec('cetvel',this)" title="Cetvel — iki nokta arası % değişim ve mum sayısı">📏 Cetvel</button>
      <button class="ind-btn" onclick="isinTemizle()" title="Tüm çizilen ışınları temizle" style="color:var(--re);border-color:rgba(239,68,68,.3)">✕ Temizle</button>
      <div class="ind-sep"></div>
      <button class="ind-btn" id="btn-cizim" onclick="cizimToggle(this)" title="Serbest çizim (trend çizgisi, dikdörtgen, daire)">✏ Serbest</button>
    </div>
    <div id="grafik-alan" style="height:370px"></div>
    <div id="hacim-alan" style="height:85px;margin-top:2px"></div>
    <div id="rsi-alan" style="height:85px;margin-top:2px"></div>
    <div id="macd-alan" style="display:none;height:85px;margin-top:2px"></div>
    <div id="stoch-alan" style="display:none;height:85px;margin-top:2px"></div>
    <div id="ind-rehber" style="margin-top:14px;border-top:1px solid var(--bd);padding-top:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px">
        <span style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--mu)">İndikatör Rehberi</span>
        <span style="font-size:9px;color:var(--cf)">Seçili hisse için anlık değerler</span>
      </div>
      <div class="ind-kart-grid" id="ind-kart-grid"></div>
    </div>
  </div>
</div>
</div>

<!-- TAB: Track Record -->
<div id="tab-trackrecord" class="tp">
<div class="con">
  <div class="trg">
    <div class="trs"><div class="trl">Toplam Sinyal</div><div class="trv" id="tr-toplam">—</div></div>
    <div class="trs"><div class="trl">Tamamlanan</div><div class="trv" id="tr-tamamlanan">—</div></div>
    <div class="trs"><div class="trl">Başarı Oranı</div><div class="trv" id="tr-basari">—</div><div style="font-size:9px;color:var(--mu);margin-top:2px">3 sınıflı rastgele: ~%33</div></div>
    <div class="trs"><div class="trl">Ort. Kar/Zarar</div><div class="trv" id="tr-ort-kar">—</div></div>
  </div>
  <div class="card">
    <div class="ctit">Kümülatif Performans</div>
    <div id="perf-grafik" style="height:220px"></div>
  </div>
  <div class="card" id="tr-portfoy-kart" style="display:none">
    <div class="ctit">Portföyünüzdeki Hisseler</div>
    <div id="track-record-portfoy"><div class="es">Yükleniyor...</div></div>
  </div>
  <div class="card">
    <div class="ctit" id="tr-genel-baslik">Sinyal Geçmişi</div>
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
</div>
</div>

<!-- TAB: Ekonomik Takvim -->
<div id="tab-takvim" class="tp">
<div class="con">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:11px">
      <div class="ctit" style="margin:0">2026 Ekonomik Takvim</div>
      <div id="tak-sonraki" style="font-size:11px;color:var(--ac);font-weight:600"></div>
    </div>
    <div class="tak-filtre">
      <button class="tak-chip active" onclick="takFiltre(\'hepsi\',this)">Tümü</button>
      <button class="tak-chip" onclick="takFiltre(\'tcmb\',this)">TCMB PPK</button>
      <button class="tak-chip" onclick="takFiltre(\'tuik\',this)">TÜİK</button>
      <button class="tak-chip" onclick="takFiltre(\'fed\',this)">FED</button>
      <button class="tak-chip" onclick="takFiltre(\'bilanco\',this)">Bilanço</button>
    </div>
    <div id="takvim-listesi"><div class="es">Yükleniyor...</div></div>
  </div>
</div>
</div>

<!-- TAB: Alarmlar -->
<div id="tab-alarmlar" class="tp">
<div class="con">
  <div class="card">
    <div class="ctit">Alarm Ekle</div>
    <div class="pf">
      <input class="pi" id="al2-sembol" placeholder="Sembol (örn: AKBNK)" list="hl2" style="max-width:140px">
      <select class="sel" id="al2-tur" onchange="alarmTurGun()">
        <option value="above">Fiyat Üstüne Çıkınca ↑</option>
        <option value="below">Fiyat Altına Düşünce ↓</option>
        <option value="rsi_ob">RSI Aşırı Alım (RSI &gt; 70)</option>
        <option value="rsi_os">RSI Aşırı Satım (RSI &lt; 30)</option>
        <option value="sinyal_al">Gösterge Pozitife Dönünce</option>
        <option value="sinyal_sat">Gösterge Negatife Dönünce</option>
      </select>
      <input class="pi" id="al2-fiyat" placeholder="Hedef Fiyat ₺" type="number" style="max-width:135px">
      <button class="ba" onclick="alarm2Ekle()">+ Alarm</button>
    </div>
    <div id="alarm2-listesi"><div class="es">Henüz alarm yok.</div></div>
  </div>
  <div class="card" id="tetik-kart" style="display:none">
    <div class="ctit" style="color:var(--gr)">✓ Tetiklenen Alarmlar</div>
    <div id="tetik-listesi"></div>
  </div>
  <div class="card">
    <div class="ctit">Tarayıcı Bildirimleri</div>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
      <div id="bildirim-durum" style="font-size:12px;color:var(--mu)">Kontrol ediliyor...</div>
      <button class="ba" id="bildirim-btn" onclick="bildirimIzniAl()" style="display:none">İzin Ver</button>
    </div>
    <div style="font-size:11px;color:var(--mu)">Alarmlar tetiklendiğinde masaüstü bildirimi alırsınız. Tarayıcınız açık olmalıdır.</div>
  </div>
</div>
</div>

<!-- TAB: Hesap -->
<div id="tab-hesap" class="tp">
<div class="con">
  <div id="hesap-cikis-yapilmis" class="card" style="display:none">
    <div class="ctit">Hesabım</div>
    <div style="font-size:13px;color:var(--tx);margin-bottom:12px">Giriş yapıldı: <strong id="hesap-eposta"></strong></div>
    <div style="font-size:11px;color:var(--mu);margin-bottom:12px">Portföyünüz ve alarmlarınız artık bu hesaba bağlı — hangi cihazdan giriş yaparsanız yapın aynı veriyi görürsünüz.</div>
    <button class="ba" style="background:var(--sf);color:var(--tx);border:1px solid var(--bd)" onclick="hesapCikisYap()">Çıkış Yap</button>
  </div>
  <div id="hesap-giris-yapilmamis" class="card">
    <div class="ctit">Kayıt Ol / Giriş Yap</div>
    <div style="font-size:11px;color:var(--mu);margin-bottom:12px">Hesap oluşturursanız portföyünüz ve alarmlarınız kalıcı olarak bu hesaba kaydedilir. Hesap açmadan da (anonim) kullanmaya devam edebilirsiniz.</div>
    <div class="pf">
      <input class="pi" id="hesap-eposta-giris" placeholder="E-posta" type="email" style="max-width:220px">
      <input class="pi" id="hesap-sifre-giris" placeholder="Şifre" type="password" style="max-width:160px">
    </div>
    <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
      <button class="ba" onclick="hesapGirisYap()">Giriş Yap</button>
      <button class="ba" style="background:var(--sf);color:var(--tx);border:1px solid var(--bd)" onclick="hesapKayitOl()">Kayıt Ol</button>
    </div>
    <div id="hesap-mesaj" style="font-size:12px;margin-top:10px"></div>
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
  <button class="bni" id="bni-takvim" onclick="tabMob('takvim',this)">
    <svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
    Takvim
  </button>
  <button class="bni" id="bni-alarmlar" onclick="tabMob('alarmlar',this)">
    <svg viewBox="0 0 24 24"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
    Alarmlar
  </button>
</nav>

<div id="sirket-modal" class="modal-bg" style="display:none" onclick="if(event.target===this)sirketKapat()">
  <div class="modal-kart" id="modal-icerik"><div class="es">Yukleniyor...</div></div>
</div>

<script>
const CBG='#07050e', CGR='#1c1830', CFN='#5e5a7a';
const KARAR_ETIKET={AL:'POZİTİF',SAT:'NEGATİF',BEKLE:'NÖTR'};
const SUPABASE_URL='{{ supabase_url }}', SUPABASE_ANON_KEY='{{ supabase_anon_key }}';
let grafikVerisi={}, trackData=null, period=90, fiyatlar={}, sinyalVerisi=[];
let indAktif={ma200:false,bb:false,macd:false,stoch:false,pivotlar:false,sinyaller:true};
let cizimModu=false, aktifArac=null, isinlar=[], cetvelIlkNokta=null;

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

// ─── Hesap (Supabase Auth) ──────────────────────────────
function _authOku(){
  try{ return JSON.parse(localStorage.getItem('lidya_auth')||'null'); }catch(e){ return null; }
}
function _authYaz(a){ localStorage.setItem('lidya_auth', JSON.stringify(a)); }
function _authSil(){ localStorage.removeItem('lidya_auth'); }

function aktifSid(){
  const a=_authOku();
  return a ? a.user_id : SID;
}

async function authYenile(auth){
  try{
    const r=await fetch(SUPABASE_URL+'/auth/v1/token?grant_type=refresh_token',{
      method:'POST',
      headers:{'Content-Type':'application/json','apikey':SUPABASE_ANON_KEY},
      body:JSON.stringify({refresh_token:auth.refresh_token})
    });
    const d=await r.json();
    if(!r.ok || !d.access_token){ _authSil(); return null; }
    const yeni={access_token:d.access_token, refresh_token:d.refresh_token,
      user_id:d.user.id, email:d.user.email, expires_at:Date.now()+d.expires_in*1000};
    _authYaz(yeni);
    return yeni;
  }catch(e){ return auth; }
}

async function fetchYetkili(url,opts){
  opts=opts||{};
  let auth=_authOku();
  if(auth){
    if(Date.now() > auth.expires_at-60000) auth=await authYenile(auth);
    if(auth) opts.headers=Object.assign({},opts.headers||{},{'Authorization':'Bearer '+auth.access_token});
  }
  return fetch(url,opts);
}

async function hesabaVeriTasi(yeniUserId){
  if(SID===yeniUserId) return;
  try{
    const [pR,aR]=await Promise.all([fetch('/api/portfoy/'+SID), fetch('/api/alarmlar/'+SID)]);
    const [pD,aD]=[await pR.json(), await aR.json()];
    for(const p of (pD||[])){
      await fetchYetkili('/api/portfoy/'+yeniUserId,{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({sembol:p.sembol, adet:p.adet, maliyet:p.maliyet})});
    }
    for(const a of (aD||[])){
      await fetchYetkili('/api/alarmlar/'+yeniUserId,{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({sembol:a.sembol, yon:a.yon, fiyat:a.fiyat})});
    }
    if((pD&&pD.length)||(aD&&aD.length)){
      const m=document.getElementById('hesap-mesaj');
      if(m){ m.className='gr'; m.textContent='Mevcut portföy/alarm verileriniz hesabınıza taşındı.'; }
    }
  }catch(e){ console.warn('Veri tasima hatasi', e); }
}

function hesapUIGuncelle(){
  const a=_authOku();
  const girisli=document.getElementById('hesap-cikis-yapilmis');
  const girissiz=document.getElementById('hesap-giris-yapilmamis');
  const cta=document.getElementById('hesap-cta');
  if(a){
    if(girisli) girisli.style.display='';
    if(girissiz) girissiz.style.display='none';
    const e=document.getElementById('hesap-eposta'); if(e) e.textContent=a.email;
    if(cta) cta.textContent='Hesabım';
  }else{
    if(girisli) girisli.style.display='none';
    if(girissiz) girissiz.style.display='';
    if(cta) cta.textContent='Kayıt Ol';
  }
}

async function hesapGirisBasarili(d){
  const auth={access_token:d.access_token, refresh_token:d.refresh_token,
    user_id:d.user.id, email:d.user.email, expires_at:Date.now()+d.expires_in*1000};
  const eskiSid=SID;
  _authYaz(auth);
  _lsPortfoyKaydet([]); _lsAlarmlarKaydet([]);
  hesapUIGuncelle();
  await hesabaVeriTasi(auth.user_id);
  portfoyGun(); alarmGun(); alarm2Gun();
}

async function hesapKayitOl(){
  const email=document.getElementById('hesap-eposta-giris').value.trim();
  const sifre=document.getElementById('hesap-sifre-giris').value;
  const m=document.getElementById('hesap-mesaj');
  if(!email||!sifre){ m.className='re'; m.textContent='E-posta ve şifre gerekli.'; return; }
  try{
    const r=await fetch(SUPABASE_URL+'/auth/v1/signup',{
      method:'POST',
      headers:{'Content-Type':'application/json','apikey':SUPABASE_ANON_KEY},
      body:JSON.stringify({email,password:sifre})
    });
    const d=await r.json();
    if(!r.ok){ m.className='re'; m.textContent=d.error_description||d.msg||'Kayıt başarısız.'; return; }
    if(d.access_token){ await hesapGirisBasarili(d); }
    else { m.className='ye'; m.textContent='Kayıt alındı — e-postanızı kontrol edip doğrulama linkine tıklayın, sonra giriş yapın.'; }
  }catch(e){ m.className='re'; m.textContent='Bağlantı hatası: '+e.message; }
}

async function hesapGirisYap(){
  const email=document.getElementById('hesap-eposta-giris').value.trim();
  const sifre=document.getElementById('hesap-sifre-giris').value;
  const m=document.getElementById('hesap-mesaj');
  if(!email||!sifre){ m.className='re'; m.textContent='E-posta ve şifre gerekli.'; return; }
  try{
    const r=await fetch(SUPABASE_URL+'/auth/v1/token?grant_type=password',{
      method:'POST',
      headers:{'Content-Type':'application/json','apikey':SUPABASE_ANON_KEY},
      body:JSON.stringify({email,password:sifre})
    });
    const d=await r.json();
    if(!r.ok||!d.access_token){ m.className='re'; m.textContent=d.error_description||d.msg||'Giriş başarısız.'; return; }
    m.className='gr'; m.textContent='Giriş yapıldı.';
    await hesapGirisBasarili(d);
  }catch(e){ m.className='re'; m.textContent='Bağlantı hatası: '+e.message; }
}

function hesapCikisYap(){
  _authSil();
  _lsPortfoyKaydet([]); _lsAlarmlarKaydet([]);
  hesapUIGuncelle();
  portfoyGun(); alarmGun(); alarm2Gun();
}

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
  if(id==='portfoy'){portfoyGun();alarmGun();}
  if(id==='takvim') takvimYukle();
  if(id==='alarmlar'){alarm2Gun();bildirimDurumGun();alarmTurGun();}
  if(id==='hesap') hesapUIGuncelle();
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
  document.getElementById('sinyal-ozet').innerHTML='<span class="gr">'+al+' Pozitif</span> / <span class="re">'+sat+' Negatif</span> / <span class="ye">'+bk+' Nötr</span>';

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
    const du=b.al>b.sat?'Pozitif agirlikli':b.sat>b.al?'Negatif agirlikli':'Karisik';
    h+='<div class="ek"><div class="enm">'+sek+'</div><div class="ev '+r+'">'+(ort>=0?'+':'')+ort.toFixed(1)+'%</div><div class="ed">'+du+'</div></div>';
  });
  document.getElementById('sektor-grid').innerHTML=h||'<div class="es">Yukleniyor...</div>';
}

function tabloGun(sn){
  if(!sn||!sn.length){document.getElementById('sinyal-tablo-alani').innerHTML='<div class="es">Modeller egitiliyor...</div>';return;}
  let h='<table class="t"><thead><tr><th>Hisse</th><th>Fiyat</th><th>Degisim</th><th>RSI</th><th>Gosterge</th><th>Guven</th><th>Ref. Direnc</th><th>Ref. Destek</th></tr></thead><tbody>';
  sn.forEach(s=>{
    const dr=s.degisim>=0?'gr':'re', di=s.degisim>=0?'+':'';
    const kc=s.karar==='AL'?'al':s.karar==='SAT'?'sat':'bekle';
    const rc=s.karar==='AL'?'al-r':s.karar==='SAT'?'sat-r':'bk-r';
    const gp=(s.guven*100).toFixed(0), rr=s.rsi<40?'gr':s.rsi>60?'re':'ye';
    h+='<tr class="'+rc+'"><td style="font-weight:700">'+s.sembol.replace('.IS','')+'</td>'+
      '<td>'+Number(s.fiyat).toFixed(2)+' TL</td>'+
      '<td class="'+dr+'">'+di+Number(s.degisim).toFixed(2)+'%</td>'+
      '<td class="'+rr+'">'+Number(s.rsi).toFixed(1)+'</td>'+
      '<td><span class="pill '+kc+'">'+KARAR_ETIKET[s.karar]+'</span></td>'+
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
  cetvelIlkNokta=null;
  ['btn-h-isin','btn-v-isin','btn-cetvel'].forEach(id=>{
    const b=document.getElementById(id);
    if(b) b.classList.remove('cizim-aktif');
  });
  const g=document.getElementById('grafik-alan');
  if(g) g.style.cursor='';
}

function aracSec(arac,btn){
  if(aktifArac===arac){_aracKapat();return;}
  aktifArac=arac;
  cetvelIlkNokta=null;
  ['btn-h-isin','btn-v-isin','btn-cetvel'].forEach(id=>{
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
    } else if(aktifArac==='cetvel'){
      const yv=parseFloat(y);
      if(!cetvelIlkNokta){
        cetvelIlkNokta={x:x,y:yv};
      } else {
        const x1=cetvelIlkNokta.x, y1=cetvelIlkNokta.y, x2=x, y2=yv;
        const yuzde=(y2-y1)/y1*100;
        const hs=document.getElementById('hisse-sec');
        const cv=hs?grafikVerisi[hs.value]:null;
        const tarArr=(cv&&cv.tarihler)?cv.tarihler:null;
        let mumTxt='';
        if(tarArr){
          const i1=tarArr.indexOf(x1), i2=tarArr.indexOf(x2);
          if(i1>=0&&i2>=0) mumTxt=' · '+Math.abs(i2-i1)+' mum';
        }
        isinlar.push({
          type:'line',xref:'x',yref:'y',
          x0:x1,x1:x2,y0:y1,y1:y2,
          line:{color:'#22d3ee',width:1.5,dash:'dot'},
          _lbl:(yuzde>=0?'+':'')+yuzde.toFixed(2)+'%'+mumTxt,_tip:'cetvel'
        });
        cetvelIlkNokta=null;
      }
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
        text:KARAR_ETIKET[sin.karar],showarrow:true,arrowhead:2,arrowsize:1.2,
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
    } else if(r._tip==='cetvel'){
      anns.push({x:r.x1,y:r.y1,xref:'x',yref:'y',
        text:r._lbl,showarrow:false,yshift:14,
        font:{size:9,color:'#22d3ee'},
        bgcolor:'rgba(7,5,14,.75)',borderpad:2,bordercolor:'rgba(34,211,238,.3)',borderwidth:1});
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
  indBilgiGun();
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
    document.getElementById('track-record-alani').innerHTML='<div class="es">Henuz tamamlanan sinyal yok.</div>';
    const pk=document.getElementById('tr-portfoy-kart'); if(pk) pk.style.display='none';
    return;
  }
  const portfoySemboller=new Set(_lsPortfoy().map(x=>x.sembol));
  const enPortfoy=tr.son_sinyaller.filter(s=>portfoySemboller.has((s.sembol||'').replace('.IS','')));
  const genel=tr.son_sinyaller.filter(s=>!portfoySemboller.has((s.sembol||'').replace('.IS','')));
  const pk=document.getElementById('tr-portfoy-kart'), gb=document.getElementById('tr-genel-baslik');
  if(enPortfoy.length){
    pk.style.display='';
    document.getElementById('track-record-portfoy').innerHTML=_trTabloOlustur(enPortfoy);
    gb.textContent='Diğer Hisseler';
  } else {
    pk.style.display='none';
    gb.textContent='Sinyal Geçmişi';
  }
  document.getElementById('track-record-alani').innerHTML=_trTabloOlustur(enPortfoy.length?genel:tr.son_sinyaller);
}

function _trTabloOlustur(liste){
  if(!liste.length) return '<div class="es">Bu grupta sinyal yok.</div>';
  let h='<table class="t"><thead><tr><th>Tarih</th><th>Hisse</th><th>Gosterge</th><th>Giris</th><th>Ref. Direnc</th><th>Ref. Destek</th><th>Cikis</th><th>K/Z</th><th>Sonuc</th></tr></thead><tbody>';
  liste.forEach(s=>{
    const kzv=s.kar_zarar?parseFloat(s.kar_zarar):null;
    const sr=s.sonuc==='KAZANDI'?'gr':s.sonuc==='KAYBETTI'?'re':'ye';
    const kc=s.karar==='AL'?'al':s.karar==='SAT'?'sat':'bekle';
    const rc=s.karar==='AL'?'al-r':s.karar==='SAT'?'sat-r':'bk-r';
    const kzs=kzv!=null?'<span class="'+(kzv>=0?'gr':'re')+'">%'+(kzv>0?'+':'')+kzv.toFixed(1)+'</span>':'-';
    h+='<tr class="'+rc+'"><td style="font-size:10px;white-space:nowrap">'+(s.zaman||'-')+'</td>'+
      '<td style="font-weight:700">'+(s.sembol||'').replace('.IS','')+'</td>'+
      '<td><span class="pill '+kc+'">'+KARAR_ETIKET[s.karar]+'</span></td>'+
      '<td>'+(s.fiyat_giris?parseFloat(s.fiyat_giris).toFixed(2)+' TL':'-')+'</td>'+
      '<td class="gr">'+(s.hedef?parseFloat(s.hedef).toFixed(2)+' TL':'-')+'</td>'+
      '<td class="'+(s.karar==='SAT'?'gr':'re')+'">'+(s.stop?parseFloat(s.stop).toFixed(2)+' TL':'-')+'</td>'+
      '<td>'+(s.fiyat_cikis?parseFloat(s.fiyat_cikis).toFixed(2)+' TL':'-')+'</td>'+
      '<td>'+kzs+'</td>'+
      '<td class="'+sr+'" style="font-weight:600">'+(s.sonuc||'Bekliyor')+'</td></tr>';
  });
  return h+'</tbody></table>';
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
  fetchYetkili('/api/portfoy/'+aktifSid(),{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sembol:s,adet:a,maliyet:m})}).catch(()=>{});
}

function portfoySil(s){
  const p=_lsPortfoy().filter(x=>x.sembol!==s);
  _lsPortfoyKaydet(p);
  _portfoyRender(p);
  fetchYetkili('/api/portfoy/'+aktifSid()+'/'+s,{method:'DELETE'}).catch(()=>{});
}

function portfoyGun(){
  const local=_lsPortfoy();
  _portfoyRender(local);
  fetchYetkili('/api/portfoy/'+aktifSid())
    .then(r=>r.json())
    .then(srv=>{
      if(Array.isArray(srv)){_lsPortfoyKaydet(srv);_portfoyRender(srv);}
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
  fetchYetkili('/api/alarmlar/'+aktifSid(),{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sembol:s,yon:y,fiyat:f})})
    .then(r=>r.json())
    .then(res=>{if(res.id){alarmGun();}})
    .catch(()=>{});
}

function alarmSil(key){
  if(typeof key==='number'){
    fetchYetkili('/api/alarmlar/'+aktifSid()+'/'+key,{method:'DELETE'}).catch(()=>{});
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
    const sinS=sinyalVerisi.find(s=>s.sembol.replace('.IS','')===alarm.sembol);
    let hit=false;
    if(alarm.yon==='above') hit=g>=alarm.fiyat;
    else if(alarm.yon==='below') hit=g<=alarm.fiyat;
    else if(alarm.yon==='rsi_ob'&&sinS) hit=sinS.rsi>70;
    else if(alarm.yon==='rsi_os'&&sinS) hit=sinS.rsi<30;
    else if(alarm.yon==='sinyal_al'&&sinS) hit=sinS.karar==='AL';
    else if(alarm.yon==='sinyal_sat'&&sinS) hit=sinS.karar==='SAT';
    if(hit){
      a[i].tetiklendi=true; ch=true;
      if('Notification' in window&&Notification.permission==='granted'){
        const msg=alarm.yon==='rsi_ob'?alarm.sembol+' RSI asiri alim bolgesine girdi!':
          alarm.yon==='rsi_os'?alarm.sembol+' RSI asiri satim bolgesinde!':
          alarm.yon==='sinyal_al'?alarm.sembol+' göstergesi pozitife döndü (deneysel, yatırım tavsiyesi değildir).':
          alarm.yon==='sinyal_sat'?alarm.sembol+' göstergesi negatife döndü (deneysel, yatırım tavsiyesi değildir).':
          alarm.sembol+' -> '+g.toFixed(2)+' TL ('+(alarm.yon==='above'?'yukseldi':'dustu')+')';
        new Notification('LIDYA Alarm',{body:msg});
      }
    }
  });
  if(ch){_lsAlarmlarKaydet(a); _alarmRender(a); _alarm2Render(a);}
}

function alarmGun(){
  const local=_lsAlarmlar();
  _alarmRender(local);
  fetchYetkili('/api/alarmlar/'+aktifSid())
    .then(r=>r.json())
    .then(srv=>{
      if(Array.isArray(srv)){_lsAlarmlarKaydet(srv);_alarmRender(srv);}
    })
    .catch(()=>{});
}

// ── Ekonomik Takvim ──────────────────────────────────
let _takFiltre='hepsi', _takvimVeri=[];

function takvimYukle(){
  const el=document.getElementById('takvim-listesi');
  if(!el||_takvimVeri.length) return takvimRender();
  fetch('/api/takvim').then(r=>r.json()).then(d=>{_takvimVeri=d;takvimRender();})
    .catch(()=>{if(el) el.innerHTML='<div class="es">Veri alinamadi.</div>';});
}

function takFiltre(tur,btn){
  _takFiltre=tur;
  document.querySelectorAll('.tak-chip').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  takvimRender();
}

function takvimRender(){
  const el=document.getElementById('takvim-listesi');
  if(!el) return;
  const bugun=new Date(); bugun.setHours(0,0,0,0);
  const fil=_takFiltre==='hepsi'?_takvimVeri:_takvimVeri.filter(e=>e.kategori===_takFiltre);
  const gelecek=fil.filter(e=>new Date(e.tarih)>=bugun);
  const sonEl=document.getElementById('tak-sonraki');
  if(gelecek.length&&sonEl){
    const s=gelecek[0];
    const fark=Math.ceil((new Date(s.tarih)-bugun)/86400000);
    sonEl.textContent=fark===0?'Bugün: '+s.baslik:(fark===1?'Yarın':fark+' gün sonra')+': '+s.baslik;
  } else if(sonEl) sonEl.textContent='';
  if(!fil.length){el.innerHTML='<div class="es">Bu kategoride etkinlik yok.</div>';return;}
  const renkMap={tcmb:'#a78bfa',tuik:'#22c55e',fed:'#f59e0b',bilanco:'#06b6d4',diger:'#6b6488'};
  const etiMap={tcmb:'TCMB PPK',tuik:'TÜİK',fed:'FED FOMC',bilanco:'Bilanço',diger:'Diğer'};
  const gunAd=['Paz','Pzt','Sal','Çar','Per','Cum','Cmt'];
  const ayAd=['Oca','Şub','Mar','Nis','May','Haz','Tem','Ağu','Eyl','Eki','Kas','Ara'];
  let h='';
  fil.forEach(ev=>{
    const d=new Date(ev.tarih);
    const gecti=d<bugun;
    const bugunmu=d.getTime()===bugun.getTime();
    const fark=Math.ceil((d-bugun)/86400000);
    const r=renkMap[ev.kategori]||renkMap.diger;
    const et=etiMap[ev.kategori]||ev.kategori;
    const geriS=gecti?'Geçti':bugunmu?'BUGÜN!':fark===1?'Yarın':fark+' gün';
    const geriR=gecti?'var(--mu)':bugunmu?'var(--ac)':fark<=7?'var(--ye)':'var(--mu)';
    const onemDot=ev.onem==='yuksek'?'<span style="color:var(--re);font-size:9px;margin-left:4px">●●●</span>':
      ev.onem==='orta'?'<span style="color:var(--ye);font-size:9px;margin-left:4px">●●</span>':'';
    h+='<div class="tak-item'+(gecti?' tak-gecti':'')+(bugunmu?' tak-aktif':'')+'">' +
      '<div class="tak-tarih-blok">' +
      '<div class="tak-gun" style="color:'+(gecti?'var(--mu)':r)+'">'+d.getDate()+'</div>' +
      '<div class="tak-ay">'+gunAd[d.getDay()]+' '+ayAd[d.getMonth()]+'</div>' +
      '</div>' +
      '<div class="tak-kart-ic">' +
      '<div><span class="tak-badge" style="background:'+r+'20;color:'+r+';border:1px solid '+r+'33">'+et+'</span>'+onemDot+'</div>' +
      '<div class="tak-basl">'+ev.baslik+'</div>' +
      (ev.aciklama?'<div class="tak-alt">'+ev.aciklama+'</div>':'')+
      '<div class="tak-geri-sayim" style="color:'+geriR+'">'+geriS+'</div>' +
      '</div></div>';
  });
  el.innerHTML=h;
}

// ── Alarm Merkezi ─────────────────────────────────────
const ALARM_TUR_LBL={above:'Fiyat Üstü ↑',below:'Fiyat Altı ↓',rsi_ob:'RSI Aşırı Alım',rsi_os:'RSI Aşırı Satım',sinyal_al:'Gösterge Pozitife Döndü',sinyal_sat:'Gösterge Negatife Döndü'};

function alarmTurGun(){
  const tur=document.getElementById('al2-tur'); if(!tur) return;
  const fp=document.getElementById('al2-fiyat'); if(!fp) return;
  fp.style.display=(tur.value==='above'||tur.value==='below')?'':'none';
}

function alarm2Ekle(){
  const s=(document.getElementById('al2-sembol').value||'').toUpperCase().trim();
  const tur=document.getElementById('al2-tur').value;
  const fp=document.getElementById('al2-fiyat');
  const f=parseFloat(fp.value);
  if(!s){alert('Sembol zorunlu.');return;}
  if((tur==='above'||tur==='below')&&!f){alert('Fiyat zorunlu.');return;}
  document.getElementById('al2-sembol').value=''; fp.value='';
  const fiyatVal=(tur==='above'||tur==='below')?f:null;
  const a=_lsAlarmlar();
  a.push({sembol:s,yon:tur,fiyat:fiyatVal,tetiklendi:false,id:Date.now()});
  _lsAlarmlarKaydet(a); _alarm2Render(a);
  fetchYetkili('/api/alarmlar/'+aktifSid(),{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sembol:s,yon:tur,fiyat:fiyatVal})})
    .then(r=>r.json()).then(res=>{if(res.id) alarm2Gun();}).catch(()=>{});
}

function alarm2Sil(key){
  if(typeof key==='number'){
    fetchYetkili('/api/alarmlar/'+aktifSid()+'/'+key,{method:'DELETE'}).catch(()=>{});
    const a=_lsAlarmlar().filter(x=>x.id!==key);
    _lsAlarmlarKaydet(a); _alarm2Render(a);
  } else {
    const a=_lsAlarmlar();
    const idx=a.findIndex(x=>JSON.stringify(x)===key);
    if(idx>=0) a.splice(idx,1);
    _lsAlarmlarKaydet(a); _alarm2Render(a);
  }
  alarmGun();
}

function alarm2Gun(){
  const local=_lsAlarmlar(); _alarm2Render(local);
  fetchYetkili('/api/alarmlar/'+aktifSid()).then(r=>r.json()).then(srv=>{
    if(Array.isArray(srv)){_lsAlarmlarKaydet(srv);_alarm2Render(srv);}
  }).catch(()=>{});
}

function _alarm2Render(a){
  const el=document.getElementById('alarm2-listesi'); if(!el) return;
  const aktif=a.filter(x=>!x.tetiklendi);
  const tetik=a.filter(x=>x.tetiklendi);
  const tk=document.getElementById('tetik-kart'), tl=document.getElementById('tetik-listesi');
  if(tk&&tl){
    if(tetik.length){
      tk.style.display='';
      let th='';
      tetik.forEach(x=>{
        const g=fiyatlar[x.sembol];
        let detay='';
        if(x.fiyat) detay='Hedef: '+parseFloat(x.fiyat).toFixed(2)+' TL'+(g?' · Şu an: '+g.toFixed(2)+' TL':'');
        th+='<div class="notif-item">'+
          '<div class="notif-dot" style="background:var(--gr)"></div>'+
          '<div><div style="font-size:12px;font-weight:600">'+x.sembol+' — '+(ALARM_TUR_LBL[x.yon]||x.yon)+'</div>'+
          (detay?'<div style="font-size:10px;color:var(--mu)">'+detay+'</div>':'')+
          '</div></div>';
      });
      tl.innerHTML=th;
    } else tk.style.display='none';
  }
  if(!aktif.length){el.innerHTML='<div class="es">Aktif alarm yok.</div>';return;}
  let h='<table class="t"><thead><tr><th>Hisse</th><th>Tür</th><th>Koşul</th><th>Güncel</th><th>Durum</th><th></th></tr></thead><tbody>';
  aktif.forEach(x=>{
    const g=fiyatlar[x.sembol];
    const delKey=x.id!==undefined?x.id:JSON.stringify(x);
    let kosul='-';
    if(x.yon==='above'||x.yon==='below') kosul=x.fiyat?parseFloat(x.fiyat).toFixed(2)+' TL':'-';
    else if(x.yon==='rsi_ob') kosul='RSI > 70';
    else if(x.yon==='rsi_os') kosul='RSI < 30';
    else if(x.yon==='sinyal_al') kosul='Gösterge = Pozitif';
    else if(x.yon==='sinyal_sat') kosul='Gösterge = Negatif';
    h+='<tr><td style="font-weight:700">'+x.sembol+'</td>'+
      '<td style="font-size:10px;white-space:nowrap">'+(ALARM_TUR_LBL[x.yon]||x.yon)+'</td>'+
      '<td>'+kosul+'</td>'+
      '<td>'+(g!==undefined?g.toFixed(2)+' TL':'-')+'</td>'+
      '<td><span class="alb">Bekliyor</span></td>'+
      '<td><button class="bd2" onclick="alarm2Sil('+JSON.stringify(delKey)+')">Sil</button></td></tr>';
  });
  el.innerHTML=h+'</tbody></table>';
}

function bildirimIzniAl(){
  if(!('Notification' in window)) return;
  Notification.requestPermission().then(()=>bildirimDurumGun());
}

function bildirimDurumGun(){
  const el=document.getElementById('bildirim-durum');
  const btn=document.getElementById('bildirim-btn');
  if(!('Notification' in window)){
    if(el) el.textContent='Tarayıcınız bildirimleri desteklemiyor.'; return;
  }
  const p=Notification.permission;
  if(el){
    el.textContent=p==='granted'?'✓ Bildirimler aktif':
      p==='denied'?'✗ Bildirimler engellendi — tarayıcı ayarlarından açın':
      'Bildirimler için izin verilmedi.';
    el.style.color=p==='granted'?'var(--gr)':p==='denied'?'var(--re)':'var(--mu)';
  }
  if(btn) btn.style.display=p==='default'?'':'none';
}

// ── İndikatör Rehberi ────────────────────────────────
const IND_KARTLAR=[
  {id:'ik-rsi',kat:'Momentum',kr:'#6366f1',isim:'RSI (14)',
   aciklama:'Göreceli Güç Endeksi. Alım ve satım baskısının gücünü ölçer.',
   hint:'&gt; 70 → Aşırı Alım (dikkat) · &lt; 30 → Aşırı Satım (fırsat) · 50 = nötr'},
  {id:'ik-ma',kat:'Trend',kr:'#f59e0b',isim:'MA20 / MA50',
   aciklama:'Kısa/orta vadeli hareketli ortalamalar. Çapraz noktalar trend değişimini işaret eder.',
   hint:'MA20 &gt; MA50 → Golden Cross (boğa) · MA20 &lt; MA50 → Death Cross (ayı)'},
  {id:'ik-ma200',kat:'Uzun Vade',kr:'#ec4899',isim:'MA200',
   aciklama:'200 günlük ortalama. Kurumsal yatırımcıların takip ettiği uzun vadeli trend göstergesi.',
   hint:'Fiyat MA200 üstünde → uzun vadeli boğa · Altında → ayı piyasası'},
  {id:'ik-bb',kat:'Volatilite',kr:'#8b5cf6',isim:'Bollinger Bantları',
   aciklama:'Fiyat volatilitesini ölçer. Bantların daralması büyük bir hareket öncesi gözlenir.',
   hint:'Bantlar daralınca kırılım bekle · Fiyat %80+ bölgesi → aşırı gerilmiş'},
  {id:'ik-macd',kat:'Momentum',kr:'#6366f1',isim:'MACD Histogramı',
   aciklama:'Trend momentum ve yön değişimlerini ölçer. Histogramın sıfır geçişi kritik sinyaldir.',
   hint:'Histogram artıyor → momentum güçleniyor · Sıfır geçişi = yön değişimi'},
  {id:'ik-stoch',kat:'Momentum',kr:'#06b6d4',isim:'Stochastic %K',
   aciklama:'Kısa vadeli momentum. Fiyatın 14 günlük aralıktaki konumunu 0-100 arasında gösterir.',
   hint:'&gt; 80 → Aşırı Alım · &lt; 20 → Aşırı Satım · %K/%D kesişimi sinyal verir'},
  {id:'ik-vol',kat:'Hacim',kr:'#22c55e',isim:'Hacim Analizi',
   aciklama:'Güncel hacimin 20 günlük ortalamasına oranı. Fiyat hareketinin güvenilirliğini teyit eder.',
   hint:'Yüksek hacimli hareket daha güvenilir · Düşük hacimde yalancı kırılım riski'},
  {id:'ik-atr',kat:'Volatilite',kr:'#f59e0b',isim:'ATR (Volatilite)',
   aciklama:'Ortalama Gerçek Aralık. Günlük fiyat dalgalanmasını ölçer; referans direnç/destek seviyesi hesabında kullanılır.',
   hint:'Referans direnç: ATR × 1.5 üstü · Referans destek: ATR × 2.5 altı · Yüksek ATR = oynak piyasa (bilgi amaçlıdır, emir değildir)'},
  {id:'ik-sr',kat:'Teknik Seviye',kr:'#a78bfa',isim:'Destek / Direnç',
   aciklama:'Pivot noktaları ile hesaplanan, fiyatın geçmişte tepki verdiği seviyeler.',
   hint:'Fiyat geçmişte bu seviyelerde yön değiştirmiş · tekrar edeceği garanti değildir'},
];

function indKartlarOlustur(){
  const grid=document.getElementById('ind-kart-grid');if(!grid) return;
  let h='';
  IND_KARTLAR.forEach(k=>{
    h+='<div class="ind-kart" id="'+k.id+'">'+
      '<span class="ind-k-kat" style="background:'+k.kr+'20;color:'+k.kr+';border:1px solid '+k.kr+'33">'+k.kat+'</span>'+
      '<div class="ind-k-isim">'+k.isim+'</div>'+
      '<div class="ind-k-deger" id="'+k.id+'-v" style="color:var(--mu)">—</div>'+
      '<div><span class="ind-k-durum" id="'+k.id+'-d" style="background:rgba(107,100,136,.12);color:var(--mu)">Veri bekleniyor</span></div>'+
      '<div class="ind-k-acik">'+k.aciklama+'</div>'+
      '<div class="ind-k-hint">'+k.hint+'</div>'+
      '</div>';
  });
  grid.innerHTML=h;
}

function _ik(id,val,dur,valRenk,durBg,durRenk){
  const v=document.getElementById(id+'-v'),d=document.getElementById(id+'-d');
  if(v){v.textContent=val;v.style.color=valRenk||'var(--tx)';}
  if(d){d.textContent=dur;d.style.background=durBg;d.style.color=durRenk;}
}

function indBilgiGun(){
  const grid=document.getElementById('ind-kart-grid');
  if(grid&&!grid.children.length) indKartlarOlustur();
  const el=document.getElementById('hisse-sec');if(!el) return;
  const hisse=el.value, veri=grafikVerisi[hisse];
  if(!veri||!veri.tarihler) return;

  const n=Math.min(period,veri.tarihler.length);
  const sl=arr=>(arr||[]).slice(-n);
  const last=arr=>{if(!arr) return null;const f=arr.filter(v=>v!=null);return f.length?f[f.length-1]:null;};

  const cl=sl(veri.close),hi=sl(veri.high),lo=sl(veri.low),tar=sl(veri.tarihler);
  const fiyat=last(cl),ma20v=last(sl(veri.ma20)),ma50v=last(sl(veri.ma50)),ma200v=last(sl(veri.ma200));
  const rsi=last(sl(veri.rsi)),bbUv=last(sl(veri.bb_ust)),bbLv=last(sl(veri.bb_alt));
  const macdHv=last(sl(veri.macd_h)),stKv=last(sl(veri.stoch_k)),volv=last(sl(veri.volume));

  // RSI
  if(rsi!=null){
    const s=rsi>70?['Aşırı Alım','rgba(239,68,68,.14)','#ef4444']:
             rsi<30?['Aşırı Satım','rgba(34,197,94,.14)','#22c55e']:
             rsi>55?['Güçlü Bölge','rgba(34,197,94,.1)','#4ade80']:
             rsi<45?['Zayıf Bölge','rgba(239,68,68,.08)','#f87171']:
             ['Nötr Bölge','rgba(245,158,11,.1)','#f59e0b'];
    _ik('ik-rsi',rsi.toFixed(1),s[0],rsi>70?'#ef4444':rsi<30?'#22c55e':'var(--tx)',s[1],s[2]);
  }

  // MA20/MA50
  if(fiyat&&ma20v&&ma50v){
    const pct=(fiyat-ma20v)/ma20v*100;
    const bull=ma20v>ma50v;
    const spread=Math.abs(ma20v-ma50v)/ma50v*100;
    const dur=bull?(spread>1.5?'Golden Cross ↑':'MA20 > MA50'):(spread>1.5?'Death Cross ↓':'MA20 < MA50');
    _ik('ik-ma',(pct>=0?'+':'')+pct.toFixed(1)+'%  (MA20\\'ye gore)',dur,'var(--tx)',
        bull?'rgba(34,197,94,.12)':'rgba(239,68,68,.12)',bull?'#22c55e':'#ef4444');
  }

  // MA200
  if(fiyat&&ma200v){
    const pct=(fiyat-ma200v)/ma200v*100;
    const ust=fiyat>ma200v;
    _ik('ik-ma200',(pct>=0?'+':'')+pct.toFixed(1)+'%  (MA200\\'e gore)',
        ust?'Fiyat MA200 Üstünde':'Fiyat MA200 Altında','var(--tx)',
        ust?'rgba(34,197,94,.12)':'rgba(239,68,68,.12)',ust?'#22c55e':'#ef4444');
  }

  // Bollinger
  if(fiyat&&bbUv&&bbLv&&bbUv>bbLv){
    const k=(fiyat-bbLv)/(bbUv-bbLv)*100;
    const gen=(bbUv-bbLv)/fiyat*100;
    const s=k>80?['Üst Banda Yakın','rgba(239,68,68,.12)','#ef4444']:
             k<20?['Alt Banda Yakın','rgba(34,197,94,.12)','#22c55e']:
             ['Orta Bölge','rgba(245,158,11,.1)','#f59e0b'];
    _ik('ik-bb',k.toFixed(0)+'%  (bant genişliği: '+gen.toFixed(1)+'%)',s[0],'var(--tx)',s[1],s[2]);
  }

  // MACD Histogram
  if(macdHv!=null){
    const pos=macdHv>=0;
    _ik('ik-macd',(pos?'+':'')+macdHv.toFixed(4),
        pos?'Pozitif Momentum':'Negatif Momentum',pos?'#22c55e':'#ef4444',
        pos?'rgba(34,197,94,.12)':'rgba(239,68,68,.12)',pos?'#22c55e':'#ef4444');
  }

  // Stochastic
  if(stKv!=null){
    const s=stKv>80?['Aşırı Alım','rgba(239,68,68,.12)','#ef4444']:
             stKv<20?['Aşırı Satım','rgba(34,197,94,.12)','#22c55e']:
             ['Normal Bölge','rgba(245,158,11,.1)','#f59e0b'];
    _ik('ik-stoch',stKv.toFixed(1),s[0],stKv>80?'#ef4444':stKv<20?'#22c55e':'var(--tx)',s[1],s[2]);
  }

  // Hacim
  const volArr=sl(veri.volume).filter(v=>v!=null);
  if(volv!=null&&volArr.length>5){
    const vSlice=volArr.slice(-20);
    const vMA=vSlice.reduce((a,b)=>a+b,0)/vSlice.length;
    const oran=volv/vMA;
    const s=oran>1.5?['Yüksek Hacim ↑','rgba(34,197,94,.12)','#22c55e']:
             oran<0.7?['Düşük Hacim ↓','rgba(239,68,68,.08)','#f87171']:
             ['Normal Hacim','rgba(245,158,11,.1)','#f59e0b'];
    _ik('ik-vol',oran.toFixed(2)+'×  ort. ('+Math.round(volv/1000)+'K lot)',s[0],'var(--tx)',s[1],s[2]);
  }

  // ATR (14-bar approximate)
  if(hi.length>15&&fiyat){
    let s=0,c=0;
    for(let i=Math.max(1,hi.length-14);i<hi.length;i++){
      if(hi[i]!=null&&lo[i]!=null&&cl[i-1]!=null){
        s+=Math.max(hi[i]-lo[i],Math.abs(hi[i]-cl[i-1]),Math.abs(lo[i]-cl[i-1])); c++;
      }
    }
    if(c>0){
      const atr=s/c, pct=atr/fiyat*100;
      const st=pct>4?['Yüksek Volatilite','rgba(239,68,68,.12)','#ef4444']:
               pct>2?['Orta Volatilite','rgba(245,158,11,.1)','#f59e0b']:
               ['Düşük Volatilite','rgba(34,197,94,.12)','#22c55e'];
      _ik('ik-atr',atr.toFixed(2)+' TL  ('+pct.toFixed(1)+'%)',st[0],'var(--tx)',st[1],st[2]);
    }
  }

  // S/R — en yakın destek/direnç
  if(fiyat&&hi.length>0){
    const pivs=_pivotHesapla(hi,lo,tar);
    const dir=pivs.filter(p=>p.t==='R').sort((a,b)=>a.f-b.f)[0];
    const des=pivs.filter(p=>p.t==='S').sort((a,b)=>b.f-a.f)[0];
    if(dir||des){
      const dStr=dir?'D '+dir.f.toFixed(2)+' TL (+'+((dir.f-fiyat)/fiyat*100).toFixed(1)+'%)':'';
      const sStr=des?'S '+des.f.toFixed(2)+' TL ('+((des.f-fiyat)/fiyat*100).toFixed(1)+'%)':'';
      const val=[dStr,sStr].filter(Boolean).join('   ');
      const dMes=dir?(dir.f-fiyat)/fiyat*100:99, sMes=des?(fiyat-des.f)/fiyat*100:99;
      const close=dMes<sMes;
      _ik('ik-sr',val,close?'Dirençe Yakın':'Desteğe Yakın','var(--tx)',
          close?'rgba(239,68,68,.1)':'rgba(34,197,94,.1)',close?'#ef4444':'#22c55e');
    }
  }
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
      '<span class="pill '+kc+'">'+KARAR_ETIKET[s.karar]+'</span></div>';
  });
  el.innerHTML=h;
}

let _modalSembol=null;

function sirketAc(sembol){
  _modalSembol=sembol;
  const modal=document.getElementById('sirket-modal');
  modal.style.display='flex';
  document.getElementById('modal-icerik').innerHTML='<div class="es">Yukleniyor...</div>';
  fetch('/api/hisse/'+sembol).then(r=>r.json()).then(d=>{
    const s=d.sinyal,f=d.finans||{};
    const kc=s?(s.karar==='AL'?'al':s.karar==='SAT'?'sat':'bekle'):'bekle';
    const fmt=(v,dec,suf)=>v==null?'-':(Number(v).toLocaleString('tr-TR',{maximumFractionDigits:dec||0}))+(suf||'');
    const fmtP=v=>v==null?'-':v>=1e12?(v/1e12).toFixed(1)+' T TL':v>=1e9?(v/1e9).toFixed(1)+' Mr TL':v>=1e6?(v/1e6).toFixed(0)+' Mn TL':fmt(v,0,' TL');
    const metriks=[
      ['Piyasa Degeri',fmtP(f.piyasaDegeri)],['F/K',fmt(f.fk,1)],['PD/DD',fmt(f.pd_dd,2)],
      ['52H Yuksek',fmt(f.yuksek52,2,' TL')],['52H Dusuk',fmt(f.dusuk52,2,' TL')],
      ['Temettu',f.temettu!=null?'%'+Number(f.temettu*100).toFixed(1):'-'],
      ['Beta',fmt(f.beta,2)],['Calisan',f.calisanSayisi?Number(f.calisanSayisi).toLocaleString('tr-TR'):'-'],
    ];
    let h='<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">'+
      '<div style="display:flex;gap:12px;align-items:center">'+
      '<div style="width:50px;height:50px;border-radius:12px;background:rgba(167,139,250,.12);border:1px solid rgba(167,139,250,.3);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:15px;color:var(--ac)">'+sembol.substring(0,2)+'</div>'+
      '<div><div style="font-size:17px;font-weight:700;color:var(--tx)">'+d.ad+'</div>'+
      '<div style="font-size:11px;color:var(--mu);margin-top:2px">'+sembol+' — '+d.sektor+'</div></div></div>'+
      '<div style="display:flex;gap:8px;align-items:center">'+
      (s?'<span class="pill '+kc+'">'+KARAR_ETIKET[s.karar]+'</span>':'')+
      '<button onclick="sirketKapat()" style="background:var(--sf);border:1px solid var(--bd);color:var(--mu);width:30px;height:30px;border-radius:6px;cursor:pointer;font-size:16px;line-height:1">x</button>'+
      '</div></div>';
    h+='<div class="modal-sekmeler">'+
      '<button class="modal-sekme active" data-tab="ozet" onclick="modalSekme(this.dataset.tab,this)">Ozet</button>'+
      '<button class="modal-sekme" data-tab="finansallar" onclick="modalSekme(this.dataset.tab,this)">Finansallar</button>'+
      '</div>';
    let ozet='';
    if(d.aciklama) ozet+='<div style="font-size:12px;color:var(--mu);line-height:1.65;padding:12px;background:var(--sf);border-radius:8px;margin-bottom:12px">'+d.aciklama+'</div>';
    ozet+='<div class="met-grid">';
    metriks.forEach(function(m){ozet+='<div class="met"><div class="met-l">'+m[0]+'</div><div class="met-v">'+m[1]+'</div></div>';});
    ozet+='</div>';
    if(s){
      const dr=s.degisim>=0?'gr':'re';
      ozet+='<div style="background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:11px;display:flex;gap:16px;flex-wrap:wrap;font-size:12px;margin-top:10px">'+
        '<span>Fiyat: <strong>'+Number(s.fiyat).toFixed(2)+' TL</strong></span>'+
        '<span class="'+dr+'">'+( s.degisim>=0?'+':'')+Number(s.degisim).toFixed(2)+'%</span>'+
        '<span>RSI: <strong>'+Number(s.rsi).toFixed(1)+'</strong></span>'+
        '<span class="gr">Ref. Direnç: <strong>'+Number(s.hedef).toFixed(2)+' TL</strong></span>'+
        '<span class="re">Ref. Destek: <strong>'+Number(s.stop).toFixed(2)+' TL</strong></span>'+
        '<span>Guven: <strong>%'+Number(s.guven*100).toFixed(0)+'</strong></span></div>';
    }
    h+='<div id="modal-ozet-k">'+ozet+'</div>';
    h+='<div id="modal-fin-k" style="display:none"><div class="es">Yukleniyor...</div></div>';
    document.getElementById('modal-icerik').innerHTML=h;
  }).catch(function(){document.getElementById('modal-icerik').innerHTML='<div class="es">Veri alinamadi.</div>';});
}

function modalSekme(id,btn){
  document.getElementById('modal-ozet-k').style.display=id==='ozet'?'':'none';
  document.getElementById('modal-fin-k').style.display=id==='finansallar'?'':'none';
  document.querySelectorAll('.modal-sekme').forEach(function(b){b.classList.remove('active');});
  if(btn) btn.classList.add('active');
  if(id==='finansallar') finansallarYukle(_modalSembol);
}

function finansallarYukle(sembol){
  const el=document.getElementById('modal-fin-k');
  if(!el||el.dataset.yuklendi===sembol) return;
  el.innerHTML='<div class="es">Finansal veriler yukleniyor...</div>';
  fetch('/api/finans/'+sembol).then(r=>r.json()).then(function(d){
    if(d.error){el.innerHTML='<div class="es">Veri alinamadi.</div>';return;}
    const c=d.ceyrekler||[], c0=c[0]||'-', c1=c[1]||'-';
    const fmtN=v=>v==null?'-':Number(v).toLocaleString('tr-TR');
    function pctStr(arr){
      const v0=arr[0],v1=arr[1];
      if(v0==null||v1==null||v1===0) return '-';
      const r=((v0-v1)/Math.abs(v1)*100);
      return (r>0?'+':'')+r.toFixed(0)+'%';
    }
    function buildTablo(baslik,satirlar){
      let t='<table class="fin-tablo"><thead><tr>'+
        '<th style="text-align:left">'+baslik+' <span style="font-weight:400">(Bin TRY)</span></th>'+
        '<th>'+c0+'</th><th style="color:var(--mu)">'+c1+'</th><th>%</th>'+
        '</tr></thead><tbody>';
      satirlar.forEach(function(sat){
        const p=pctStr(sat.deger);
        const pc=p.startsWith('+')?'style="color:var(--gr)"':p.startsWith('-')?'style="color:var(--re)"':'';
        t+='<tr><td>'+sat.etiket+'</td>'+
          '<td>'+fmtN(sat.deger[0])+'</td>'+
          '<td style="color:var(--mu)">'+fmtN(sat.deger[1])+'</td>'+
          '<td '+pc+'>'+p+'</td></tr>';
      });
      t+='</tbody></table>';
      return t;
    }
    const o=d.oranlar||{};
    const fmtO=v=>v==null?'—':v;
    let html='<div class="oran-serit">'+
      '<div class="oran-item"><div class="oran-l">F/K</div><div class="oran-v">'+(o.fk?Number(o.fk).toFixed(1):'—')+'</div></div>'+
      '<div class="oran-item"><div class="oran-l">PD/DD</div><div class="oran-v">'+(o.pd_dd?Number(o.pd_dd).toFixed(2):'—')+'</div></div>'+
      '<div class="oran-item"><div class="oran-l">ROE</div><div class="oran-v">'+(o.roe!=null?'%'+o.roe:'—')+'</div></div>'+
      '<div class="oran-item"><div class="oran-l">ROA</div><div class="oran-v">'+(o.roa!=null?'%'+o.roa:'—')+'</div></div>'+
      '<div class="oran-item"><div class="oran-l">Temettu</div><div class="oran-v">'+(o.temettu?'%'+(o.temettu*100).toFixed(1):'—')+'</div></div>'+
      '<div class="oran-item"><div class="oran-l">Beta</div><div class="oran-v">'+(o.beta?Number(o.beta).toFixed(2):'—')+'</div></div>'+
      '</div>';
    html+='<div class="fin-2kol">'+
      '<div>'+buildTablo('Ozet Gelir Tablosu',d.gelir)+'</div>'+
      '<div>'+buildTablo('Ozet Bilanco',d.bilanco)+'</div>'+
      '</div>';
    html+='<div class="fin-grafik-grid">'+
      '<div><div class="fin-grafik-baslik">'+d.grafik.net_kar.etiket+'</div><div id="fg-nk" style="height:130px"></div></div>'+
      '<div><div class="fin-grafik-baslik">'+d.grafik.ana_gelir.etiket+'</div><div id="fg-ag" style="height:130px"></div></div>'+
      '<div><div class="fin-grafik-baslik">'+d.grafik.ozkaynak.etiket+'</div><div id="fg-ok" style="height:130px"></div></div>'+
      '</div>';
    el.innerHTML=html;
    el.dataset.yuklendi=sembol;
    const BL={paper_bgcolor:'#07050e',plot_bgcolor:'#07050e',font:{color:'#5e5a7a',size:9},
      margin:{t:4,r:4,b:28,l:52},showlegend:false};
    function barCiz(divId,veriler,etiketler){
      const n=Math.min(veriler.length,etiketler.length);
      const x=[],y=[];
      for(let i=n-1;i>=0;i--){x.push(etiketler[i]||'');y.push(veriler[i]||0);}
      Plotly.newPlot(divId,[{type:'bar',x:x,y:y,
        marker:{color:y.map(v=>v>=0?'rgba(99,102,241,.75)':'rgba(239,68,68,.75)')}}],
        {...BL,xaxis:{gridcolor:'#1c1830',tickfont:{size:8}},yaxis:{gridcolor:'#1c1830',tickformat:'.2s'}},
        {responsive:true,displayModeBar:false});
    }
    barCiz('fg-nk',d.grafik.net_kar.deger,c);
    barCiz('fg-ag',d.grafik.ana_gelir.deger,c);
    barCiz('fg-ok',d.grafik.ozkaynak.deger,c);
  }).catch(function(){el.innerHTML='<div class="es">Veri alinamadi.</div>';});
}

function sirketKapat(){document.getElementById('sirket-modal').style.display='none';}
document.addEventListener('keydown',function(e){if(e.key==='Escape')sirketKapat();});

// ── Borsa tatil takvimi ───────────────────────────────
const BORSA_TAM_KAPALI=new Set([
  '2025-03-30','2025-03-31','2025-04-01',
  '2025-06-06','2025-06-07','2025-06-08','2025-06-09',
  '2026-03-19','2026-03-20','2026-03-21',
  '2026-05-26','2026-05-27','2026-05-28','2026-05-29',
]);
const BORSA_ARIFE=new Set([
  '2025-03-29','2025-06-05','2026-03-18','2026-05-25',
]);

function borsaAcikMi(tr){
  const gun=tr.getDay();
  if(gun===0||gun===6) return false;
  const ay=tr.getMonth()+1, gun2=tr.getDate(), yil=tr.getFullYear();
  const dak=tr.getHours()*60+tr.getMinutes();
  const pad=n=>String(n).padStart(2,'0');
  const tarihStr=yil+'-'+pad(ay)+'-'+pad(gun2);
  // Sabit ulusal tatiller
  const sabit=[[1,1],[4,23],[5,1],[5,19],[7,15],[8,30],[10,29]];
  if(sabit.some(([m,d])=>m===ay&&d===gun2)) return false;
  // 28 Ekim arife — 12:30'dan sonra kapalı
  if(ay===10&&gun2===28&&dak>=750) return false;
  // Tam kapalı dini bayramlar
  if(BORSA_TAM_KAPALI.has(tarihStr)) return false;
  // Arife günleri — 12:30'dan sonra kapalı
  if(BORSA_ARIFE.has(tarihStr)&&dak>=750) return false;
  // Normal seans 09:40 – 18:00
  return dak>=580&&dak<1080;
}

function saatGuncelle(){
  try{
    const tr=new Date(new Date().toLocaleString('en-US',{timeZone:'Europe/Istanbul'}));
    const p=n=>n.toString().padStart(2,'0');
    const saat=p(tr.getHours())+':'+p(tr.getMinutes())+':'+p(tr.getSeconds());
    const sg=document.getElementById('son-guncelleme');
    if(sg) sg.textContent=saat;
    const acik=borsaAcikMi(tr);
    const d=document.getElementById('borsa-durum');
    if(d&&!d.textContent.includes('YUKLEN')){
      const yeni=acik?'BORSA ACIK':'BORSA KAPALI';
      if(d.textContent!==yeni){d.textContent=yeni;d.className='badge'+(acik?'':' kapali');}
    }
  }catch(e){}
}
saatGuncelle();
setInterval(saatGuncelle,1000);

hesapUIGuncelle(); alarmGun(); alarm2Gun(); portfoyGun(); bildirimDurumGun(); alarmTurGun(); veriCek(); setInterval(veriCek,10000);
</script>
</body>
</html>'''

# ── BACKEND ────────────────────────────────────────────

def borsa_acik_mi():
    from datetime import date, time as dtime
    now   = datetime.now()
    bugun = now.date()

    if now.weekday() >= 5:          # Cumartesi / Pazar
        return False

    # ── Sabit ulusal tatiller (ay, gün) ────────────────
    SABIT = {(1,1),(4,23),(5,1),(5,19),(7,15),(8,30),(10,29)}
    if (bugun.month, bugun.day) in SABIT:
        return False

    # ── 28 Ekim arife — öğleden sonra 12:30'dan itibaren kapalı ──
    if bugun.month == 10 and bugun.day == 28 and now.time() >= dtime(12, 30):
        return False

    # ── Tam kapalı dini bayram günleri ─────────────────
    TAM_KAPALI = {
        # Ramazan Bayramı 2025 (30 Mar – 1 Nis)
        date(2025,3,30), date(2025,3,31), date(2025,4,1),
        # Kurban Bayramı 2025 (6–9 Haz)
        date(2025,6,6), date(2025,6,7), date(2025,6,8), date(2025,6,9),
        # Ramazan Bayramı 2026 (19–21 Mar)
        date(2026,3,19), date(2026,3,20), date(2026,3,21),
        # Kurban Bayramı 2026 (26–29 May)
        date(2026,5,26), date(2026,5,27), date(2026,5,28), date(2026,5,29),
    }
    if bugun in TAM_KAPALI:
        return False

    # ── Arife günleri — öğleden sonra 12:30'dan itibaren kapalı ──
    ARIFE = {
        date(2025,3,29),   # Ramazan arife 2025
        date(2025,6,5),    # Kurban arife 2025
        date(2026,3,18),   # Ramazan arife 2026
        date(2026,5,25),   # Kurban arife 2026
    }
    if bugun in ARIFE and now.time() >= dtime(12, 30):
        return False

    # ── Normal seans: 09:40 – 18:00 ────────────────────
    return dtime(9, 40) <= now.time() <= dtime(18, 0)

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

def _feed_entry_zamani(entry):
    if getattr(entry, 'published_parsed', None):
        import calendar
        try:
            return datetime.fromtimestamp(calendar.timegm(entry.published_parsed))
        except Exception:
            pass
    return datetime.now()

def hisse_ozel_haberler():
    if not _HABER_ANALIZI_OK:
        return []
    sonuc = []
    for sembol in HISSELER:
        sembol_adi = sembol.replace('.IS', '')
        try:
            for h in _google_news_rss(f"{sembol_adi} hisse borsa", 'tr'):
                if not h.get('baslik'):
                    continue
                sonuc.append({
                    'baslik': h['baslik'][:90],
                    'kaynak': f"{h['kaynak']} — {sembol_adi}",
                    'link'  : f"https://news.google.com/search?q={sembol_adi}%20hisse%20borsa&hl=tr",
                    'zaman' : h.get('zaman') or datetime.now(),
                })
        except Exception:
            pass
        try:
            for h in _yahoo_news_cek(sembol):
                if not h.get('baslik'):
                    continue
                sonuc.append({
                    'baslik': h['baslik'][:90],
                    'kaynak': f"{h['kaynak']} — {sembol_adi}",
                    'link'  : f"https://finance.yahoo.com/quote/{sembol}/news",
                    'zaman' : h.get('zaman') or datetime.now(),
                })
        except Exception:
            pass
    return sonuc

def haber_cek():
    if _feedparser is None:
        return []
    kaynaklar = [
        ("https://www.bloomberght.com/rss",                          "Bloomberg HT"),
        ("https://www.haberturk.com/rss/ekonomi.xml",                "Haberturk"),
        ("https://www.sabah.com.tr/rss/ekonomi.xml",                 "Sabah Ekonomi"),
        ("https://tr.investing.com/rss/news.rss",                    "Investing.com TR"),
        ("https://tr.investing.com/rss/stock_stock_picks.rss",       "Investing.com Hisse"),
        ("https://www.ntv.com.tr/ekonomi.rss",                       "NTV Ekonomi"),
        ("https://www.milliyet.com.tr/rss/rssNew/ekonomiRss.xml",    "Milliyet Ekonomi"),
    ]
    gruplar = []
    for url, kaynak in kaynaklar:
        try:
            feed = _feedparser.parse(url)
            for entry in (feed.entries or [])[:4]:
                baslik = (entry.get('title') or '').strip()
                if baslik:
                    gruplar.append({
                        'baslik': baslik[:90],
                        'kaynak': kaynak,
                        'link'  : entry.get('link', '#'),
                        'zaman' : _feed_entry_zamani(entry),
                    })
        except:
            pass

    gruplar += hisse_ozel_haberler()

    simdi = datetime.now()
    gruplar = [g for g in gruplar if (simdi - g['zaman']) <= timedelta(hours=48)]
    gruplar.sort(key=lambda g: g['zaman'], reverse=True)
    gruplar = gruplar[:60]
    for g in gruplar:
        g.pop('zaman', None)
    return gruplar

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

EKONOMIK_TAKVIM = [
    # TCMB PPK Faiz Kararları (TCMB'nin resmi 2026 takvimi)
    {'tarih':'2026-01-22','baslik':'TCMB PPK Faiz Kararı','aciklama':'Para Politikası Kurulu toplantısı ve faiz kararı açıklaması','kategori':'tcmb','onem':'yuksek'},
    {'tarih':'2026-03-12','baslik':'TCMB PPK Faiz Kararı','aciklama':'Para Politikası Kurulu toplantısı','kategori':'tcmb','onem':'yuksek'},
    {'tarih':'2026-04-22','baslik':'TCMB PPK Faiz Kararı','aciklama':'Para Politikası Kurulu toplantısı','kategori':'tcmb','onem':'yuksek'},
    {'tarih':'2026-06-11','baslik':'TCMB PPK Faiz Kararı','aciklama':'Para Politikası Kurulu toplantısı','kategori':'tcmb','onem':'yuksek'},
    {'tarih':'2026-07-23','baslik':'TCMB PPK Faiz Kararı','aciklama':'Para Politikası Kurulu toplantısı','kategori':'tcmb','onem':'yuksek'},
    {'tarih':'2026-09-10','baslik':'TCMB PPK Faiz Kararı','aciklama':'Para Politikası Kurulu toplantısı','kategori':'tcmb','onem':'yuksek'},
    {'tarih':'2026-10-22','baslik':'TCMB PPK Faiz Kararı','aciklama':'Para Politikası Kurulu toplantısı','kategori':'tcmb','onem':'yuksek'},
    {'tarih':'2026-12-10','baslik':'TCMB PPK Faiz Kararı','aciklama':'Para Politikası Kurulu toplantısı','kategori':'tcmb','onem':'yuksek'},
    # TÜİK TÜFE & ÜFE (aylık enflasyon)
    {'tarih':'2026-01-05','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Aralık 2025 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-02-03','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Ocak 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-03-03','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Şubat 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-04-03','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Mart 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-05-05','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Nisan 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-06-05','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Mayıs 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-07-03','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Haziran 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-08-03','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Temmuz 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-09-03','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Ağustos 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-10-05','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Eylül 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-11-03','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Ekim 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-12-03','baslik':'TÜİK TÜFE & ÜFE','aciklama':'Kasım 2026 enflasyon verileri','kategori':'tuik','onem':'yuksek'},
    # TÜİK GSYiH
    {'tarih':'2026-02-27','baslik':'TÜİK GSYiH Büyüme (Q4 2025)','aciklama':'2025 dördüncü çeyrek ve yıllık büyüme verisi','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-06-01','baslik':'TÜİK GSYiH Büyüme (Q1 2026)','aciklama':'2026 birinci çeyrek büyüme verisi','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-09-01','baslik':'TÜİK GSYiH Büyüme (Q2 2026)','aciklama':'2026 ikinci çeyrek büyüme verisi','kategori':'tuik','onem':'yuksek'},
    {'tarih':'2026-11-30','baslik':'TÜİK GSYiH Büyüme (Q3 2026)','aciklama':'2026 üçüncü çeyrek büyüme verisi','kategori':'tuik','onem':'yuksek'},
    # FED FOMC (federalreserve.gov resmi 2026 takvimi — karar 2. gun aciklanir)
    {'tarih':'2026-01-28','baslik':'FED Faiz Kararı (FOMC)','aciklama':'Federal Açık Piyasa Komitesi toplantısı ve karar açıklaması','kategori':'fed','onem':'orta'},
    {'tarih':'2026-03-18','baslik':'FED Faiz Kararı (FOMC)','aciklama':'Federal Açık Piyasa Komitesi toplantısı','kategori':'fed','onem':'orta'},
    {'tarih':'2026-04-29','baslik':'FED Faiz Kararı (FOMC)','aciklama':'Federal Açık Piyasa Komitesi toplantısı','kategori':'fed','onem':'orta'},
    {'tarih':'2026-06-17','baslik':'FED Faiz Kararı (FOMC)','aciklama':'Federal Açık Piyasa Komitesi toplantısı','kategori':'fed','onem':'orta'},
    {'tarih':'2026-07-29','baslik':'FED Faiz Kararı (FOMC)','aciklama':'Federal Açık Piyasa Komitesi toplantısı','kategori':'fed','onem':'orta'},
    {'tarih':'2026-09-16','baslik':'FED Faiz Kararı (FOMC)','aciklama':'Federal Açık Piyasa Komitesi toplantısı','kategori':'fed','onem':'orta'},
    {'tarih':'2026-10-28','baslik':'FED Faiz Kararı (FOMC)','aciklama':'Federal Açık Piyasa Komitesi toplantısı','kategori':'fed','onem':'orta'},
    {'tarih':'2026-12-09','baslik':'FED Faiz Kararı (FOMC)','aciklama':'Federal Açık Piyasa Komitesi toplantısı','kategori':'fed','onem':'orta'},
    # BIST Bilanço Sezonları
    {'tarih':'2026-02-13','baslik':'Bilanço Sezonu Başlangıcı — Q4 2025','aciklama':'BIST şirketleri 2025 yıllık bilanço açıklamaları başlıyor','kategori':'bilanco','onem':'orta'},
    {'tarih':'2026-03-31','baslik':'Bilanço Son Tarihi — Yıllık 2025','aciklama':'2025 yıllık bilanço açıklama son tarihi','kategori':'bilanco','onem':'dusuk'},
    {'tarih':'2026-05-15','baslik':'Bilanço Sezonu — Q1 2026','aciklama':'BIST şirketleri 2026 birinci çeyrek bilançoları','kategori':'bilanco','onem':'orta'},
    {'tarih':'2026-08-14','baslik':'Bilanço Sezonu — Q2 2026','aciklama':'BIST şirketleri 2026 ikinci çeyrek bilançoları','kategori':'bilanco','onem':'orta'},
    {'tarih':'2026-11-13','baslik':'Bilanço Sezonu — Q3 2026','aciklama':'BIST şirketleri 2026 üçüncü çeyrek bilançoları','kategori':'bilanco','onem':'orta'},
]

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
    return render_template_string(HTML, hisseler=HISSELER,
                                   supabase_url=SUPABASE_URL,
                                   supabase_anon_key=SUPABASE_ANON_KEY)

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

_finans_cache = {}   # sembol -> (datetime, data)

# Uygulama başlar başlamaz arka plan thread'ini başlat (ziyaretçi beklenmez)
threading.Thread(target=sistem_baslat, daemon=True, name='sinyal-motoru').start()
print("Sinyal motoru başlatıldı.")

@app.route('/api/veri')
def api_veri():
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

@app.route('/api/finans/<sembol>')
def api_finans(sembol):
    global _finans_cache
    sembol_is = sembol + '.IS' if not sembol.endswith('.IS') else sembol
    if sembol in _finans_cache:
        ts, cached = _finans_cache[sembol]
        if (datetime.now() - ts).total_seconds() < 86400:
            return jsonify(cached)
    try:
        ticker = yf.Ticker(sembol_is)
        inc = ticker.quarterly_income_stmt
        bal = ticker.quarterly_balance_sheet
        info = ticker.info or {}

        def safe(df, key, n=5):
            if df is None or df.empty or key not in df.index:
                return [None] * n
            vals = list(df.loc[key].values[:n])
            return [None if (v is None or pd.isna(v)) else int(v / 1000) for v in vals]

        ceyrekler = []
        if inc is not None and not inc.empty:
            ceyrekler = [str(c)[:7].replace('-', '/') for c in list(inc.columns)[:5]]

        fi = safe(inc, 'Interest Income')
        tg = safe(inc, 'Total Revenue')
        try:
            banka = (fi[0] or 0) > (tg[0] or 1) * 0.4
        except Exception:
            banka = False

        if banka:
            gelir = [
                {'etiket': 'Faiz Gelirleri',    'deger': safe(inc, 'Interest Income')},
                {'etiket': 'Faiz Giderleri',    'deger': [(-v if v else None) for v in safe(inc, 'Interest Expense')]},
                {'etiket': 'Net Faiz Geliri',   'deger': safe(inc, 'Net Interest Income')},
                {'etiket': 'Net Faaliyet Kari', 'deger': safe(inc, 'Pretax Income')},
                {'etiket': 'Net Donem Kari',    'deger': safe(inc, 'Net Income')},
            ]
            ana_gelir_key = 'Net Interest Income'
            ana_gelir_lbl = 'Net Faiz Geliri'
        else:
            gelir = [
                {'etiket': 'Toplam Gelir',     'deger': safe(inc, 'Total Revenue')},
                {'etiket': 'Brut Kar',         'deger': safe(inc, 'Gross Profit')},
                {'etiket': 'Faaliyet Kari',    'deger': safe(inc, 'Operating Income')},
                {'etiket': 'Vergi Oncesi Kar', 'deger': safe(inc, 'Pretax Income')},
                {'etiket': 'Net Donem Kari',   'deger': safe(inc, 'Net Income')},
            ]
            ana_gelir_key = 'Total Revenue'
            ana_gelir_lbl = 'Toplam Gelir'

        bilanco = [
            {'etiket': 'Toplam Varliklar',    'deger': safe(bal, 'Total Assets')},
            {'etiket': 'Nakit ve Benzerleri', 'deger': safe(bal, 'Cash And Cash Equivalents')},
            {'etiket': 'Toplam Borc',         'deger': safe(bal, 'Total Debt')},
            {'etiket': 'Yukumlulukler',       'deger': safe(bal, 'Total Liabilities Net Minority Interest')},
            {'etiket': 'Ozkaynaklar',         'deger': safe(bal, 'Stockholders Equity')},
        ]

        roe = info.get('returnOnEquity')
        roa = info.get('returnOnAssets')
        oranlar = {
            'fk':      info.get('trailingPE'),
            'pd_dd':   info.get('priceToBook'),
            'roe':     round(roe * 100, 1) if roe else None,
            'roa':     round(roa * 100, 1) if roa else None,
            'temettu': round(info.get('dividendYield', 0) or 0, 4),
            'beta':    info.get('beta'),
        }

        result = json_temizle({
            'sembol':   sembol,
            'ceyrekler': ceyrekler,
            'banka':    banka,
            'gelir':    gelir,
            'bilanco':  bilanco,
            'oranlar':  oranlar,
            'grafik': {
                'net_kar':   {'etiket': 'Net Donem Kari', 'deger': safe(inc, 'Net Income')},
                'ana_gelir': {'etiket': ana_gelir_lbl,    'deger': safe(inc, ana_gelir_key)},
                'ozkaynak':  {'etiket': 'Ozkaynaklar',   'deger': safe(bal, 'Stockholders Equity')},
            },
        })
        _finans_cache[sembol] = (datetime.now(), result)
        return jsonify(result)
    except Exception as e:
        print(f"[FINANS] {sembol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/takvim')
def api_takvim():
    return jsonify(sorted(EKONOMIK_TAKVIM, key=lambda e: e['tarih']))

@app.route('/api/portfoy/<sid>', methods=['GET'])
def api_portfoy_oku(sid):
    hata = sid_yetki_hatasi(sid)
    if hata: return hata
    data = portfoy_db_oku(sid)
    return jsonify(data if data is not None else [])

@app.route('/api/portfoy/<sid>', methods=['POST'])
def api_portfoy_kaydet(sid):
    hata = sid_yetki_hatasi(sid)
    if hata: return hata
    d = request.get_json() or {}
    ok = portfoy_db_kaydet(sid, d.get('sembol', ''),
                            float(d.get('adet', 0)), float(d.get('maliyet', 0)))
    return jsonify({'ok': ok})

@app.route('/api/portfoy/<sid>/<sembol>', methods=['DELETE'])
def api_portfoy_sil(sid, sembol):
    hata = sid_yetki_hatasi(sid)
    if hata: return hata
    ok = portfoy_db_sil(sid, sembol)
    return jsonify({'ok': ok})

@app.route('/api/alarmlar/<sid>', methods=['GET'])
def api_alarmlar_oku(sid):
    hata = sid_yetki_hatasi(sid)
    if hata: return hata
    data = alarmlar_db_oku(sid)
    return jsonify(data if data is not None else [])

@app.route('/api/alarmlar/<sid>', methods=['POST'])
def api_alarm_ekle(sid):
    hata = sid_yetki_hatasi(sid)
    if hata: return hata
    d = request.get_json() or {}
    new_id = alarm_db_ekle(sid, d.get('sembol', ''), d.get('yon', 'above'),
                            float(d.get('fiyat', 0)))
    return jsonify({'ok': new_id is not None, 'id': new_id})

@app.route('/api/alarmlar/<sid>/<int:alarm_id>', methods=['DELETE'])
def api_alarm_sil(sid, alarm_id):
    hata = sid_yetki_hatasi(sid)
    if hata: return hata
    ok = alarm_db_sil(sid, alarm_id)
    return jsonify({'ok': ok})

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  LIDYA BORSA PLATFORMU BAŞLATILIYOR")
    print(f"  Tarayıcıda aç: http://localhost:{PORT}")
    print("  Durdurmak için CTRL+C")
    print("="*55)

    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
