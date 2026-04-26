import schedule
import time
import subprocess
import sys
from datetime import datetime

# ─── Ayarlar ───────────────────────────────────────────
TARAMA_SAATI = "18:30"   # Her gün bu saatte çalışır (borsa kapanış sonrası)
PYTHON       = sys.executable  # Hangi Python kullanılıyorsa onu kullan
# ───────────────────────────────────────────────────────

def tarama_yap():
    zaman = datetime.now().strftime("%d.%m.%Y %H:%M")
    print(f"\n[{zaman}] Otomatik tarama başlıyor...")
    
    try:
        sonuc = subprocess.run(
            [PYTHON, "bildirim.py"],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        print(sonuc.stdout)
        if sonuc.stderr:
            print(f"UYARI: {sonuc.stderr}")
    except Exception as e:
        print(f"HATA: {e}")
    
    print(f"[{zaman}] Tarama tamamlandı. Sonraki tarama yarın {TARAMA_SAATI}")

def main():
    print("=" * 50)
    print("  BIST Otomatik Sinyal Sistemi Başlatıldı")
    print(f"  Her gün saat {TARAMA_SAATI}'de tarama yapılacak")
    print("  Durdurmak için CTRL+C'ye bas")
    print("=" * 50)

    schedule.every().day.at(TARAMA_SAATI).do(tarama_yap)

    # Başlarken bir kez hemen çalıştır
    print("\nİlk tarama hemen başlıyor...")
    tarama_yap()

    while True:
        schedule.run_pending()
        time.sleep(60)   # Her dakika kontrol et

if __name__ == "__main__":
    main()