import sys
import os
import datetime
import warnings
import sqlite3
import requests
import easyocr
from PIL import ImageGrab
import numpy as np
import re
import yfinance as yf
import pandas as pd
import time
import logging

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QTreeWidget, QTreeWidgetItem, QLabel, QComboBox, QHeaderView, QTextEdit, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

# --- YAHOO FINANCE SUSTURUCUSU ---
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

# ==========================================
# --- GENEL AYARLAR ---
# ==========================================
OZEL_TARIH_KULLAN = False  
BASLANGIC_TARIHI = "2026-06-15" 
BITIS_TARIHI = "2026-07-17"     

ozel_oturum = requests.Session()
ozel_oturum.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

def masaustu_yolunu_bul():
    masaustu_yolu = os.path.expanduser("~")
    for yol in [os.path.join(masaustu_yolu, "OneDrive", "Masaüstü"), os.path.join(masaustu_yolu, "Masaüstü"), os.path.join(masaustu_yolu, "Desktop")]:
        if os.path.exists(yol):
            return yol
    return masaustu_yolu

DB_YOLU = os.path.join(masaustu_yolunu_bul(), "birlesik_taramalar_v20.db")
TXT_RAPOR_YOLU = os.path.join(masaustu_yolunu_bul(), "birlesik_rapor_v20.txt")

# ==========================================
# --- AKILLI SIRALAMA SINIFI ---
# ==========================================
class SortableTreeWidgetItem(QTreeWidgetItem):
    def __lt__(self, other):
        column = self.treeWidget().sortColumn()
        text1 = self.text(column)
        text2 = other.text(column)

        if "📅" in text1 and "📅" in text2:
            return text1 < text2

        def puanla(metin):
            m = str(metin).upper()
            if "YILDIZ" in m or "AL" in m or "✅" in m or "BOĞA" in m or "ÇEKİÇ" in m or "GÜÇLÜ TETİK" in m or "KOPUŞ" in m: return 4
            if "1-3" in m or "TEPKİ" in m: return 3
            if "BEKLE" in m or "➖" in m: return 2
            if "ÇIKIŞ" in m or "SAT" in m or "❌" in m: return 1
            return 0
        
        p1 = puanla(text1)
        p2 = puanla(text2)
        
        if p1 != 0 or p2 != 0:
            if p1 != p2:
                return p1 < p2

        def sayi_yap(metin):
            try:
                temiz = re.sub(r'[^\d\.\-]', '', metin)
                if temiz == '' or temiz == '-': return float('-inf')
                return float(temiz)
            except:
                return float('-inf')

        num1 = sayi_yap(text1)
        num2 = sayi_yap(text2)

        if num1 != float('-inf') and num2 != float('-inf'):
            return num1 < num2

        return str(text1) < str(text2)

# ==========================================
# --- 1. ARKA PLAN TARAMA MOTORU ---
# ==========================================
class TaramaMotoru(QThread):
    log_sinyali = pyqtSignal(str)
    tarama_bitti = pyqtSignal()

    def run(self):
        try:
            self.hisseleri_oku_ve_analiz_et()
        except Exception as hata:
            self.log_sinyali.emit(f"KRİTİK HATA: {hata}")
        finally:
            self.tarama_bitti.emit()

    def veritabani_hazirla(self):
        baglanti = sqlite3.connect(DB_YOLU)
        imlec = baglanti.cursor()
        imlec.execute('''
            CREATE TABLE IF NOT EXISTS yildiz_takip_v20 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tarih TEXT,
                hisse TEXT,
                ana_vade TEXT,
                tetik_sinyal TEXT,
                tetik_4s_sinyal TEXT,
                mfi_degeri REAL,
                bxt_degeri REAL,
                ozel_sinyal TEXT,
                ssl_durum TEXT,
                st_durum TEXT,
                ha_rsi_deger REAL,
                yildiz_sinyal TEXT,
                sha_durum TEXT,
                ao_1w_durum TEXT,
                ao_1g_durum TEXT,
                ao_4s_durum TEXT,
                ao_2s_durum TEXT,
                wt_durum TEXT
            )
        ''')
        baglanti.commit()
        return baglanti

    def tetik_formasyonu_bul(self, data):
        if data.empty or len(data) < 5:
            return "➖"

        o = data['Open'].iloc[-1]
        c = data['Close'].iloc[-1]
        h = data['High'].iloc[-1]
        l = data['Low'].iloc[-1]
        
        o_prev = data['Open'].iloc[-2]
        c_prev = data['Close'].iloc[-2]

        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)

        yutan_boga = (c_prev < o_prev) and (c > o) and (c >= o_prev) and (o <= c_prev)
        cekic = (c > o) and (lower_wick >= 2 * body) and (upper_wick <= 0.5 * body)

        ha_close = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        ha_open = pd.Series(0.0, index=data.index)
        ha_open.iloc[0] = (data['Open'].iloc[0] + data['Close'].iloc[0]) / 2
        
        for i in range(1, len(data)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
            
        ha_low = pd.concat([data['Low'], ha_open, ha_close], axis=1).min(axis=1)

        ha_c_guncel = ha_close.iloc[-1]
        ha_o_guncel = ha_open.iloc[-1]
        ha_l_guncel = ha_low.iloc[-1]

        kusursuz_ha = (ha_c_guncel > ha_o_guncel) and (abs(ha_o_guncel - ha_l_guncel) < (data['Close'].iloc[-1] * 0.001))

        if kusursuz_ha and (yutan_boga or cekic):
            return "GÜÇLÜ TETİK 🔥"
        elif yutan_boga:
            return "YUTAN BOĞA 🚀"
        elif cekic:
            return "ÇEKİÇ 🔨"
        elif kusursuz_ha:
            return "HA KOPUŞ 🟩"
        else:
            return "⏳ BEKLE"

    def ana_sistem_hesapla(self, data):
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        if data.empty or len(data) < 36:
            return None, None, False, False, False, None, False, False, 0

        high, low = data['High'].squeeze(), data['Low'].squeeze()
        mid_price = (high + low) / 2
        sma_5, sma_35 = mid_price.rolling(window=5).mean(), mid_price.rolling(window=35).mean()
        
        ao_series = (sma_5 - sma_35).dropna()
        if len(ao_series) < 3: return None, None, False, False, False, None, False, False, 0
            
        ao_guncel_deger, ao_onceki_deger = float(ao_series.iloc[-1]), float(ao_series.iloc[-2])
        ao_kirmizi = ao_guncel_deger < ao_onceki_deger

        ha_close = (data['Open'].squeeze() + data['High'].squeeze() + data['Low'].squeeze() + data['Close'].squeeze()) / 4
        ema_5, ema_20 = ha_close.ewm(span=5, adjust=False).mean(), ha_close.ewm(span=20, adjust=False).mean()
        
        bxt_series = (ema_5 - ema_20).dropna()
        if len(bxt_series) < 3: return None, None, False, False, False, None, False, False, 0
            
        bxt_guncel = float(bxt_series.iloc[-1])
        bxt_onceki = float(bxt_series.iloc[-2])
        bxt_2_onceki = float(bxt_series.iloc[-3])
        bxt_mavi = bxt_guncel > bxt_onceki
        
        mavi_mum_sayaci = 0
        for i in range(len(bxt_series)-1, 0, -1):
            if float(bxt_series.iloc[i]) > float(bxt_series.iloc[i-1]):
                mavi_mum_sayaci += 1
            else:
                break

        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        delta = typical_price.diff()

        positive_flow = pd.Series(np.where(delta > 0, money_flow, 0), index=data.index)
        negative_flow = pd.Series(np.where(delta < 0, money_flow, 0), index=data.index)
        mf_ratio = positive_flow.rolling(window=14).sum() / negative_flow.rolling(window=14).sum().replace(0, np.nan)
        mfi_series = (100 - (100 / (1 + mf_ratio))).fillna(100)

        mfi_guncel = float(mfi_series.iloc[-1])
        yeni_mavi_sinyali = (bxt_onceki <= bxt_2_onceki) and (bxt_guncel > bxt_onceki)
        kesin_al_sinyali = (bxt_guncel < 0) and yeni_mavi_sinyali and (mfi_guncel < 80)
        kesin_sat_sinyali = (bxt_guncel > 0) and (bxt_onceki >= bxt_2_onceki) and (bxt_guncel < bxt_onceki) and (mfi_guncel > 80)

        return ao_guncel_deger, bxt_guncel, bxt_mavi, ao_kirmizi, kesin_al_sinyali, mfi_guncel, yeni_mavi_sinyali, kesin_sat_sinyali, mavi_mum_sayaci

    def get_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
        
    def calc_wma(self, series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)

    def ozel_ve_yildiz_hesapla(self, data):
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
        if data.empty or len(data) < 95:
            return "-", "-", 0.0, "BEKLE", "-", "-", "-", "BEKLE"

        close = data['Close']
        high = data['High']
        low = data['Low']
        open_p = data['Open']

        sma_high = high.rolling(10).mean()
        sma_low = low.rolling(10).mean()
        hlv = pd.Series(0, index=data.index)
        hlv[close > sma_high] = 1
        hlv[close < sma_low] = -1
        hlv = hlv.replace(0, np.nan).ffill()
        ssl_down = np.where(hlv < 0, sma_high, sma_low)
        ssl_up = np.where(hlv < 0, sma_low, sma_high)
        data['SSL_Up'] = ssl_up
        data['SSL_Down'] = ssl_down

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_st = tr.rolling(10).mean()
        hl2 = (high + low) / 2
        final_ub = hl2 + (2.5 * atr_st)
        final_lb = hl2 - (2.5 * atr_st)
        st_dir = pd.Series(1, index=data.index)
        for i in range(1, len(data)):
            if close.iloc[i-1] <= final_ub.iloc[i-1]:
                final_ub.iloc[i] = min(final_ub.iloc[i], final_ub.iloc[i-1])
            if close.iloc[i-1] >= final_lb.iloc[i-1]:
                final_lb.iloc[i] = max(final_lb.iloc[i], final_lb.iloc[i-1])

            if close.iloc[i] > final_ub.iloc[i-1]: st_dir.iloc[i] = 1
            elif close.iloc[i] < final_lb.iloc[i-1]: st_dir.iloc[i] = -1
            else: st_dir.iloc[i] = st_dir.iloc[i-1]

        rsi = self.get_rsi(close, 14)
        rsi_o = self.get_rsi(open_p)
        rsi_h = self.get_rsi(high)
        rsi_l = self.get_rsi(low)
        ha_c = (rsi_o + rsi_h + rsi_l + rsi) / 4

        ssl_kesisim_oldu_mu = False
        for i in range(1, 6):
            if len(data) < i + 1: break
            curr, prev = -i, -i - 1
            ssl_up_g, ssl_down_g = data['SSL_Up'].iloc[curr], data['SSL_Down'].iloc[curr]
            ssl_up_o, ssl_down_o = data['SSL_Up'].iloc[prev], data['SSL_Down'].iloc[prev]
            ssl_kesisim = (ssl_up_g > ssl_down_g) and (ssl_up_o <= ssl_down_o)
            if ssl_kesisim and ha_c.iloc[curr] < 35:
                ssl_kesisim_oldu_mu = True
                break

        ozel_strateji_al = ssl_kesisim_oldu_mu and st_dir.iloc[-1] == 1
        ssl_durum = "YUKARI" if data['SSL_Up'].iloc[-1] > data['SSL_Down'].iloc[-1] else "AŞAĞI"
        st_durum = "BUY" if st_dir.iloc[-1] == 1 else "SELL"
        ha_rsi_guncel = float(ha_c.iloc[-1])
        ozel_sinyal = "AL" if ozel_strateji_al else "BEKLE"

        ema_o_sha = open_p.ewm(span=7, adjust=False).mean()
        ema_h_sha = high.ewm(span=7, adjust=False).mean()
        ema_l_sha = low.ewm(span=7, adjust=False).mean()
        ema_c_sha = close.ewm(span=7, adjust=False).mean()
        ha_close_sha = (ema_o_sha + ema_h_sha + ema_l_sha + ema_c_sha) / 4
        ha_open_sha = pd.Series(0.0, index=data.index)
        ha_open_sha.iloc[0] = (ema_o_sha.iloc[0] + ema_c_sha.iloc[0]) / 2
        for i in range(1, len(data)):
            ha_open_sha.iloc[i] = (ha_open_sha.iloc[i-1] + ha_close_sha.iloc[i-1]) / 2
            
        sha_close = ha_close_sha.ewm(span=10, adjust=False).mean()
        sha_open = ha_open_sha.ewm(span=10, adjust=False).mean()
        sha_yesil = sha_close.iloc[-1] > sha_open.iloc[-1]
        sha_durum = "YEŞİL" if sha_yesil else "KIRMIZI"

        wma_half = self.calc_wma(close, int(55/2))
        wma_full = self.calc_wma(close, 55)
        raw_hma = 2 * wma_half - wma_full
        hma_55 = self.calc_wma(raw_hma, int(np.sqrt(55)))
        
        hull_dondu = False
        hull_yesil_guncel = hma_55.iloc[-1] > hma_55.iloc[-2]
        hull_durum = "YEŞİL" if hull_yesil_guncel else "KIRMIZI"

        for i in range(1, 6):
            if len(hma_55) > i + 2:
                if hma_55.iloc[-i] > hma_55.iloc[-i-1] and hma_55.iloc[-i-1] <= hma_55.iloc[-i-2]:
                    hull_dondu = True
                    break

        ap_wt = (high + low + close) / 3.0
        esa_wt = ap_wt.ewm(span=10, adjust=False).mean()
        d_wt = (ap_wt - esa_wt).abs().ewm(span=10, adjust=False).mean()
        ci_wt = (ap_wt - esa_wt) / (0.015 * d_wt.replace(0, np.nan))
        ci_wt = ci_wt.fillna(0)
        wt1 = ci_wt.ewm(span=18, adjust=False).mean()
        wt2 = wt1.rolling(window=4).mean()
        
        wt1_g = float(wt1.iloc[-1])
        wt_kesisim_yakin = False
        for i in range(1, 6):
            if len(wt1) > i + 1:
                if wt1.iloc[-i] > wt2.iloc[-i] and wt1.iloc[-i-1] <= wt2.iloc[-i-1] and wt1.iloc[-i] < -40:
                    wt_kesisim_yakin = True
                    break

        wt_durum = f"{wt1_g:.1f}" + (" (Dip)" if wt_kesisim_yakin else "")
        yildiz_al_durumu = sha_yesil and hull_dondu and wt_kesisim_yakin
        yildiz_sinyal = "AL" if yildiz_al_durumu else "BEKLE"

        return ssl_durum, st_durum, ha_rsi_guncel, ozel_sinyal, sha_durum, hull_durum, wt_durum, yildiz_sinyal

    def guvenli_veri_indir(self, hisse_kodu, period, interval):
        eski_stdout = sys.stdout
        eski_stderr = sys.stderr
        for deneme in range(3):
            try:
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                
                if OZEL_TARIH_KULLAN:
                    data = yf.download(f"{hisse_kodu}.IS", start=BASLANGIC_TARIHI, end=BITIS_TARIHI, interval=interval, session=ozel_oturum, progress=False)
                    if data.empty: data = yf.download(hisse_kodu, start=BASLANGIC_TARIHI, end=BITIS_TARIHI, interval=interval, session=ozel_oturum, progress=False)
                else:
                    data = yf.download(f"{hisse_kodu}.IS", period=period, interval=interval, session=ozel_oturum, progress=False)
                    if data.empty: data = yf.download(hisse_kodu, period=period, interval=interval, session=ozel_oturum, progress=False)
                
                sys.stdout = eski_stdout
                sys.stderr = eski_stderr
                
                if data is not None and not data.empty:
                    return data
            except Exception:
                sys.stdout = eski_stdout
                sys.stderr = eski_stderr
                time.sleep(1) 
        return pd.DataFrame()

    def periyot_verilerini_cek(self, hisse_kodu):
        # 1 Haftalık (1W) Veri
        data_1w = self.guvenli_veri_indir(hisse_kodu, period="2y", interval="1wk")
        if not data_1w.empty and isinstance(data_1w.columns, pd.MultiIndex): data_1w.columns = data_1w.columns.get_level_values(0)
        try: a_1w = self.ana_sistem_hesapla(data_1w)
        except: a_1w = (None, None, False, False, False, None, False, False, 0)

        # 1 Günlük (1D) Veri
        data_1g = self.guvenli_veri_indir(hisse_kodu, period="1y", interval="1d")
        if not data_1g.empty and isinstance(data_1g.columns, pd.MultiIndex): data_1g.columns = data_1g.columns.get_level_values(0)
        try: a_1g = self.ana_sistem_hesapla(data_1g)
        except: a_1g = (None, None, False, False, False, None, False, False, 0)
        try: ozel_ve_yildiz_1g = self.ozel_ve_yildiz_hesapla(data_1g)
        except: ozel_ve_yildiz_1g = ("-", "-", 0.0, "BEKLE", "-", "-", "-", "BEKLE")

        # 1 Saatlik Veriden 2S ve 4S Türetme
        data_1h = self.guvenli_veri_indir(hisse_kodu, period="3mo", interval="1h")
        if not data_1h.empty:
            if isinstance(data_1h.columns, pd.MultiIndex): data_1h.columns = data_1h.columns.get_level_values(0)
            if data_1h.index.tz is not None: data_1h.index = data_1h.index.tz_localize(None)
            
            data_2s = data_1h[['Open', 'High', 'Low', 'Close', 'Volume']].resample('2h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            try: a_2s = self.ana_sistem_hesapla(data_2s)
            except: a_2s = (None, None, False, False, False, None, False, False, 0)
            try: tetik_2s = self.tetik_formasyonu_bul(data_2s)
            except: tetik_2s = "BEKLE"
            
            data_4s = data_1h[['Open', 'High', 'Low', 'Close', 'Volume']].resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            try: a_4s = self.ana_sistem_hesapla(data_4s)
            except: a_4s = (None, None, False, False, False, None, False, False, 0)
            try: tetik_4s = self.tetik_formasyonu_bul(data_4s)
            except: tetik_4s = "BEKLE"
        else:
            a_2s = (None, None, False, False, False, None, False, False, 0)
            a_4s = (None, None, False, False, False, None, False, False, 0)
            tetik_2s = "BEKLE"
            tetik_4s = "BEKLE"

        return a_1w, a_1g, a_2s, a_4s, ozel_ve_yildiz_1g, tetik_2s, tetik_4s

    def hisseleri_oku_ve_analiz_et(self):
        self.log_sinyali.emit("Panodaki (Clipboard) görsel kontrol ediliyor...")
        img = ImageGrab.grabclipboard()
        
        if img is None:
            self.log_sinyali.emit("HATA: Panoda bir görsel bulunamadı. Lütfen bir hisse listesi kopyalayın.")
            return

        self.log_sinyali.emit("Görsel okundu, OCR ile hisseler ayrıştırılıyor... Lütfen bekleyin.")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False) 
        okunan_metin = " ".join(reader.readtext(np.array(img.convert('RGB')), detail=0))
        ham_hisseler = re.findall(r'\b[A-Z]{2,5}\b', okunan_metin)
        
        yasakli_kelimeler = {'INC', 'LLC', 'CORP', 'LTD', 'CO', 'THE', 'AND', 'BIST', 'TCMB', 'POLDY', 'GIB', 'YATIRIM', 'FON', 'AS', 'A.S', 'USD', 'EUR', 'TRY', 'SAAT', 'GUN', 'HAFTA'}
        temiz_hisseler = sorted(list(set(h for h in ham_hisseler if h not in yasakli_kelimeler)))
        
        if temiz_hisseler:
            self.log_sinyali.emit(f"\nToplam {len(temiz_hisseler)} hisse tespit edildi. Sistemler çapraz sorgulanıyor...\n")
            zaman_tam = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            
            db_baglanti = self.veritabani_hazirla()
            db_imlec = db_baglanti.cursor()
            
            with open(TXT_RAPOR_YOLU, "a", encoding="utf-8") as dosya:
                dosya.write(f"\n{'='*85}\nTARAMA TARİHİ: {zaman_tam}\n{'='*85}\n")
                
                for hisse in temiz_hisseler:
                    a_1w, a_1g, a_2s, a_4s, ozel_ve_yildiz_1g, tetik_2s, tetik_4s = self.periyot_verilerini_cek(hisse)
                    
                    ao_1w, _, _, ao_kirmizi_1w, _, _, _, _, _ = a_1w
                    ao_1g, bxt_1g, mavi_1g, ao_kirmizi_1g, kesin_al_1g, mfi_1g, _, k_sat_1g, mavi_mum_1g = a_1g
                    ao_4s, _, _, ao_kirmizi_4s, _, _, _, _, _ = a_4s
                    ao_2s, bxt_2s, mavi_2s, ao_kirmizi_2s, kesin_al_2s, mfi_2s, _, k_sat_2s, mavi_mum_2s = a_2s
                    ssl_1g, st_1g, ha_rsi_1g, ozel_s_1g, sha_1g, hull_1g, wt_1g, yildiz_s_1g = ozel_ve_yildiz_1g
                    
                    skor = 1
                    if mavi_1g:
                        if 0 < mavi_mum_1g <= 6: skor += 2
                        elif mavi_mum_1g > 6: skor += 1

                    if mavi_2s: skor += 1
                    if mfi_2s is not None and mfi_1g is not None:
                        if mfi_2s < 80 and mfi_1g < 80: skor += 1
                        elif mfi_2s > 80 or mfi_1g > 80: skor -= 1
                    
                    skor = max(1, min(5, skor))
                    
                    ozel_yildiz_sarti = mavi_2s and (ao_2s is not None and ao_2s > 0) and (bxt_2s is not None and bxt_2s > 0)
                    if k_sat_1g or k_sat_2s: vade = "ÇIKIŞ VAKTİ 🛑"
                    elif ozel_yildiz_sarti: vade = f"🌟 YILDIZ 🌟 ({mavi_mum_2s}. Mum)"
                    elif skor >= 4 and mavi_1g: vade = "1-3 Hafta (Ana Trend)"
                    elif skor >= 3 and kesin_al_2s: vade = "1-3 Gün (Vur-Kaç)"
                    elif mavi_2s: vade = "1-2 Gün (Tepki)"
                    elif skor <= 2: vade = "İzleme Modu"
                    else: vade = "Belirsiz"
                    
                    # AO Değerlerinin Hazırlanması
                    ao_durum_1w = f"{ao_1w:.2f} KIRMIZI" if ao_1w is not None and ao_kirmizi_1w else f"{ao_1w:.2f} YEŞİL" if ao_1w is not None else "-"
                    ao_durum_1g = f"{ao_1g:.2f} KIRMIZI" if ao_1g is not None and ao_kirmizi_1g else f"{ao_1g:.2f} YEŞİL" if ao_1g is not None else "-"
                    ao_durum_4s = f"{ao_4s:.2f} KIRMIZI" if ao_4s is not None and ao_kirmizi_4s else f"{ao_4s:.2f} YEŞİL" if ao_4s is not None else "-"
                    ao_durum_2s = f"{ao_2s:.2f} KIRMIZI" if ao_2s is not None and ao_kirmizi_2s else f"{ao_2s:.2f} YEŞİL" if ao_2s is not None else "-"

                    kutu = f"""
{'='*85}
HİSSE: {hisse}
{'='*85}
[ 1. ANA SİSTEM ] -> {vade}
MFI: {f"{mfi_1g:.1f}" if mfi_1g else "-"} | BXT: {f"{bxt_1g:.2f}" if bxt_1g else "-"}

[ 2. ÖZEL STRATEJİ ] -> SİNYAL: {ozel_s_1g}
SSL: {ssl_1g} | S.Trend: {st_1g} | HA-RSI: {ha_rsi_1g:.1f}

[ 3. IŞILDAYAN YILDIZ ] -> SİNYAL: {yildiz_s_1g}
S.HA: {sha_1g} | AO 1G: {ao_durum_1g} | WT: {wt_1g}

[ 4. TETİKLEYİCİLER ] -> 2 SAAT: {tetik_2s} | 4 SAAT: {tetik_4s}
"""
                    self.log_sinyali.emit(kutu)
                    dosya.write(kutu + "\n")
                    
                    db_imlec.execute('''
                        INSERT INTO yildiz_takip_v20 (
                            tarih, hisse, ana_vade, tetik_sinyal, tetik_4s_sinyal, mfi_degeri, bxt_degeri, 
                            ozel_sinyal, ssl_durum, st_durum, ha_rsi_deger, 
                            yildiz_sinyal, sha_durum, ao_1w_durum, ao_1g_durum, ao_4s_durum, ao_2s_durum, wt_durum
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (zaman_tam, hisse, vade, tetik_2s, tetik_4s, mfi_1g, bxt_1g, ozel_s_1g, ssl_1g, st_1g, ha_rsi_1g, yildiz_s_1g, sha_1g, ao_durum_1w, ao_durum_1g, ao_durum_4s, ao_durum_2s, wt_1g))
                
                ozet_mesaji = f"\n🔥 TARAMA ÖZETİ - İNCELENEN HİSSE: {len(temiz_hisseler)}"
                self.log_sinyali.emit(ozet_mesaji)
                dosya.write(ozet_mesaji + "\n")
                        
            db_baglanti.commit(); db_baglanti.close()
            self.log_sinyali.emit(f"\n{'='*85}\nANALİZ TAMAMLANDI. Veritabanı (v20) güncellendi.")
        else:
            self.log_sinyali.emit("Panoda geçerli bir hisse kodu tespit edilemedi.")

# ==========================================
# --- 2. CANLI FİYAT ÇEKİCİ (QThread) ---
# ==========================================
class FiyatGuncelleyici(QThread):
    fiyat_geldi = pyqtSignal(str, float, float) 
    bitti = pyqtSignal()

    def __init__(self, hisseler):
        super().__init__()
        self.hisseler = hisseler

    def run(self):
        eski_stdout = sys.stdout
        eski_stderr = sys.stderr
        for hisse in self.hisseler:
            try:
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                
                ticker = yf.Ticker(f"{hisse}.IS")
                data = ticker.history(period="5d")
                data = data.dropna(subset=['Close']) 
                
                sys.stdout = eski_stdout
                sys.stderr = eski_stderr

                if len(data) >= 2:
                    guncel_fiyat = float(data['Close'].iloc[-1])
                    onceki_kapanis = float(data['Close'].iloc[-2])
                    degisim = ((guncel_fiyat - onceki_kapanis) / onceki_kapanis) * 100 if onceki_kapanis > 0 else 0.0
                elif len(data) == 1:
                    guncel_fiyat = float(data['Close'].iloc[-1])
                    degisim = 0.0
                else:
                    continue
                
                if pd.isna(guncel_fiyat) or pd.isna(degisim):
                    continue

                self.fiyat_geldi.emit(hisse, guncel_fiyat, degisim)
            except Exception:
                sys.stdout = eski_stdout
                sys.stderr = eski_stderr
        self.bitti.emit()

# ==========================================
# --- 3. ANA ARAYÜZ (GUI) ---
# ==========================================
class MasterRadarArayuz(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌟 Birleşik Radar & Manuel Karar Terminali (v20 - Multi-Timeframe AO)")
        self.resize(1900, 850) 
        self.setStyleSheet("background-color: #0b0b0b; color: #e0e0e0;")
        self.initUI()
        self.verileri_yukle()

    def initUI(self):
        merkez = QWidget()
        self.setCentralWidget(merkez)
        ana_layout = QVBoxLayout(merkez)

        ust_panel = QHBoxLayout()
        lbl_baslik = QLabel("Tarih Filtresi:")
        lbl_baslik.setFont(QFont("Arial", 10, QFont.Bold))
        self.combo_tarih = QComboBox()
        self.combo_tarih.setStyleSheet("background-color: #1a1a1a; padding: 5px; border: 1px solid #333;")
        self.combo_tarih.currentTextChanged.connect(self.filtrele)
        
        self.btn_fiyat = QPushButton("🔄 Canlı Fiyatları Çek")
        self.btn_fiyat.setStyleSheet("background-color: #1b3b22; color: #a3ffb1; font-weight: bold; padding: 8px; border: 1px solid #2e6938;")
        self.btn_fiyat.clicked.connect(self.fiyatlari_baslat)

        self.btn_tara = QPushButton("📋 PANODAN YENİ TARAMA BAŞLAT")
        self.btn_tara.setStyleSheet("background-color: #2b4b7c; color: #a3ccff; font-weight: bold; padding: 8px; border: 1px solid #3c6ab5;")
        self.btn_tara.clicked.connect(self.tarama_baslat)

        ust_panel.addWidget(lbl_baslik)
        ust_panel.addWidget(self.combo_tarih)
        ust_panel.addStretch()
        ust_panel.addWidget(self.btn_fiyat)
        ust_panel.addWidget(self.btn_tara)
        ana_layout.addLayout(ust_panel)

        splitter = QSplitter(Qt.Vertical)

        self.log_ekrani = QTextEdit()
        self.log_ekrani.setReadOnly(True)
        self.log_ekrani.setFont(QFont("Courier New", 9))
        self.log_ekrani.setStyleSheet("background-color: #050505; color: #00ff00; border: 1px solid #333;")
        self.log_ekrani.append("--- SİSTEM HAZIR ---\nGrafiği kopyalayıp 'Panodan Yeni Tarama Başlat' butonuna basın.")
        splitter.addWidget(self.log_ekrani)

        self.agac = QTreeWidget()
        self.agac.setColumnCount(19)
        # Sütunlara yeni AO periyotları eklendi
        self.agac.setHeaderLabels([
            "Hisse Adı", "[1] Ana Sistem", "[2] Özel Strateji", "[3] Yıldız Sistemi", "[4] 2S Tetik", "[5] 4S Tetik",
            "MFI", "BXT", "SSL", "S.Trend", "HA-RSI", 
            "S.HA", "AO (1Hft)", "AO (1G)", "AO (4S)", "AO (2S)", "WaveTrend", 
            "Fiyat (₺)", "Günlük Değişim"
        ])
        
        self.agac.setSortingEnabled(True) 
        self.agac.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.agac.setStyleSheet("""
            QTreeWidget { background-color: #121212; border: 1px solid #222; }
            QHeaderView::section { background-color: #222; color: #fff; padding: 5px; border: 1px solid #333; font-weight: bold;}
            QTreeWidget::item { padding: 5px; border-right: 1px solid #2a2a2a; }
            QToolTip { 
                background-color: #121212; 
                color: #a3ffb1; 
                border: 1px solid #ffaa00; 
                font-family: Consolas, Courier New; 
                font-size: 14px; 
                padding: 10px; 
            }
        """)
        splitter.addWidget(self.agac)
        
        splitter.setSizes([250, 600]) 
        ana_layout.addWidget(splitter)

    def dinamik_yorum_uret(self, hisse, vade, mfi, bxt, ozel_s, ssl, st, ha_rsi, yildiz_s, sha, ao_1w, ao_1g, ao_4s, ao_2s, wt, t2, t4):
        metin = f"========== {hisse} DETAYLI SENTEZ ANALİZİ ==========\n\n"
        
        metin += f"📊 1. ANA SİSTEM: Neden '{vade}' Diyor?\n"
        try:
            mfi_val = float(mfi)
            if mfi_val < 40: metin += f"• MFI ({mfi_val:.1f}): Para akışı düşük. Hisse aşırı satım bölgesinde.\n"
            elif mfi_val > 80: metin += f"• MFI ({mfi_val:.1f}): Para akışı çok yüksek. Hisse aşırı alım bölgesinde.\n"
            else: metin += f"• MFI ({mfi_val:.1f}): Para akışı nötr seviyelerde seyrediyor.\n"
        except: pass

        try:
            bxt_val = float(bxt)
            if bxt_val > 0: metin += f"• BXT ({bxt_val:.2f}): Trend göstergesi pozitif bölgede, ana ivme net bir şekilde yukarı yönlü.\n"
            else: metin += f"• BXT ({bxt_val:.2f}): Trend negatif bölgede olmasına rağmen toparlanma ivmesi başlamış.\n"
        except: pass

        metin += f"\n🎯 MULTI-TIMEFRAME AO (AWESOME OSCILLATOR) İVMESİ\n"
        metin += f"• Haftalık İvme (Ana Yön) : {ao_1w}\n"
        metin += f"• Günlük İvme   (Mevcut)  : {ao_1g}\n"
        metin += f"• 4 Saatlik İvme(Kısa)    : {ao_4s}\n"
        metin += f"• 2 Saatlik İvme(Mikro)   : {ao_2s}\n"

        metin += f"\n🌟 2. ÖZEL STRATEJİ & IŞILDAYAN YILDIZ\n"
        if "BEKLE" in str(ozel_s) or "BEKLE" in str(yildiz_s):
            metin += "• Ağır Abiler: Günlük grafikteki ana trend takipçileri (SSL, S.Trend) henüz dönüş yapamamış (BEKLE).\n"
            
            try: hr_val = float(ha_rsi)
            except: hr_val = 50.0
            
            if hr_val < 35 or (wt is not None and "Dip" in str(wt)):
                wt_temiz = str(wt).split()[0] if wt else "-"
                metin += f"• Gizli Hazineler: Ancak alt indikatörler (HA-RSI: {hr_val:.1f}, WT: {wt_temiz}) tarihi diplerde.\n"
        else:
            metin += "• Sistem Onayı: Ağır hareketli ortalamalar ve trend takipçileri pozitif (AL) bölgeye geçmiş.\n"

        metin += f"\n⚡ 3. TETİKLEYİCİLER (2S & 4S)\n"
        if "BEKLE" not in str(t2) or "BEKLE" not in str(t4):
            metin += f"• {t2} / {t4}: Alt periyotlarda (2S/4S) kusursuz mum formasyonları (Yutan Boğa/Kopuş) oluşmuş.\n"
        else:
            metin += "• Şu an kısa vadede (2S ve 4S) net bir dönüş (kopuş) mumu görülmüyor.\n"
            
        return metin

    def verileri_yukle(self):
        if not os.path.exists(DB_YOLU): return

        try:
            baglanti = sqlite3.connect(DB_YOLU)
            imlec = baglanti.cursor()
            imlec.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='yildiz_takip_v20'")
            if not imlec.fetchone(): return

            imlec.execute("SELECT tarih, hisse, ana_vade, tetik_sinyal, tetik_4s_sinyal, mfi_degeri, bxt_degeri, ozel_sinyal, ssl_durum, st_durum, ha_rsi_deger, yildiz_sinyal, sha_durum, ao_1w_durum, ao_1g_durum, ao_4s_durum, ao_2s_durum, wt_durum FROM yildiz_takip_v20 ORDER BY id DESC")
            kayitlar = imlec.fetchall()
            baglanti.close()

            self.gunluk_veriler = {}
            tarihler = set()
            eklenen_hisseler = set() 

            for kayit in kayitlar:
                sadece_gun = kayit[0].split(" ")[0]
                hisse_kodu = kayit[1]
                
                benzersiz_id = f"{sadece_gun}_{hisse_kodu}"
                if benzersiz_id in eklenen_hisseler: continue
                eklenen_hisseler.add(benzersiz_id)

                tarihler.add(sadece_gun)
                if sadece_gun not in self.gunluk_veriler: self.gunluk_veriler[sadece_gun] = []
                self.gunluk_veriler[sadece_gun].append(kayit)

            self.combo_tarih.blockSignals(True)
            self.combo_tarih.clear()
            self.combo_tarih.addItem("Tüm Zamanlar")
            for t in sorted(list(tarihler), reverse=True): self.combo_tarih.addItem(t)
            self.combo_tarih.blockSignals(False)

            self.agaci_doldur(self.combo_tarih.currentText())
        except Exception as e:
            self.log_yaz(f"Veritabanı Okuma Hatası: {e}")

    def agaci_doldur(self, secili_tarih):
        self.agac.setSortingEnabled(False)
        self.agac.clear()
        self.hisse_item_map = {}
        if not hasattr(self, 'gunluk_veriler'): return

        for tarih, hisseler in self.gunluk_veriler.items():
            if secili_tarih != "Tüm Zamanlar" and tarih != secili_tarih: continue

            tarih_kok = SortableTreeWidgetItem(self.agac, [f"📅 {tarih}"] + [""]*18)
            tarih_kok.setFont(0, QFont("Arial", 11, QFont.Bold))
            tarih_kok.setForeground(0, QColor("#e3a130"))
            
            for h in hisseler:
                tam_tarih, hisse_kodu, ana_vade, tetik_sinyali, tetik_4s_sinyali, mfi, bxt, ozel_s, ssl, st, ha_rsi, yildiz_s, sha, ao_1w_durum, ao_1g_durum, ao_4s_durum, ao_2s_durum, wt = h
                
                s1_sinyal = ana_vade 
                s2_sinyal = f"🚀 {ozel_s}" if "AL" in str(ozel_s) else "⏳ BEKLE"
                s3_sinyal = f"🌟 {yildiz_s}" if "AL" in str(yildiz_s) else "⏳ BEKLE"
                s4_sinyal = f"{tetik_sinyali}"
                s5_sinyal = f"{tetik_4s_sinyali}"
                
                try: mfi_val = float(mfi)
                except: mfi_val = None
                mfi_str = f"{mfi_val:.1f} ✅" if mfi_val is not None and mfi_val < 80 else f"{mfi_val:.1f} ❌" if mfi_val is not None else "-"
                
                try: bxt_val = float(bxt)
                except: bxt_val = None
                bxt_str = f"{bxt_val:.2f} ✅" if bxt_val is not None and bxt_val > 0 else f"{bxt_val:.2f} ❌" if bxt_val is not None else "-"
                
                ssl_str = f"{ssl} ✅" if ssl and "YUKARI" in str(ssl) else f"{ssl} ❌" if ssl and "AŞAĞI" in str(ssl) else "-"
                st_str = f"{st} ✅" if st and "BUY" in str(st) else f"{st} ❌" if st and "SELL" in str(st) else "-"
                
                try: ha_rsi_val = float(ha_rsi)
                except: ha_rsi_val = None
                ha_rsi_str = f"{ha_rsi_val:.1f} ✅ (Dip)" if ha_rsi_val is not None and ha_rsi_val <= 35 else f"{ha_rsi_val:.1f} ❌" if ha_rsi_val is not None and ha_rsi_val >= 70 else f"{ha_rsi_val:.1f} ➖" if ha_rsi_val is not None else "-"
                
                sha_str = f"{sha} ✅" if sha and "YEŞİL" in str(sha) else f"{sha} ❌" if sha and "KIRMIZI" in str(sha) else "-"
                
                ao_1w_str = f"{ao_1w_durum} ✅" if ao_1w_durum and "YEŞİL" in str(ao_1w_durum) else f"{ao_1w_durum} ❌" if ao_1w_durum and "KIRMIZI" in str(ao_1w_durum) else "-"
                ao_1g_str = f"{ao_1g_durum} ✅" if ao_1g_durum and "YEŞİL" in str(ao_1g_durum) else f"{ao_1g_durum} ❌" if ao_1g_durum and "KIRMIZI" in str(ao_1g_durum) else "-"
                ao_4s_str = f"{ao_4s_durum} ✅" if ao_4s_durum and "YEŞİL" in str(ao_4s_durum) else f"{ao_4s_durum} ❌" if ao_4s_durum and "KIRMIZI" in str(ao_4s_durum) else "-"
                ao_2s_str = f"{ao_2s_durum} ✅" if ao_2s_durum and "YEŞİL" in str(ao_2s_durum) else f"{ao_2s_durum} ❌" if ao_2s_durum and "KIRMIZI" in str(ao_2s_durum) else "-"
                
                wt_str = f"{wt} ✅" if wt and "Dip" in str(wt) else f"{wt} ❌" if wt else "-"

                hisse_item = SortableTreeWidgetItem(tarih_kok, [
                    f"🏷️ {hisse_kodu}", 
                    s1_sinyal, 
                    s2_sinyal, 
                    s3_sinyal, 
                    s4_sinyal,
                    s5_sinyal,
                    mfi_str, 
                    bxt_str, 
                    ssl_str, 
                    st_str, 
                    ha_rsi_str, 
                    sha_str, 
                    ao_1w_str,
                    ao_1g_str,
                    ao_4s_str,
                    ao_2s_str, 
                    wt_str, 
                    "Bekleniyor...", 
                    "-"
                ])
                hisse_item.setFont(0, QFont("Arial", 12, QFont.Bold))
                
                if "YILDIZ" in s1_sinyal or "Tepki" in s1_sinyal or "1-3" in s1_sinyal: hisse_item.setForeground(1, QColor("#00ff00"))
                elif "ÇIKIŞ" in s1_sinyal: hisse_item.setForeground(1, QColor("#ff4444"))
                else: hisse_item.setForeground(1, QColor("#a3ccff"))
                
                if "AL" in s2_sinyal: hisse_item.setForeground(2, QColor("#00ff00"))
                else: hisse_item.setForeground(2, QColor("#e3a130"))
                
                if "AL" in s3_sinyal: hisse_item.setForeground(3, QColor("#FFD700"))
                else: hisse_item.setForeground(3, QColor("#e3a130"))

                if "BEKLE" not in s4_sinyal and "➖" not in s4_sinyal: hisse_item.setForeground(4, QColor("#00ffff"))
                else: hisse_item.setForeground(4, QColor("#e3a130"))
                
                if "BEKLE" not in s5_sinyal and "➖" not in s5_sinyal: hisse_item.setForeground(5, QColor("#00ffff"))
                else: hisse_item.setForeground(5, QColor("#e3a130"))
                
                if "✅" in mfi_str: hisse_item.setForeground(6, QColor("#00ff00"))
                else: hisse_item.setForeground(6, QColor("#ff4444"))
                if "✅" in bxt_str: hisse_item.setForeground(7, QColor("#00ff00"))
                else: hisse_item.setForeground(7, QColor("#ff4444"))
                if "✅" in ssl_str: hisse_item.setForeground(8, QColor("#00ff00"))
                else: hisse_item.setForeground(8, QColor("#ff4444"))
                if "✅" in st_str: hisse_item.setForeground(9, QColor("#00ff00"))
                else: hisse_item.setForeground(9, QColor("#ff4444"))
                if "✅" in ha_rsi_str: hisse_item.setForeground(10, QColor("#00ff00"))
                elif "❌" in ha_rsi_str: hisse_item.setForeground(10, QColor("#ff4444"))
                else: hisse_item.setForeground(10, QColor("#a3ccff"))
                if "✅" in sha_str: hisse_item.setForeground(11, QColor("#00ff00"))
                else: hisse_item.setForeground(11, QColor("#ff4444"))
                
                if "✅" in ao_1w_str: hisse_item.setForeground(12, QColor("#00ff00"))
                else: hisse_item.setForeground(12, QColor("#ff4444"))
                if "✅" in ao_1g_str: hisse_item.setForeground(13, QColor("#00ff00"))
                else: hisse_item.setForeground(13, QColor("#ff4444"))
                if "✅" in ao_4s_str: hisse_item.setForeground(14, QColor("#00ff00"))
                else: hisse_item.setForeground(14, QColor("#ff4444"))
                if "✅" in ao_2s_str: hisse_item.setForeground(15, QColor("#00ff00"))
                else: hisse_item.setForeground(15, QColor("#ff4444"))
                
                if "✅" in wt_str: hisse_item.setForeground(16, QColor("#FFD700"))
                else: hisse_item.setForeground(16, QColor("#ff4444"))

                hover_metni = self.dinamik_yorum_uret(hisse_kodu, ana_vade, mfi, bxt, ozel_s, ssl, st, ha_rsi, yildiz_s, sha, ao_1w_durum, ao_1g_durum, ao_4s_durum, ao_2s_durum, wt, tetik_sinyali, tetik_4s_sinyali)
                for col in range(19):
                    hisse_item.setToolTip(col, hover_metni)
                
                hisse_item.setExpanded(True)
                
                if hisse_kodu not in self.hisse_item_map: self.hisse_item_map[hisse_kodu] = []
                self.hisse_item_map[hisse_kodu].append(hisse_item)

            tarih_kok.setExpanded(True)

        self.agac.setSortingEnabled(True)

    def filtrele(self, secili_tarih):
        if secili_tarih: self.agaci_doldur(secili_tarih)

    def tarama_baslat(self):
        self.btn_tara.setEnabled(False)
        self.btn_tara.setText("⏳ TARAMA YAPILIYOR...")
        self.log_ekrani.clear()
        
        self.tarayici = TaramaMotoru()
        self.tarayici.log_sinyali.connect(self.log_yaz)
        self.tarayici.tarama_bitti.connect(self.tarama_bitti_islem)
        self.tarayici.start()

    def log_yaz(self, metin):
        self.log_ekrani.append(metin)
        self.log_ekrani.verticalScrollBar().setValue(self.log_ekrani.verticalScrollBar().maximum())

    def tarama_bitti_islem(self):
        self.btn_tara.setEnabled(True)
        self.btn_tara.setText("📋 PANODAN YENİ TARAMA BAŞLAT")
        self.verileri_yukle() 

    def fiyatlari_baslat(self):
        if not hasattr(self, 'hisse_item_map') or not self.hisse_item_map: return
        self.btn_fiyat.setEnabled(False)
        self.btn_fiyat.setText("⏳ Fiyatlar Çekiliyor...")
        
        self.fiyatci = FiyatGuncelleyici(list(self.hisse_item_map.keys()))
        self.fiyatci.fiyat_geldi.connect(self.fiyati_yaz)
        self.fiyatci.bitti.connect(self.fiyat_cekimi_bitti)
        self.fiyatci.start()

    def fiyati_yaz(self, hisse, fiyat, degisim):
        if pd.isna(fiyat) or pd.isna(degisim): return 
        
        if hisse in self.hisse_item_map:
            for item in self.hisse_item_map[hisse]:
                item.setText(17, f"{fiyat:.2f} ₺") 
                isaret = "+" if degisim > 0 else ""
                item.setText(18, f"{isaret}{degisim:.2f}%") 
                
                if degisim > 0:
                    item.setForeground(17, QColor("#00ff00"))
                    item.setForeground(18, QColor("#00ff00"))
                elif degisim < 0:
                    item.setForeground(17, QColor("#ff4444"))
                    item.setForeground(18, QColor("#ff4444"))

    def fiyat_cekimi_bitti(self):
        self.btn_fiyat.setEnabled(True)
        self.btn_fiyat.setText("🔄 Canlı Fiyatları Çek")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pencere = MasterRadarArayuz()
    pencere.show()
    sys.exit(app.exec_())
