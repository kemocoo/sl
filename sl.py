import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import easyocr
from PIL import Image
import re
import datetime
import warnings

# Uyarıları gizle
warnings.filterwarnings('ignore')

# Sayfa Ayarları
st.set_page_config(page_title="Birleşik Radar Web v20 (Multi-Timeframe AO)", layout="wide", page_icon="🌟")

# ==========================================
# --- MATEMATİK VE İNDİKATÖR SİSTEMLERİ ---
# ==========================================

def tetik_formasyonu_bul(data):
    if data.empty or len(data) < 5:
        return "➖"

    o, c, h, l = data['Open'].iloc[-1], data['Close'].iloc[-1], data['High'].iloc[-1], data['Low'].iloc[-1]
    o_prev, c_prev = data['Open'].iloc[-2], data['Close'].iloc[-2]

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

    if kusursuz_ha and (yutan_boga or cekic): return "GÜÇLÜ TETİK 🔥"
    elif yutan_boga: return "YUTAN BOĞA 🚀"
    elif cekic: return "ÇEKİÇ 🔨"
    elif kusursuz_ha: return "HA KOPUŞ 🟩"
    else: return "⏳ BEKLE"

def ana_sistem_hesapla(data):
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    if data.empty or len(data) < 36: return None, None, False, False, False, None, False, False, 0

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
        
    bxt_guncel, bxt_onceki, bxt_2_onceki = float(bxt_series.iloc[-1]), float(bxt_series.iloc[-2]), float(bxt_series.iloc[-3])
    bxt_mavi = bxt_guncel > bxt_onceki
    
    mavi_mum_sayaci = sum(1 for i in range(len(bxt_series)-1, 0, -1) if float(bxt_series.iloc[i]) > float(bxt_series.iloc[i-1]))

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

def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)

def self_get_rsi(series, period=14): return get_rsi(series, period)

def ozel_ve_yildiz_hesapla(data):
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    if data.empty or len(data) < 95: return "-", "-", 0.0, "BEKLE", "-", "-", "-", "BEKLE"

    close, high, low, open_p = data['Close'], data['High'], data['Low'], data['Open']

    # SSL 
    sma_high, sma_low = high.rolling(10).mean(), low.rolling(10).mean()
    hlv = pd.Series(0, index=data.index)
    hlv[close > sma_high] = 1
    hlv[close < sma_low] = -1
    hlv = hlv.replace(0, np.nan).ffill()
    data['SSL_Up'] = np.where(hlv < 0, sma_low, sma_high)
    data['SSL_Down'] = np.where(hlv < 0, sma_high, sma_low)

    # SuperTrend
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_st = tr.rolling(10).mean()
    hl2 = (high + low) / 2
    final_ub, final_lb = hl2 + (2.5 * atr_st), hl2 - (2.5 * atr_st)
    st_dir = pd.Series(1, index=data.index)
    for i in range(1, len(data)):
        if close.iloc[i-1] <= final_ub.iloc[i-1]: final_ub.iloc[i] = min(final_ub.iloc[i], final_ub.iloc[i-1])
        if close.iloc[i-1] >= final_lb.iloc[i-1]: final_lb.iloc[i] = max(final_lb.iloc[i], final_lb.iloc[i-1])
        if close.iloc[i] > final_ub.iloc[i-1]: st_dir.iloc[i] = 1
        elif close.iloc[i] < final_lb.iloc[i-1]: st_dir.iloc[i] = -1
        else: st_dir.iloc[i] = st_dir.iloc[i-1]

    # HA-RSI
    ha_c = (self_get_rsi(open_p) + self_get_rsi(high) + self_get_rsi(low) + self_get_rsi(close)) / 4
    
    ssl_kesisim_oldu_mu = any((data['SSL_Up'].iloc[-i] > data['SSL_Down'].iloc[-i]) and (data['SSL_Up'].iloc[-i-1] <= data['SSL_Down'].iloc[-i-1]) and ha_c.iloc[-i] < 35 for i in range(1, min(6, len(data)-1)))
    
    ozel_strateji_al = ssl_kesisim_oldu_mu and st_dir.iloc[-1] == 1
    ssl_durum = "YUKARI" if data['SSL_Up'].iloc[-1] > data['SSL_Down'].iloc[-1] else "AŞAĞI"
    st_durum = "BUY" if st_dir.iloc[-1] == 1 else "SELL"
    
    # SHA, Hull ve WT hesaplamaları (Basitleştirilmiş dönüşler)
    ema_o_sha = open_p.ewm(span=7, adjust=False).mean()
    ema_c_sha = close.ewm(span=7, adjust=False).mean()
    sha_durum = "YEŞİL" if ema_c_sha.iloc[-1] > ema_o_sha.iloc[-1] else "KIRMIZI"
    
    # WaveTrend
    ap_wt = (high + low + close) / 3.0
    esa_wt = ap_wt.ewm(span=10, adjust=False).mean()
    d_wt = (ap_wt - esa_wt).abs().ewm(span=10, adjust=False).mean()
    ci_wt = (ap_wt - esa_wt) / (0.015 * d_wt.replace(0, np.nan))
    wt1 = ci_wt.fillna(0).ewm(span=18, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    
    wt1_g = float(wt1.iloc[-1])
    wt_kesisim_yakin = any((wt1.iloc[-i] > wt2.iloc[-i]) and (wt1.iloc[-i-1] <= wt2.iloc[-i-1]) and (wt1.iloc[-i] < -40) for i in range(1, min(6, len(wt1)-1)))
    wt_durum = f"{wt1_g:.1f}" + (" (Dip)" if wt_kesisim_yakin else "")
    
    yildiz_sinyal = "AL" if sha_durum == "YEŞİL" and wt_kesisim_yakin else "BEKLE"

    return ssl_durum, st_durum, float(ha_c.iloc[-1]), ("AL" if ozel_strateji_al else "BEKLE"), sha_durum, "YEŞİL", wt_durum, yildiz_sinyal

def periyot_verilerini_cek(hisse_kodu):
    # 1 Haftalık (1W) Veri
    data_1w = yf.download(f"{hisse_kodu}.IS", period="2y", interval="1wk", progress=False)
    if data_1w.empty: data_1w = yf.download(hisse_kodu, period="2y", interval="1wk", progress=False)
    if not data_1w.empty and isinstance(data_1w.columns, pd.MultiIndex): data_1w.columns = data_1w.columns.get_level_values(0)
    try: a_1w = ana_sistem_hesapla(data_1w)
    except: a_1w = (None, None, False, False, False, None, False, False, 0)

    # 1 Günlük (1D) Veri
    data_1g = yf.download(f"{hisse_kodu}.IS", period="1y", interval="1d", progress=False)
    if data_1g.empty: data_1g = yf.download(hisse_kodu, period="1y", interval="1d", progress=False)
    if not data_1g.empty and isinstance(data_1g.columns, pd.MultiIndex): data_1g.columns = data_1g.columns.get_level_values(0)
    
    try: a_1g = ana_sistem_hesapla(data_1g)
    except: a_1g = (None, None, False, False, False, None, False, False, 0)
    try: ozel_ve_yildiz_1g = ozel_ve_yildiz_hesapla(data_1g)
    except: ozel_ve_yildiz_1g = ("-", "-", 0.0, "BEKLE", "-", "-", "-", "BEKLE")

    # 1 Saatlik Veriden 2S ve 4S Türetme
    data_1h = yf.download(f"{hisse_kodu}.IS", period="3mo", interval="1h", progress=False)
    if not data_1h.empty:
        if isinstance(data_1h.columns, pd.MultiIndex): data_1h.columns = data_1h.columns.get_level_values(0)
        if data_1h.index.tz is not None: data_1h.index = data_1h.index.tz_localize(None)
        
        data_2s = data_1h[['Open', 'High', 'Low', 'Close', 'Volume']].resample('2h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        a_2s = ana_sistem_hesapla(data_2s)
        tetik_2s = tetik_formasyonu_bul(data_2s)
        
        data_4s = data_1h[['Open', 'High', 'Low', 'Close', 'Volume']].resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        a_4s = ana_sistem_hesapla(data_4s)
        tetik_4s = tetik_formasyonu_bul(data_4s)
    else:
        a_2s = (None, None, False, False, False, None, False, False, 0)
        a_4s = (None, None, False, False, False, None, False, False, 0)
        tetik_2s = "BEKLE"
        tetik_4s = "BEKLE"

    return a_1w, a_1g, a_2s, a_4s, ozel_ve_yildiz_1g, tetik_2s, tetik_4s


# ==========================================
# --- STREAMLIT ARAYÜZÜ (WEB / MOBİL) ---
# ==========================================

st.title("🌟 Birleşik Radar & Manuel Karar Terminali (Web v20 - Multi-Timeframe AO)")
st.markdown("Hisseleri manuel yazabilir veya ekran görüntüsü yükleyebilirsin.")

# Yükleme ve Girdi Alanı
col1, col2 = st.columns(2)
with col1:
    hisse_input = st.text_input("Hisse Kodlarını Yazın (Örn: THYAO, EREGL):")
with col2:
    uploaded_file = st.file_uploader("Hisse Listesi Görüntüsü Yükleyin (Ekran Görüntüsü)", type=['png', 'jpg', 'jpeg'])

temiz_hisseler = []

if uploaded_file is not None:
    with st.spinner("Görseldeki hisseler yapay zeka ile okunuyor..."):
        image = Image.open(uploaded_file)
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        okunan_metin = " ".join(reader.readtext(np.array(image.convert('RGB')), detail=0))
        ham_hisseler = re.findall(r'\b[A-Z]{2,5}\b', okunan_metin)
        yasakli = {'INC', 'LLC', 'CORP', 'LTD', 'CO', 'THE', 'AND', 'BIST', 'TCMB', 'POLDY', 'GIB', 'YATIRIM', 'FON', 'AS', 'A.S', 'USD', 'EUR', 'TRY'}
        temiz_hisseler = sorted(list(set(h for h in ham_hisseler if h not in yasakli)))
        st.success(f"Görselden {len(temiz_hisseler)} hisse bulundu!")

if hisse_input:
    elle_girilenler = [h.strip().upper() for h in hisse_input.split(',')]
    temiz_hisseler = sorted(list(set(temiz_hisseler + elle_girilenler)))

if st.button("🚀 Kapsamlı Taramayı Başlat", type="primary") and temiz_hisseler:
    sonuclar = []
    progress_bar = st.progress(0)
    
    for i, hisse in enumerate(temiz_hisseler):
        st.text(f"Taranıyor: {hisse}...")
        a_1w, a_1g, a_2s, a_4s, ozel_1g, t_2s, t_4s = periyot_verilerini_cek(hisse)
        
        # Sonuçları Çıkart
        ao_1w, _, _, ao_k_1w, _, _, _, _, _ = a_1w
        ao_1g, bxt_1g, mavi_1g, ao_k_1g, kesin_al_1g, mfi_1g, _, k_sat_1g, mavi_mum_1g = a_1g
        ao_4s, _, _, ao_k_4s, _, _, _, _, _ = a_4s
        ao_2s, bxt_2s, mavi_2s, ao_k_2s, kesin_al_2s, mfi_2s, _, k_sat_2s, mavi_mum_2s = a_2s
        ssl_1g, st_1g, ha_rsi_1g, ozel_s, sha, hull, wt, yildiz_s = ozel_1g
        
        # Karar Mekanizması
        skor = 1
        if mavi_1g:
            if 0 < mavi_mum_1g <= 6: skor += 2
            elif mavi_mum_1g > 6: skor += 1

        if mavi_2s: skor += 1
        if mfi_2s is not None and mfi_1g is not None:
            if mfi_2s < 80 and mfi_1g < 80: skor += 1
            elif mfi_2s > 80 or mfi_1g > 80: skor -= 1
        
        skor = max(1, min(5, skor))
        
        ozel_yildiz = mavi_2s and (ao_2s is not None and ao_2s > 0) and (bxt_2s is not None and bxt_2s > 0)
        
        if k_sat_1g or k_sat_2s: vade = "ÇIKIŞ VAKTİ 🛑"
        elif ozel_yildiz: vade = f"🌟 YILDIZ ({mavi_mum_2s}. Mum)"
        elif skor >= 4 and mavi_1g: vade = "1-3 Hafta (Ana Trend)"
        elif skor >= 3 and kesin_al_2s: vade = "1-3 Gün (Vur-Kaç)"
        elif mavi_2s: vade = "1-2 Gün (Tepki)"
        elif skor <= 2: vade = "İzleme Modu"
        else: vade = "Belirsiz"
        
        # AO Durumları
        ao_1w_str = f"{ao_1w:.2f} KIRMIZI" if ao_1w is not None and ao_k_1w else (f"{ao_1w:.2f} YEŞİL" if ao_1w is not None else "-")
        ao_1g_str = f"{ao_1g:.2f} KIRMIZI" if ao_1g is not None and ao_k_1g else (f"{ao_1g:.2f} YEŞİL" if ao_1g is not None else "-")
        ao_4s_str = f"{ao_4s:.2f} KIRMIZI" if ao_4s is not None and ao_k_4s else (f"{ao_4s:.2f} YEŞİL" if ao_4s is not None else "-")
        ao_2s_str = f"{ao_2s:.2f} KIRMIZI" if ao_2s is not None and ao_k_2s else (f"{ao_2s:.2f} YEŞİL" if ao_2s is not None else "-")

        # Canlı Fiyat
        try:
            ticker = yf.Ticker(f"{hisse}.IS")
            canli_fiyat = float(ticker.history(period="1d")['Close'].iloc[-1])
        except:
            canli_fiyat = 0.0

        sonuclar.append({
            "Hisse": hisse,
            "Fiyat (₺)": round(canli_fiyat, 2),
            "Ana Sistem": vade,
            "Özel Sinyal": ozel_s,
            "Yıldız": yildiz_s,
            "2S Tetik": t_2s,
            "4S Tetik": t_4s,
            "MFI": round(mfi_1g, 1) if mfi_1g else "-",
            "BXT": round(bxt_1g, 2) if bxt_1g else "-",
            "AO (1 Hft)": ao_1w_str,
            "AO (1 Gün)": ao_1g_str,
            "AO (4 Saat)": ao_4s_str,
            "AO (2 Saat)": ao_2s_str,
            "WaveTrend": wt
        })
        progress_bar.progress((i + 1) / len(temiz_hisseler))
        
    st.success("Tarama Tamamlandı!")
    
    # DataFrame oluştur ve Göster 
    df = pd.DataFrame(sonuclar)
    
    def renk_ayarla(val):
        val_str = str(val)
        if "AL" in val_str or "YILDIZ" in val_str or "YEŞİL" in val_str:
            return "background-color: #1a4222"
        elif "ÇIKIŞ" in val_str or "KIRMIZI" in val_str:
            return "background-color: #4a1f1f"
        return ""

    stil_tablosu = df.style.map(renk_ayarla, subset=["Ana Sistem", "Özel Sinyal", "Yıldız", "AO (1 Hft)", "AO (1 Gün)", "AO (4 Saat)", "AO (2 Saat)"])
    st.dataframe(stil_tablosu, use_container_width=True)
