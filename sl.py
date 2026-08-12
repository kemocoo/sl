import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import easyocr
from PIL import Image
import re
import datetime
import warnings
import time

# Uyarıları gizle
warnings.filterwarnings('ignore')

# Sayfa Ayarları
st.set_page_config(page_title="Birleşik Radar Web v26 (İvme Röntgeni)", layout="wide", page_icon="🌟")

# ==========================================
# --- YAPAY ZEKA MOTORUNU HAFIZAYA ALMA ---
# ==========================================
@st.cache_resource(show_spinner=False)
def ocr_motorunu_baslat():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

# ==========================================
# --- MATEMATİK VE İNDİKATÖR SİSTEMLERİ ---
# ==========================================

def msl_squeeze_hesapla(data):
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    if len(data) < 22:
        return 0.0, 0.0, "EXPANDED"
        
    chanLen = 20
    priceMid = (data['High'].rolling(chanLen).max() + data['Low'].rolling(chanLen).min()) / 2
    oscBaseline = (priceMid + data['Close'].rolling(chanLen).mean()) / 2
    val = data['Close'] - oscBaseline
    
    def linreg_end(y):
        try:
            x = np.arange(len(y))
            m, c = np.polyfit(x, y, 1)
            return m * x[-1] + c
        except:
            return 0.0
        
    pulseVal = val.rolling(window=chanLen).apply(linreg_end, raw=True)
    
    bandMid = data['Close'].rolling(20).mean()
    bandHalf = 2.0 * data['Close'].rolling(20).std()
    bandTop = bandMid + bandHalf
    bandFloor = bandMid - bandHalf
    
    tr1 = data['High'] - data['Low']
    tr2 = (data['High'] - data['Close'].shift(1)).abs()
    tr3 = (data['Low'] - data['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    chanMid = data['Close'].rolling(20).mean()
    chanWidth = tr.rolling(20).mean() * 1.5
    chanTop = chanMid + chanWidth
    chanFloor = chanMid - chanWidth
    
    coiled = (bandFloor > chanFloor) & (bandTop < chanTop)
    
    p_guncel = float(pulseVal.iloc[-1]) if not pd.isna(pulseVal.iloc[-1]) else 0.0
    p_onceki = float(pulseVal.iloc[-2]) if not pd.isna(pulseVal.iloc[-2]) else 0.0
    c_durum = "COILED" if coiled.iloc[-1] else "EXPANDED"
    
    return p_guncel, p_onceki, c_durum

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
    if data.empty or len(data) < 36:
        return None, None, False, False, False, None, False, False, 0, 0.0, "➖"

    high, low = data['High'].squeeze(), data['Low'].squeeze()
    mid_price = (high + low) / 2
    sma_5, sma_35 = mid_price.rolling(window=5).mean(), mid_price.rolling(window=35).mean()
    
    ao_series = (sma_5 - sma_35).dropna()
    if len(ao_series) < 3: return None, None, False, False, False, None, False, False, 0, 0.0, "➖"
        
    ao_guncel_deger = float(ao_series.iloc[-1])
    ao_onceki_deger = float(ao_series.iloc[-2])
    ao_2_onceki_deger = float(ao_series.iloc[-3])
    ao_kirmizi = ao_guncel_deger < ao_onceki_deger

    guncel_yesil = ao_guncel_deger > ao_onceki_deger
    onceki_yesil = ao_onceki_deger > ao_2_onceki_deger
    
    if not onceki_yesil and guncel_yesil: ao_donus = "🚀 AL (Dip)"
    elif onceki_yesil and not guncel_yesil: ao_donus = "🛑 SAT (Tepe)"
    elif guncel_yesil: ao_donus = "🟩 İvme +"
    else: ao_donus = "🟥 İvme -"

    ha_close = (data['Open'].squeeze() + data['High'].squeeze() + data['Low'].squeeze() + data['Close'].squeeze()) / 4
    ema_5, ema_20 = ha_close.ewm(span=5, adjust=False).mean(), ha_close.ewm(span=20, adjust=False).mean()
    
    bxt_series = (ema_5 - ema_20).dropna()
    if len(bxt_series) < 3: return None, None, False, False, False, None, False, False, 0, 0.0, "➖"
        
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

    return ao_guncel_deger, bxt_guncel, bxt_mavi, ao_kirmizi, kesin_al_sinyali, mfi_guncel, yeni_mavi_sinyali, kesin_sat_sinyali, mavi_mum_sayaci, ao_onceki_deger, ao_donus

def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)

def ozel_ve_yildiz_hesapla(data):
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    if data.empty or len(data) < 95: return "-", "-", 0.0, "BEKLE", "-", "-", "-", "BEKLE"

    close, high, low, open_p = data['Close'], data['High'], data['Low'], data['Open']

    sma_high, sma_low = high.rolling(10).mean(), low.rolling(10).mean()
    hlv = pd.Series(0, index=data.index)
    hlv[close > sma_high] = 1
    hlv[close < sma_low] = -1
    hlv = hlv.replace(0, np.nan).ffill()
    data['SSL_Up'] = np.where(hlv < 0, sma_low, sma_high)
    data['SSL_Down'] = np.where(hlv < 0, sma_high, sma_low)

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

    ha_c = (get_rsi(open_p) + get_rsi(high) + get_rsi(low) + get_rsi(close)) / 4
    
    ssl_kesisim_oldu_mu = any((data['SSL_Up'].iloc[-i] > data['SSL_Down'].iloc[-i]) and (data['SSL_Up'].iloc[-i-1] <= data['SSL_Down'].iloc[-i-1]) and ha_c.iloc[-i] < 35 for i in range(1, min(6, len(data)-1)))
    
    ozel_strateji_al = ssl_kesisim_oldu_mu and st_dir.iloc[-1] == 1
    ssl_durum = "YUKARI" if data['SSL_Up'].iloc[-1] > data['SSL_Down'].iloc[-1] else "AŞAĞI"
    st_durum = "BUY" if st_dir.iloc[-1] == 1 else "SELL"
    
    ema_o_sha = open_p.ewm(span=7, adjust=False).mean()
    ema_c_sha = close.ewm(span=7, adjust=False).mean()
    sha_durum = "YEŞİL" if ema_c_sha.iloc[-1] > ema_o_sha.iloc[-1] else "KIRMIZI"
    
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

def guvenli_veri_indir(hisse_kodu, period, interval):
    for deneme in range(3):
        try:
            data = yf.download(f"{hisse_kodu}.IS", period=period, interval=interval, progress=False)
            if data.empty: data = yf.download(hisse_kodu, period=period, interval=interval, progress=False)
            if not data.empty: return data
        except:
            time.sleep(1)
    return pd.DataFrame()

def periyot_verilerini_cek(hisse_kodu):
    # 1 Haftalık
    data_1w = guvenli_veri_indir(hisse_kodu, period="2y", interval="1wk")
    if not data_1w.empty and isinstance(data_1w.columns, pd.MultiIndex): data_1w.columns = data_1w.columns.get_level_values(0)
    try: a_1w = ana_sistem_hesapla(data_1w)
    except: a_1w = (None, None, False, False, False, None, False, False, 0, 0.0, "➖")
    try: msl_1w = msl_squeeze_hesapla(data_1w)
    except: msl_1w = (0.0, 0.0, "EXPANDED")

    # 1 Günlük
    data_1g = guvenli_veri_indir(hisse_kodu, period="1y", interval="1d")
    if not data_1g.empty and isinstance(data_1g.columns, pd.MultiIndex): data_1g.columns = data_1g.columns.get_level_values(0)
    try: a_1g = ana_sistem_hesapla(data_1g)
    except: a_1g = (None, None, False, False, False, None, False, False, 0, 0.0, "➖")
    try: ozel_ve_yildiz_1g = ozel_ve_yildiz_hesapla(data_1g)
    except: ozel_ve_yildiz_1g = ("-", "-", 0.0, "BEKLE", "-", "-", "-", "BEKLE")
    try: msl_1g = msl_squeeze_hesapla(data_1g)
    except: msl_1g = (0.0, 0.0, "EXPANDED")

    # 1 Saatlik Veriden 2S ve 4S Türetme
    data_1h = guvenli_veri_indir(hisse_kodu, period="3mo", interval="1h")
    if not data_1h.empty:
        if isinstance(data_1h.columns, pd.MultiIndex): data_1h.columns = data_1h.columns.get_level_values(0)
        if data_1h.index.tz is not None: data_1h.index = data_1h.index.tz_localize(None)
        
        data_2s = data_1h[['Open', 'High', 'Low', 'Close', 'Volume']].resample('2h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        try: a_2s = ana_sistem_hesapla(data_2s)
        except: a_2s = (None, None, False, False, False, None, False, False, 0, 0.0, "➖")
        try: tetik_2s = tetik_formasyonu_bul(data_2s)
        except: tetik_2s = "BEKLE"
        try: msl_2s = msl_squeeze_hesapla(data_2s)
        except: msl_2s = (0.0, 0.0, "EXPANDED")
        
        data_4s = data_1h[['Open', 'High', 'Low', 'Close', 'Volume']].resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        try: a_4s = ana_sistem_hesapla(data_4s)
        except: a_4s = (None, None, False, False, False, None, False, False, 0, 0.0, "➖")
        try: tetik_4s = tetik_formasyonu_bul(data_4s)
        except: tetik_4s = "BEKLE"
        try: msl_4s = msl_squeeze_hesapla(data_4s)
        except: msl_4s = (0.0, 0.0, "EXPANDED")
        
    else:
        a_2s = (None, None, False, False, False, None, False, False, 0, 0.0, "➖")
        a_4s = (None, None, False, False, False, None, False, False, 0, 0.0, "➖")
        tetik_2s = "BEKLE"
        tetik_4s = "BEKLE"
        msl_2s = (0.0, 0.0, "EXPANDED")
        msl_4s = (0.0, 0.0, "EXPANDED")

    return a_1w, a_1g, a_2s, a_4s, ozel_ve_yildiz_1g, tetik_2s, tetik_4s, msl_1w, msl_1g, msl_4s, msl_2s


# ==========================================
# --- STREAMLIT ARAYÜZÜ (WEB / MOBİL) ---
# ==========================================

st.title("🌟 Birleşik Radar & Terminal (Web v26 - İvme Röntgeni)")
st.markdown("Hızlı optizimasyon: Fiyat modülü kaldırıldı, tüm sinyaller ve MTF (Çoklu Zaman Dilimi) göstergeleri doğrudan hücrelere işlendi.")

# Yükleme ve Girdi Alanı
col1, col2 = st.columns(2)
with col1:
    hisse_input = st.text_input("Hisse Kodlarını Yazın (Örn: THYAO, EREGL):")
with col2:
    uploaded_file = st.file_uploader("Hisse Listesi Görüntüsü Yükleyin (Ekran Görüntüsü)", type=['png', 'jpg', 'jpeg'])

temiz_hisseler = []

if uploaded_file is not None:
    with st.spinner("Görseldeki hisseler yapay zeka ile okunuyor... (Bu işlem sadece ilk seferde uzun sürer)"):
        reader = ocr_motorunu_baslat()
        image = Image.open(uploaded_file)
        okunan_metin = " ".join(reader.readtext(np.array(image.convert('RGB')), detail=0))
        ham_hisseler = re.findall(r'\b[A-Z]{2,5}\b', okunan_metin)
        yasakli = {'INC', 'LLC', 'CORP', 'LTD', 'CO', 'THE', 'AND', 'BIST', 'TCMB', 'POLDY', 'GIB', 'YATIRIM', 'FON', 'AS', 'A.S', 'USD', 'EUR', 'TRY'}
        temiz_hisseler = sorted(list(set(h for h in ham_hisseler if h not in yasakli)))
        st.success(f"Görselden {len(temiz_hisseler)} hisse bulundu!")

if hisse_input:
    elle_girilenler = [h.strip().upper() for h in hisse_input.split(',')]
    temiz_hisseler = sorted(list(set(temiz_hisseler + elle_girilenler)))

if st.button("🚀 Kapsamlı Röntgen Taramasını Başlat", type="primary") and temiz_hisseler:
    sonuclar = []
    progress_bar = st.progress(0)
    
    for i, hisse in enumerate(temiz_hisseler):
        st.text(f"İvme Röntgeni Çekiliyor: {hisse}...")
        a_1w, a_1g, a_2s, a_4s, ozel_1g, t_2s, t_4s, msl_1w, msl_1g, msl_4s, msl_2s = periyot_verilerini_cek(hisse)
        
        # Sonuçları Çıkart
        ao_1w, _, _, _, _, _, _, _, _, _, ao_donus_1w = a_1w
        ao_1g, bxt_1g, mavi_1g, _, kesin_al_1g, mfi_1g, _, k_sat_1g, mavi_mum_1g, ao_onceki_1g, ao_donus_1g = a_1g
        ao_4s, _, _, _, _, _, _, _, _, _, ao_donus_4s = a_4s
        ao_2s, bxt_2s, mavi_2s, _, kesin_al_2s, mfi_2s, _, k_sat_2s, mavi_mum_2s, _, ao_donus_2s = a_2s
        ssl_1g, st_1g, ha_rsi_1g, ozel_s, sha, hull, wt, yildiz_s = ozel_1g
        
        # MSL Pulse Mantığı
        def get_msl_str(msl_data):
            p, p_prev, _ = msl_data
            if p > 0 and p > p_prev: s = "🟩 İvme +"
            elif p > 0 and p <= p_prev: s = "🛑 SAT (Tepe)"
            elif p < 0 and p < p_prev: s = "🟥 İvme -"
            elif p < 0 and p >= p_prev: s = "🚀 AL (Dip)"
            else: s = "⏳ BEKLE"
            return f"{p:.2f} | {s}"

        msl_str_1w = get_msl_str(msl_1w)
        msl_str_1g = get_msl_str(msl_1g)
        msl_str_4s = get_msl_str(msl_4s)
        msl_str_2s = get_msl_str(msl_2s)

        msl_pulse_1g_val, _, _ = msl_1g

        # KUSURSUZ MSL & AO KESİŞİM SİNYALİ
        ao_kesisim_al = ao_1g is not None and ao_onceki_1g is not None and (ao_onceki_1g < 0) and (ao_1g > 0)
        ao_kesisim_sat = ao_1g is not None and ao_onceki_1g is not None and (ao_onceki_1g > 0) and (ao_1g < 0)
        
        msl_ao_sinyal = "BEKLE"
        if ao_kesisim_al and msl_pulse_1g_val >= 0:
            msl_ao_sinyal = "🚀 AL"
        elif ao_kesisim_sat and msl_pulse_1g_val <= 0:
            msl_ao_sinyal = "🛑 SAT"

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
        
        # AO Sütunları
        ao_str_1w = f"{ao_1w:.2f} | {ao_donus_1w}" if ao_1w is not None else "-"
        ao_str_1g = f"{ao_1g:.2f} | {ao_donus_1g}" if ao_1g is not None else "-"
        ao_str_4s = f"{ao_4s:.2f} | {ao_donus_4s}" if ao_4s is not None else "-"
        ao_str_2s = f"{ao_2s:.2f} | {ao_donus_2s}" if ao_2s is not None else "-"

        mfi_str = f"{mfi_1g:.1f} ✅" if mfi_1g is not None and mfi_1g < 80 else (f"{mfi_1g:.1f} ❌" if mfi_1g is not None else "-")
        bxt_str = f"{bxt_1g:.2f} ✅" if bxt_1g is not None and bxt_1g > 0 else (f"{bxt_1g:.2f} ❌" if bxt_1g is not None else "-")
        ssl_str = f"{ssl_1g} ✅" if ssl_1g and "YUKARI" in str(ssl_1g) else (f"{ssl_1g} ❌" if ssl_1g and "AŞAĞI" in str(ssl_1g) else "-")
        st_str = f"{st_1g} ✅" if st_1g and "BUY" in str(st_1g) else (f"{st_1g} ❌" if st_1g and "SELL" in str(st_1g) else "-")
        ha_rsi_str = f"{ha_rsi_1g:.1f} ✅ (Dip)" if ha_rsi_1g is not None and ha_rsi_1g <= 35 else (f"{ha_rsi_1g:.1f} ❌" if ha_rsi_1g is not None and ha_rsi_1g >= 70 else (f"{ha_rsi_1g:.1f} ➖" if ha_rsi_1g is not None else "-"))
        sha_str = f"{sha} ✅" if sha and "YEŞİL" in str(sha) else (f"{sha} ❌" if sha and "KIRMIZI" in str(sha) else "-")
        wt_str = f"{wt} ✅" if wt and "Dip" in str(wt) else (f"{wt} ❌" if wt else "-")

        sonuclar.append({
            "Hisse Adı": hisse,
            "[1] Ana Sistem": vade,
            "[2] Özel Strateji": f"🚀 {ozel_s}" if "AL" in str(ozel_s) else "⏳ BEKLE",
            "[3] Yıldız Sistemi": f"🌟 {yildiz_s}" if "AL" in str(yildiz_s) else "⏳ BEKLE",
            "[4] 2S Tetik": t_2s,
            "[5] 4S Tetik": t_4s,
            "[6] MSL+AO S.Kesişimi": msl_ao_sinyal,
            "MSL İvme (1W)": msl_str_1w,
            "MSL İvme (1D)": msl_str_1g,
            "MSL İvme (4H)": msl_str_4s,
            "MSL İvme (2H)": msl_str_2s,
            "AO İvme (1W)": ao_str_1w,
            "AO İvme (1D)": ao_str_1g,
            "AO İvme (4H)": ao_str_4s,
            "AO İvme (2H)": ao_str_2s,
            "MFI": mfi_str,
            "BXT": bxt_str,
            "SSL": ssl_str,
            "S.Trend": st_str,
            "HA-RSI": ha_rsi_str,
            "S.HA": sha_str,
            "WaveTrend": wt_str
        })
        progress_bar.progress((i + 1) / len(temiz_hisseler))
        
    st.success("Tarama Tamamlandı!")
    
    df = pd.DataFrame(sonuclar)
    
    # --- YENİ RENKLENDİRME MANTIĞI ---
    def renk_ayarla(val):
        val_str = str(val)
        if "AL" in val_str or "YILDIZ" in val_str or "🟩" in val_str or "✅" in val_str:
            return "background-color: #1a4222"
        elif "ÇIKIŞ" in val_str or "SAT" in val_str or "🟥" in val_str or "❌" in val_str:
            return "background-color: #4a1f1f"
        return ""

    # Tablonun tamamına renk filtresini uygula
    stil_tablosu = df.style.map(renk_ayarla)
    st.dataframe(stil_tablosu, width='stretch')
