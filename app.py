import streamlit as st
import obspy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import glob
import os
import folium
from streamlit_folium import st_folium
import numpy as np
from folium.plugins import HeatMap

# ==========================================
# 0. 페이지 및 세션 상태 초기화
# ==========================================
st.set_page_config(layout="wide", page_title="지진 분석 및 위치결정 실습", initial_sidebar_state="expanded")

if 'picks' not in st.session_state:
    st.session_state.picks = {}

# ==========================================
# 1. 데이터 로딩 및 수학 계산 엔진
# ==========================================
@st.cache_data
def load_station_info():
    try:
        df = pd.read_csv('station.txt', sep='\s+', header=0)
        df.columns = ['network', 'station_name', 'latitude', 'longitude', 'elevation']
        return df.set_index('station_name')
    except Exception as e:
        return None

@st.cache_data
def get_station_components():
    files = glob.glob(os.path.join('Waveform', '*.sac')) + glob.glob(os.path.join('Waveform', '*.SAC'))
    sta_dict = {}
    for f in files:
        try:
            st_head = obspy.read(f, headonly=True)
            sta = st_head[0].stats.station.upper()
            comp = st_head[0].stats.channel.upper()[-1]
            if sta not in sta_dict: sta_dict[sta] = {}
            sta_dict[sta][comp] = f
        except: continue
    return sta_dict

# ⭐️ 연구원님 맞춤형 전처리 (경주 지진 시간 자르기 적용)
@st.cache_data
def process_trace(file_path):
    eventtime = obspy.UTCDateTime("2016-09-12T11:33:00")
    starttime = eventtime - 50
    endtime = eventtime + 500
    
    tr = obspy.read(file_path)[0]
    tr.trim(starttime, endtime)
    tr.detrend('demean')
    tr.taper(0.05)
    # tr.filter('bandpass', freqmin=1.0, freqmax=10.0)
    return tr

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calculate_epicenter_grid(picks, station_db):
    sta_lats, sta_lons, obs_dists = [], [], []
    for sta, data in picks.items():
        if sta in station_db.index:
            sta_lats.append(station_db.loc[sta, 'latitude'])
            sta_lons.append(station_db.loc[sta, 'longitude'])
            obs_dists.append(data['dist'])
            
    if not sta_lats: return None, None

    margin = 2.0
    res = 50 
    grid_lats = np.linspace(min(sta_lats) - margin, max(sta_lats) + margin, res)
    grid_lons = np.linspace(min(sta_lons) - margin, max(sta_lons) + margin, res)
    
    g_lats, g_lons = np.meshgrid(grid_lats, grid_lons)
    g_lats, g_lons = g_lats.flatten(), g_lons.flatten()
    
    errors = np.zeros_like(g_lats)
    for lat, lon, d_obs in zip(sta_lats, sta_lons, obs_dists):
        dLat = np.radians(g_lats - lat)
        dLon = np.radians(g_lons - lon)
        a = np.sin(dLat / 2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(g_lats)) * np.sin(dLon / 2)**2
        d_calc = 6371.0 * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
        errors += (d_calc - d_obs)**2
    
    rmse = np.sqrt(errors / len(sta_lats))
    min_idx = np.argmin(rmse)
    best_lat, best_lon, best_rmse = g_lats[min_idx], g_lons[min_idx], rmse[min_idx]

    heatmap_data = []
    max_err = np.max(rmse)
    for i in range(len(g_lats)):
        weight = (1.0 - (rmse[i] / max_err))**5 
        if weight > 0.3:
            heatmap_data.append([g_lats[i], g_lons[i], weight])

    return (best_lat, best_lon, best_rmse), heatmap_data

station_db = load_station_info()
sac_inventory = get_station_components()

# ==========================================
# 2. 사이드바 네비게이션
# ==========================================
st.sidebar.title("🌋 실습 단계")
app_mode = st.sidebar.radio("바로가기", ["1. 3성분 파형 피킹", "2. 위치 결정 완료"])
st.sidebar.markdown("---")

# ==============================================================================
# [모드 1] 3성분 파형 피킹 화면
# ==============================================================================
if app_mode == "1. 3성분 파형 피킹":
    if not sac_inventory: st.stop()
    st.sidebar.subheader("📡 관측소")
    selected_sta = st.sidebar.radio("선택", sorted(list(sac_inventory.keys())))
    st.title(f"🔍 {selected_sta} 3성분 피킹 및 저장")
    
    saved_p = st.session_state.picks.get(selected_sta, {}).get('p', 0.0)
    saved_s = st.session_state.picks.get(selected_sta, {}).get('s', 0.0)

    # 3성분 데이터 모두 로드
    comps = sac_inventory[selected_sta]
    traces = {}
    max_time = 0
    for comp in ['Z', 'N', 'E']:
        if comp in comps:
            tr = process_trace(comps[comp])
            traces[comp] = tr
            max_time = max(max_time, float(tr.times()[-1]))

    st.markdown("### ⏱️ 시간 입력 (+/- 버튼으로 미세조정)")
    col1, col2 = st.columns(2)
    with col1: p_pick = st.number_input("🔴 P파 (Z성분 위주)", min_value=0.0, max_value=max_time, value=saved_p, step=0.01, format="%.2f")
    with col2: s_pick = st.number_input("🔵 S파 (N/E성분 위주)", min_value=0.0, max_value=max_time, value=saved_s, step=0.01, format="%.2f")

    if st.button(f"💾 {selected_sta} 결과 저장", use_container_width=True):
        if s_pick > p_pick:
            dist = (s_pick - p_pick) * 7.5 # k-factor
            st.session_state.picks[selected_sta] = {'p': p_pick, 's': s_pick, 'dist': dist}
            st.success(f"✅ 저장됨! (S-P: {s_pick-p_pick:.2f}s, 거리: {dist:.1f}km)")
        else: st.error("S파 도착 시간이 P파보다 빠를 수 없습니다.")

    # ⭐️ 3성분 동기화 그래프 (make_subplots 복구)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("<b>Z 성분</b> (상하)", "<b>N 성분</b> (남북)", "<b>E 성분</b> (동서)"))

    colors = {'Z': 'black', 'N': '#3366cc', 'E': '#dc3912'}
    row_idx = 1
    for comp in ['Z', 'N', 'E']:
        if comp in traces:
            tr = traces[comp]
            fig.add_trace(go.Scatter(x=tr.times(), y=tr.data, mode='lines', line=dict(color=colors[comp], width=1)), row=row_idx, col=1)
        row_idx += 1

    # 피킹 수직선 (세 그래프 관통)
    if p_pick > 0: fig.add_vline(x=p_pick, line_width=2, line_color="red", row="all", col=1)
    if s_pick > 0: fig.add_vline(x=s_pick, line_width=2, line_color="blue", row="all", col=1)

    # ⭐️ 줌 고정 및 Y축 고정 마법
    fig.update_layout(
        height=650, 
        margin=dict(l=10, r=10, t=40, b=30), 
        dragmode="zoom", 
        hovermode="x unified", 
        showlegend=False,
        uirevision="constant" # 줌 풀림 방지
    )
    fig.update_yaxes(fixedrange=True) # 위아래 확대 방지
    fig.update_xaxes(title_text="시간 (초)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True, 'doubleClick': 'reset'})

# ==============================================================================
# [모드 2] 진앙 위치 결정 화면
# ==============================================================================
elif app_mode == "2. 위치 결정 완료":
    st.title("📍 수학적 진앙 결정 및 오차 분석")
    
    if len(st.session_state.picks) < 3:
        st.warning("⚠️ 최소 3개 이상 관측소의 피킹 데이터가 필요합니다.")
        st.stop()

    with st.spinner("최적의 진앙 위치를 계산 중입니다..."):
        best_sol, heatmap_data = calculate_epicenter_grid(st.session_state.picks, station_db)
    
    if best_sol is None:
        st.error("위치 계산에 실패했습니다.")
        st.stop()
        
    calc_lat, calc_lon, calc_rmse = best_sol

    st.subheader("📊 계산 결과 리포트")
    c1, c2, c3 = st.columns(3)
    c1.metric("💡 계산된 위도", f"{calc_lat:.4f}°N")
    c2.metric("💡 계산된 경도", f"{calc_lon:.4f}°E")
    c3.metric("📉 평균 위치 오차 (RMSE)", f"{calc_rmse:.1f} km")
    st.markdown("---")

    st.subheader("🗺️ 진앙 분석 지도 (확률 분포 및 오차 비교)")
    first_sta = list(st.session_state.picks.keys())[0]
    m = folium.Map(location=[station_db.loc[first_sta, 'latitude'], station_db.loc[first_sta, 'longitude']], zoom_start=8, tiles='CartoDB positron')

    # if heatmap_data:
    #     HeatMap(heatmap_data, radius=20, blur=15, min_opacity=0.2, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)

    folium.Marker([calc_lat, calc_lon], popup=f"계산된 진앙", icon=folium.Icon(color="blue", icon="star")).add_to(m)

    for sta, data in st.session_state.picks.items():
        lat, lon = station_db.loc[sta, 'latitude'], station_db.loc[sta, 'longitude']
        folium.CircleMarker([lat, lon], radius=3, color="black", fill=True).add_to(m)
        folium.Circle(location=[lat, lon], radius=data['dist'] * 1000, color="#3366cc", weight=1, fill=False, opacity=0.3).add_to(m)

    with st.expander("🎯 실제 지진 위치와 비교하기", expanded=True):
        col_t1, col_t2 = st.columns(2)
        true_lat = col_t1.number_input("실제 위도", value=35.77, format="%.4f")
        true_lon = col_t2.number_input("실제 경도", value=129.18, format="%.4f")
        true_dist = haversine_km(calc_lat, calc_lon, true_lat, true_lon)
        st.info(f"📍 계산된 진앙은 실제 위치로부터 약 **{true_dist:.1f} km** 떨어져 있습니다.")
        folium.Marker([true_lat, true_lon], popup="실제 위치", icon=folium.Icon(color="red", icon="star")).add_to(m)

    st_folium(m, width=1000, height=600)
