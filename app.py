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
# 0. 페이지 및 세션 상태(저장공간) 초기화
# ==========================================
st.set_page_config(layout="wide", page_title="실전! 진앙 결정 실습", initial_sidebar_state="expanded")

if 'picks' not in st.session_state:
    st.session_state.picks = {}

# ==========================================
# 1. 데이터 로딩 및 수학 계산 함수 (핵심 로직)
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

@st.cache_data
def process_trace(file_path):
    """SAC 파일을 읽고 노이즈 제거를 위한 필터링 및 시간 자르기를 수행합니다."""
    eventtime = obspy.UTCDateTime("2016-09-12T11:33:00")
    starttime = eventtime - 50
    endtime = eventtime + 500
    
    tr = obspy.read(file_path)[0]
    tr.trim(starttime, endtime) # 지정한 시간대로 파형 자르기
    tr.detrend('demean')
    tr.taper(0.05)
    tr.filter('bandpass', freqmin=1.0, freqmax=10.0)
    
    return tr

# ⭐️ 하버사인 공식 (두 위경도 사이의 대권 거리 계산 - km)
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # 지구 반지름 (km)
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + \
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * \
        np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ⭐️ 격자 검색(Grid Search) 기반 위치 결정 엔진
def calculate_epicenter_grid(picks, station_db):
    """학생들이 피킹한 거리 정보를 바탕으로 최적의 진앙지와 오차 분포를 계산합니다."""
    sta_lats = []
    sta_lons = []
    obs_dists = []
    
    for sta, data in picks.items():
        if sta in station_db.index:
            sta_lats.append(station_db.loc[sta, 'latitude'])
            sta_lons.append(station_db.loc[sta, 'longitude'])
            obs_dists.append(data['dist'])
            
    if not sta_lats: return None, None

    # 1. 검색 영역 설정 (관측소 주변 ±2도)
    margin = 2.0
    lat_min, lat_max = min(sta_lats) - margin, max(sta_lats) + margin
    lon_min, lon_max = min(sta_lons) - margin, max(sta_lons) + margin
    
    # 2. 격자 생성 (50x50 = 2500개 지점 테스트) - 성능을 위해 적절히 설정
    res = 50 
    grid_lats = np.linspace(lat_min, lat_max, res)
    grid_lons = np.linspace(lon_min, lon_max, res)
    
    # 계산 효율을 위해 numpy 벡터화 사용
    g_lats, g_lons = np.meshgrid(grid_lats, grid_lons)
    g_lats = g_lats.flatten()
    g_lons = g_lons.flatten()
    
    # 3. 각 격자점별 오차 계산 (RMSE: Root Mean Square Error)
    # 이론적 거리 계산 (벡터화)
    errors = np.zeros_like(g_lats)
    for lat, lon, d_obs in zip(sta_lats, sta_lons, obs_dists):
        # Haversine 공식의 벡터화 구현
        R = 6371.0
        dLat = np.radians(g_lats - lat)
        dLon = np.radians(g_lons - lon)
        a = np.sin(dLat / 2) ** 2 + \
            np.cos(np.radians(lat)) * np.cos(np.radians(g_lats)) * np.sin(dLon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d_calc = R * c
        
        # (이론 거리 - 학생 피킹 거리)의 제곱을 누적
        errors += (d_calc - d_obs) ** 2
    
    # RMSE 계산
    rmse = np.sqrt(errors / len(sta_lats))
    
    # 4. 최적의 지점 (RMSE가 최소인 곳) 찾기
    min_idx = np.argmin(rmse)
    best_lat = g_lats[min_idx]
    best_lon = g_lons[min_idx]
    best_rmse = rmse[min_idx]

    # 5. 확률 분포 시각화용 데이터 준비 (Heatmap용)
    # 오차가 적을수록 확률이 높으므로 인버스 처리. Heatmap은 [lat, lon, weight] 형식
    # weight를 가공하여 겹치는 영역이 붉게 보이도록 함
    heatmap_data = []
    # 오차 최대값으로 정규화하여 가중치 계산
    max_err = np.max(rmse)
    for i in range(len(g_lats)):
        # 오차가 작을수록(0에 가까울수록) 1에 가깝고, 클수록 0에 가까운 가중치
        weight = (1.0 - (rmse[i] / max_err)) ** 5 # 겹치는 영역을 강조하기 위해 5제곱
        if weight > 0.3: # 가중치가 너무 낮은 곳은 제외하여 깔끔하게 표시
            heatmap_data.append([g_lats[i], g_lons[i], weight])

    return (best_lat, best_lon, best_rmse), heatmap_data

# 데이터 불러오기
station_db = load_station_info()
sac_inventory = get_station_components()

# ==========================================
# 2. 사이드바 네비게이션
# ==========================================
st.sidebar.title("🌋 실습 단계")
app_mode = st.sidebar.radio("바로가기", ["1. 파형 피킹", "2. 위치 결정 완료"])
st.sidebar.markdown("---")

# ==============================================================================
# [모드 1] 피킹 화면 (변경 없음, 최적화 유지)
# ==============================================================================
if app_mode == "1. 파형 피킹":
    if not sac_inventory: st.stop()
    st.sidebar.subheader("📡 관측소")
    selected_sta = st.sidebar.radio("선택", sorted(list(sac_inventory.keys())))
    st.title(f"🔍 {selected_sta} 피킹 및 저장")
    
    saved_p = st.session_state.picks.get(selected_sta, {}).get('p', 0.0)
    saved_s = st.session_state.picks.get(selected_sta, {}).get('s', 0.0)

    tr_z = process_trace(sac_inventory[selected_sta]['Z']) if 'Z' in sac_inventory[selected_sta] else None
    max_time = float(tr_z.times()[-1]) if tr_z else 100.0

    st.markdown("### ⏱️ 시간 입력")
    col1, col2 = st.columns(2)
    with col1: p_pick = st.number_input("🔴 P파 (Z)", min_value=0.0, max_value=max_time, value=saved_p, step=0.01, format="%.2f")
    with col2: s_pick = st.number_input("🔵 S파 (N/E)", min_value=0.0, max_value=max_time, value=saved_s, step=0.01, format="%.2f")

    if st.button(f"💾 {selected_sta} 결과 저장", use_container_width=True):
        if s_pick > p_pick:
            dist = (s_pick - p_pick) * 7.5
            st.session_state.picks[selected_sta] = {'p': p_pick, 's': s_pick, 'dist': dist}
            st.success(f"✅ 저장됨! 거리: {dist:.1f}km")
        else: st.error("S-P 시간을 확인하세요.")

    # 그래프 렌더링 (Z성분만 예시로, 필요시 make_subplots 사용)
    if tr_z:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr_z.times(), y=tr_z.data, mode='lines', line=dict(color='black', width=1)))
        if p_pick > 0: fig.add_vline(x=p_pick, line_width=2, line_color="red")
        if s_pick > 0: fig.add_vline(x=s_pick, line_width=2, line_color="blue")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), dragmode="zoom", uirevision="constant")
        fig.update_yaxes(fixedrange=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True, 'doubleClick': 'reset'})

# ==============================================================================
# [모드 2] 진앙 위치 결정 화면 (수학적 엔진 탑재)
# ==============================================================================
elif app_mode == "2. 위치 결정 완료":
    st.title("📍 수학적 진앙 결정 및 오차 분석")
    
    if len(st.session_state.picks) < 3:
        st.warning("⚠️ 최소 3개 이상 관측소의 피킹 데이터가 필요합니다. '1. 파형 피킹'에서 저장해 주세요.")
        st.info(f"현재 저장된 관측소: {', '.join(st.session_state.picks.keys())}")
        st.stop()

    # 1. 위치 결정 계산 실행!
    with st.spinner("최적의 진앙 위치를 계산 중입니다..."):
        best_sol, heatmap_data = calculate_epicenter_grid(st.session_state.picks, station_db)
    
    if best_sol is None:
        st.error("위치 계산에 실패했습니다. 데이터를 확인하세요.")
        st.stop()
        
    calc_lat, calc_lon, calc_rmse = best_sol

    # 2. 결과 리포트 출력
    st.subheader("📊 계산 결과 리포트")
    c1, c2, c3 = st.columns(3)
    c1.metric("💡 계산된 위도", f"{calc_lat:.4f}°N")
    c2.metric("💡 계산된 경도", f"{calc_lon:.4f}°E")
    c3.metric("📉 평균 위치 오차 (RMSE)", f"{calc_rmse:.1f} km")
    
    st.markdown("---")

    # 3. 지도 시각화 (Folium + HeatMap + Markers)
    st.subheader("🗺️ 진앙 분석 지도 (확률 분포 및 오차 비교)")
    
    # 지도 생성 (첫 번째 관측소 기준)
    first_sta = list(st.session_state.picks.keys())[0]
    center = [station_db.loc[first_sta, 'latitude'], station_db.loc[first_sta, 'longitude']]
    m = folium.Map(location=center, zoom_start=8, tiles='CartoDB positron')

    # A. ⭐️ 확률 분포 열지도 (HeatMap) 레이어 추가
    # 원들이 겹치는 확률이 높은 영역을 붉은색으로 시각화
    if heatmap_data:
        HeatMap(heatmap_data, radius=20, blur=15, min_opacity=0.2, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)

    # B. ⭐️ 계산된 최적 진앙지 (파란색 별) 마커
    folium.Marker(
        [calc_lat, calc_lon],
        popup=f"<b>계산된 진앙</b><br>오차: {calc_rmse:.1f}km",
        icon=folium.Icon(color="blue", icon="star")
    ).add_to(m)

    # C. 관측소 및 학생 피킹 원 그리기
    for sta, data in st.session_state.picks.items():
        lat, lon = station_db.loc[sta, 'latitude'], station_db.loc[sta, 'longitude']
        
        # 관측소 마커 (검은색 작은 점)
        folium.CircleMarker([lat, lon], radius=3, color="black", fill=True).add_to(m)
        
        # 학생 피킹 반경 원 (연한 파란색선)
        folium.Circle(
            location=[lat, lon], radius=data['dist'] * 1000,
            color="#3366cc", weight=1, fill=False, opacity=0.3
        ).add_to(m)

    # D. 실제 지진 위치 입력 및 비교 (연구원님 입력용)
    with st.expander("🎯 실제 지진 위치와 비교하기 (연구원님 입력)", expanded=False):
        col_t1, col_t2 = st.columns(2)
        true_lat = col_t1.number_input("실제 위도", value=35.80, format="%.4f")
        true_lon = col_t2.number_input("실제 경도", value=129.20, format="%.4f")
        
        # 오차 계산
        true_dist = haversine_km(calc_lat, calc_lon, true_lat, true_lon)
        st.info(f"📍 **분석 결과:** 학생들이 찾은 진앙은 실제 위치로부터 약 **{true_dist:.1f} km** 떨어져 있습니다.")

        # 실제 위치 마커 (빨간색 별)
        folium.Marker(
            [true_lat, true_lon], popup="실제 위치",
            icon=folium.Icon(color="red", icon="star")
        ).add_to(m)

    # 지도 출력
    st_folium(m, width=1000, height=600)
    st.markdown("💡 **Tip:** 지도의 확률 분포(붉은 영역)와 파란색 별(계산된 위치)이 빨간색 별(실제 위치)과 얼마나 일치하는지 토론해 보세요.")
