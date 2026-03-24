import streamlit as st
import obspy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import glob
import os
import folium
from streamlit_folium import st_folium

# ==========================================
# 0. 페이지 및 세션 상태(저장공간) 초기화
# ==========================================
st.set_page_config(layout="wide", page_title="지진위치결정 실습", initial_sidebar_state="expanded")

# 학생들의 피킹 결과를 저장할 공간 생성
if 'picks' not in st.session_state:
    st.session_state.picks = {}

# ==========================================
# 1. 데이터 로딩 함수
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
    """SAC 파일을 읽고 노이즈 제거를 위한 필터링을 수행합니다."""
    eventtime = obspy.UTCDateTime("2016-09-12T11:33:00")
    starttime = eventtime - 50
    endtime = eventtime + 500
    tr = obspy.read(file_path)[0]
    tr.trim(starttime, endtime)
    tr.detrend('demean')
    tr.taper(0.05)
    tr.filter('bandpass', freqmin=1.0, freqmax=10.0)
    return tr

station_db = load_station_info()
sac_inventory = get_station_components()

if not sac_inventory:
    st.error("⚠️ 'Waveform' 폴더에서 SAC 파일을 찾을 수 없습니다.")
    st.stop()

# ==========================================
# 2. 사이드바 네비게이션 (모드 선택)
# ==========================================
st.sidebar.title("🌋 실습 진행 단계")
app_mode = st.sidebar.radio("단계 선택", ["1. 지진파 피킹 (관측소별)", "2. 진앙 위치 결정 (지도)"])
st.sidebar.markdown("---")

# ==============================================================================
# [모드 1] 지진파 피킹 화면
# ==============================================================================
if app_mode == "1. 지진파 피킹 (관측소별)":
    st.sidebar.subheader("📡 관측소 선택")
    selected_sta = st.sidebar.radio("👉 피킹할 관측소", sorted(list(sac_inventory.keys())))
    
    st.title(f"🔍 {selected_sta} 관측소 파형 피킹")
    
    # 이전에 저장해둔 값이 있으면 불러오기
    saved_p = st.session_state.picks.get(selected_sta, {}).get('p', 0.0)
    saved_s = st.session_state.picks.get(selected_sta, {}).get('s', 0.0)

    comps = sac_inventory[selected_sta]
    traces = {}
    max_time = 0
    for comp in ['Z', 'N', 'E']:
        if comp in comps:
            tr = process_trace(comps[comp])
            traces[comp] = tr
            max_time = max(max_time, float(tr.times()[-1]))

    # 시간 입력창 (직접 입력 및 미세조정)
    st.markdown("### ⏱️ 도달 시간 입력")
    col1, col2 = st.columns(2)
    with col1:
        p_pick = st.number_input("🔴 P파 도착 시간 (Z성분 위주)", min_value=0.0, max_value=max_time, value=saved_p, step=0.01, format="%.2f")
    with col2:
        s_pick = st.number_input("🔵 S파 도착 시간 (N/E성분 위주)", min_value=0.0, max_value=max_time, value=saved_s, step=0.01, format="%.2f")

    # 저장 버튼 (피킹 결과를 세션에 저장)
    if st.button(f"💾 {selected_sta} 피킹 결과 저장하기", use_container_width=True):
        if s_pick > p_pick:
            distance = (s_pick - p_pick) * 7.5 # Omori 공식 (k=7.5)
            st.session_state.picks[selected_sta] = {'p': p_pick, 's': s_pick, 'dist': distance}
            st.success(f"✅ {selected_sta} 관측소 데이터 저장 완료! (S-P: {s_pick-p_pick:.2f}s, 거리: {distance:.1f}km)")
        else:
            st.error("⚠️ S파 도착 시간이 P파보다 빠를 수 없습니다.")

    # 그래프 렌더링
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("<b>Z 성분</b>", "<b>N 성분</b>", "<b>E 성분</b>"))
    colors = {'Z': 'black', 'N': '#3366cc', 'E': '#dc3912'}
    row_idx = 1
    for comp in ['Z', 'N', 'E']:
        if comp in traces:
            fig.add_trace(go.Scatter(x=traces[comp].times(), y=traces[comp].data, mode='lines', line=dict(color=colors[comp], width=1)), row=row_idx, col=1)
        row_idx += 1

    if p_pick > 0:
        fig.add_vline(x=p_pick, line_width=2, line_color="red", row="all", col=1)
    if s_pick > 0:
        fig.add_vline(x=s_pick, line_width=2, line_color="blue", row="all", col=1)

    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=30), dragmode="zoom", hovermode="x unified", showlegend=False, uirevision="constant")
    fig.update_yaxes(fixedrange=True)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True, 'doubleClick': 'reset'})

# ==============================================================================
# [모드 2] 진앙 위치 결정 화면 (지도 시각화)
# ==============================================================================
elif app_mode == "2. 진앙 위치 결정 (지도)":
    st.title("📍 진앙 위치 결정 (Epicenter Location)")
    
    if len(st.session_state.picks) == 0:
        st.warning("⚠️ 아직 피킹된 데이터가 없습니다. 왼쪽 메뉴에서 '1. 지진파 피킹'으로 돌아가 관측소 데이터를 저장해 주세요.")
        st.stop()

    # 1. 피킹 현황 요약표 보여주기
    st.subheader("📊 관측소별 피킹 및 거리 계산 결과")
    result_df = pd.DataFrame(st.session_state.picks).T
    result_df.columns = ['P파(s)', 'S파(s)', '진원거리(km)']
    st.dataframe(result_df.style.format("{:.2f}"))

    if len(st.session_state.picks) < 3:
        st.info("💡 정확한 진앙을 찾기(3변 측량) 위해서는 최소 3개 이상의 관측소 데이터가 필요합니다. 피킹을 더 진행해 보세요!")

    # 2. 실제 지진 위치 입력 (비교용)
    st.markdown("---")
    st.subheader("🎯 실제 지진 발생 위치 입력 (비교용)")
    col1, col2 = st.columns(2)
    with col1:
        true_lat = st.number_input("실제 위도 (Latitude)", value=35.8, format="%.4f")
    with col2:
        true_lon = st.number_input("실제 경도 (Longitude)", value=129.2, format="%.4f")

    # 3. 지도 생성 (Folium)
    if station_db is not None:
        # 지도의 중심을 첫 번째 관측소로 설정
        first_sta = list(st.session_state.picks.keys())[0]
        center_lat = station_db.loc[first_sta, 'latitude']
        center_lon = station_db.loc[first_sta, 'longitude']
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='CartoDB positron')

        # 실제 지진 위치 표시 (빨간색 별)
        folium.Marker(
            [true_lat, true_lon], 
            popup="실제 지진 발생 위치",
            icon=folium.Icon(color="red", icon="star")
        ).add_to(m)

        # 관측소 및 계산된 반경(원) 그리기
        for sta, data in st.session_state.picks.items():
            if sta in station_db.index:
                lat = station_db.loc[sta, 'latitude']
                lon = station_db.loc[sta, 'longitude']
                radius_km = data['dist']
                
                # 관측소 위치 (파란색 마커)
                folium.Marker(
                    [lat, lon], 
                    popup=f"{sta} 관측소",
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m)
                
                # 거리 반경 원 그리기 (radius는 미터 단위이므로 1000을 곱함)
                folium.Circle(
                    location=[lat, lon],
                    radius=radius_km * 1000,
                    color="blue",
                    weight=2,
                    fill_opacity=0.1,
                    popup=f"반경: {radius_km:.1f} km"
                ).add_to(m)

        st.markdown("### 🗺️ 피킹 결과를 바탕으로 한 3변 측량 지도")
        st.write("파란색 원들이 겹치는 교집합 구역이 학생이 찾은 진앙지입니다. 실제 위치(빨간 별)와 비교해 보세요!")
        st_folium(m, width=1000, height=600)
    else:
        st.error("station.txt 정보를 불러오지 못해 지도를 그릴 수 없습니다.")
