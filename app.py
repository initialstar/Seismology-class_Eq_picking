import streamlit as st
import obspy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import glob
import os

# ==========================================
# 1. 웹페이지 기본 설정
# ==========================================
# PC에서는 사이드바가 열려있고, 모바일에서는 햄버거 메뉴로 자동 숨김 처리됩니다.
st.set_page_config(layout="wide", page_title="지진위치결정 실습", initial_sidebar_state="expanded")

# ==========================================
# 2. 데이터 로딩 및 전처리 함수 (캐싱 적용)
# ==========================================

@st.cache_data
def load_station_info():
    """station.txt 파일에서 관측소 위도/경도 정보를 읽어옵니다."""
    try:
        # sep='\s+' : 스페이스바가 몇 개든 공백을 기준으로 열을 나눔
        # header=0 : 맨 첫 줄(Network Station ...)을 컬럼 이름으로 인식
        df = pd.read_csv('station.txt', sep='\s+', header=0)
        
        # 코드 호환성을 위해 컬럼 이름을 소문자로 변경
        df.columns = ['network', 'station_name', 'latitude', 'longitude', 'elevation']
        
        # 'Station' 열을 검색용 인덱스로 설정
        return df.set_index('station_name')
    except Exception as e:
        st.error(f"station.txt 읽기 에러: {e}")
        return None

@st.cache_data
def get_station_components():
    """Waveform 폴더 내의 SAC 파일을 읽어 관측소별/성분별(Z,N,E)로 분류합니다."""
    files = glob.glob(os.path.join('Waveform', '*.sac')) + glob.glob(os.path.join('Waveform', '*.SAC'))
    sta_dict = {}
    
    for f in files:
        try:
            st_head = obspy.read(f, headonly=True)
            tr = st_head[0]
            sta = tr.stats.station.upper()
            chan = tr.stats.channel.upper()
            comp = chan[-1] # Z, N, E 추출
            
            if sta not in sta_dict:
                sta_dict[sta] = {}
            sta_dict[sta][comp] = f
        except:
            continue
    return sta_dict

@st.cache_data
def process_trace(file_path):
    """SAC 파일을 읽고 노이즈 제거를 위한 필터링을 수행합니다."""
    tr = obspy.read(file_path)[0]
    tr.detrend('demean')
    tr.taper(0.05)
    tr.filter('bandpass', freqmin=1.0, freqmax=10.0)
    return tr

# 데이터 불러오기 실행
station_db = load_station_info()
sac_inventory = get_station_components()

if not sac_inventory:
    st.error("⚠️ 'Waveform' 폴더에서 SAC 파일을 찾을 수 없습니다. 깃허브 경로를 확인해주세요.")
    st.stop()

# ==========================================
# 3. 왼쪽 사이드바 (관측소 선택 컨트롤러)
# ==========================================
st.sidebar.title("📡 관측소 목록")
st.sidebar.write("분석할 관측소를 선택하세요.")

# 폴더에서 찾은 관측소 목록을 라디오 버튼으로 표시
selected_sta = st.sidebar.radio("👉 관측소 선택", sorted(list(sac_inventory.keys())))

st.sidebar.markdown("---")
st.sidebar.subheader("📍 관측소 정보")

if station_db is not None and selected_sta in station_db.index:
    sta_lat = station_db.loc[selected_sta, 'latitude']
    sta_lon = station_db.loc[selected_sta, 'longitude']
    st.sidebar.success(f"**{selected_sta}**\n\n위도: {sta_lat:.3f}°\n\n경도: {sta_lon:.3f}°")
else:
    st.sidebar.warning("위치 정보가 없습니다.")

# ==========================================
# 4. 메인 화면 (3성분 파형 출력 및 피킹 입력)
# ==========================================
st.title(f"🌋 {selected_sta} 관측소 파형 분석")
st.markdown("💡 **Tip:** 파형을 좌우로 **드래그(Drag)**하여 확대하고, **더블 클릭**하면 원래대로 돌아옵니다.")

# 선택된 관측소의 3성분 데이터 로드
comps = sac_inventory[selected_sta]
traces = {}
max_time = 0

for comp in ['Z', 'N', 'E']:
    if comp in comps:
        tr = process_trace(comps[comp])
        traces[comp] = tr
        max_time = max(max_time, float(tr.times()[-1]))
        
if not traces:
    st.warning("선택한 관측소의 파형을 불러올 수 없습니다.")
    st.stop()

# 피킹 시간 입력 슬라이더
st.markdown("### ⏱️ 도달 시간 입력")
col1, col2 = st.columns(2)
with col1:
    p_pick = st.slider("🔴 P파 도착 시간 (Z성분 위주)", 0.0, max_time, 0.0, step=0.01)
with col2:
    s_pick = st.slider("🔵 S파 도착 시간 (N/E성분 위주)", 0.0, max_time, 0.0, step=0.01)

# ==========================================
# 5. Plotly 3단 그래프 렌더링 (동기화 및 마우스 조작 최적화)
# ==========================================
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=("<b>Z 성분</b> (상하)", "<b>N 성분</b> (남북)", "<b>E 성분</b> (동서)"))

colors = {'Z': 'black', 'N': '#3366cc', 'E': '#dc3912'}
row_idx = 1

for comp in ['Z', 'N', 'E']:
    if comp in traces:
        tr = traces[comp]
        fig.add_trace(
            go.Scatter(x=tr.times(), y=tr.data, mode='lines', line=dict(color=colors[comp], width=1)),
            row=row_idx, col=1
        )
    row_idx += 1

# P파 피킹 수직선
if p_pick > 0:
    fig.add_vline(x=p_pick, line_width=2, line_color="red", row="all", col=1)
    fig.add_annotation(x=p_pick, y=1, yref="paper", text="P", showarrow=False, font=dict(color="red", size=16), yanchor="bottom")

# S파 피킹 수직선
if s_pick > 0:
    fig.add_vline(x=s_pick, line_width=2, line_color="blue", row="all", col=1)
    fig.add_annotation(x=s_pick, y=1, yref="paper", text="S", showarrow=False, font=dict(color="blue", size=16), yanchor="bottom")

# 레이아웃 및 마우스 조작 설정
fig.update_layout(
    height=600, 
    margin=dict(l=10, r=10, t=40, b=30),
    dragmode="zoom",       # 마우스 드래그 시 확대 기능 활성화
    hovermode="x unified", # 마우스 오버 시 3성분 시간/진폭 동시 확인
    showlegend=False
)

# ⭐️ 핵심: Y축(진폭) 줌 고정. 마우스 드래그 시 X축(시간)만 확대되도록 설정
fig.update_yaxes(fixedrange=True)
fig.update_xaxes(title_text="시간 (초)", row=3, col=1)

# 상단 툴바 숨김 및 더블클릭/마우스휠 설정
config = {
    'displayModeBar': False, 
    'scrollZoom': True,      
    'doubleClick': 'reset'   
}
st.plotly_chart(fig, use_container_width=True, config=config)

# ==========================================
# 6. 결과 출력
# ==========================================
if p_pick > 0 and s_pick > 0:
    if s_pick > p_pick:
        s_p_time = s_pick - p_pick
        # Omori 공식을 이용한 간이 거리 계산 (k=7.5 가정)
        distance = s_p_time * 7.5
        st.success(f"✅ **S-P 시간차:** {s_p_time:.2f}초 | 📍 **{selected_sta}로부터의 거리:** 약 **{distance:.1f} km**")
    else:
        st.error("⚠️ S파 도착 시간이 P파보다 빠릅니다. 다시 확인해 주세요.")
