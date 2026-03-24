import streamlit as st
import obspy
import plotly.graph_objects as go
import numpy as np

# 1. 모바일 화면을 위해 전체 레이아웃 사용 및 사이드바 기본 숨김
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.title("📱 지진파 피킹 실습")
st.markdown("💡 **Tip:** 두 손가락으로 그래프를 벌려서 줌인(Zoom-in)하고 도착 시간을 찾으세요.")

# --- 가상 데이터 생성 (실제 SAC 파일 연동 시 이 부분 수정) ---
@st.cache_data
def get_sample_data():
    tr = obspy.Trace(data=np.random.randn(2000) * 0.1)
    tr.stats.sampling_rate = 100
    tr.data[500:800] += np.sin(np.linspace(0, 20, 300)) * 2
    tr.data[1200:1700] += np.sin(np.linspace(0, 50, 500)) * 5
    return tr

tr = get_sample_data()
time_axis = tr.times()
amplitude = tr.data
max_time = float(time_axis[-1])
# -------------------------------------------------------------

# 2. 모바일 친화적인 터치 입력부 (Expander 사용으로 공간 절약)
with st.expander("⏱️ P파 / S파 도착 시간 입력 (여기를 터치하세요)", expanded=True):
    # 모바일에서는 숫자 직접 입력보다 슬라이더가 조작하기 편합니다.
    p_pick = st.slider("P파 도착 시간 (초)", 0.0, max_time, 0.0, step=0.1)
    s_pick = st.slider("S파 도착 시간 (초)", 0.0, max_time, 0.0, step=0.1)

# 3. Plotly 모바일 맞춤 설정
fig = go.Figure()

# 원본 파형
fig.add_trace(go.Scatter(x=time_axis, y=amplitude, mode='lines', line=dict(color='black', width=1)))

# P파 표시 (모바일에서는 글씨가 겹치지 않게 위치 조정)
if p_pick > 0:
    fig.add_vline(x=p_pick, line_width=2, line_dash="dash", line_color="red")
    fig.add_annotation(x=p_pick, y=max(amplitude)*0.9, text="P", showarrow=False, font=dict(color="red", size=16))

# S파 표시
if s_pick > 0:
    fig.add_vline(x=s_pick, line_width=2, line_dash="dash", line_color="blue")
    fig.add_annotation(x=s_pick, y=max(amplitude)*0.9, text="S", showarrow=False, font=dict(color="blue", size=16))

# 레이아웃: 여백 최소화 및 모바일 제스처 최적화
fig.update_layout(
    margin=dict(l=5, r=5, t=10, b=30), # 좌우 여백을 5px로 최소화
    dragmode="zoom",                   # 핀치 줌 활성화
    xaxis_title="시간 (초)",
    hovermode="x unified",
    height=300                         # 모바일 세로 화면에 맞춘 그래프 높이
)

# Plotly 차트 렌더링 (화면 가리는 툴바 제거)
config = {'displayModeBar': False} 
st.plotly_chart(fig, use_container_width=True, config=config)

# 4. 결과 출력 (모바일 화면에 맞게 간결한 한 줄 출력)
if p_pick > 0 and s_pick > 0:
    if s_pick > p_pick:
        s_p_time = s_pick - p_pick
        st.success(f"✅ S-P 시간차: **{s_p_time:.2f}초** | 📍 거리: 약 **{s_p_time * 8.0:.1f}km**")
    else:
        st.error("⚠️ S파가 먼저 올 수 없습니다.")
