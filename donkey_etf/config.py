"""
config.py — 전역 설정 (폰트 / 벤치마크 / 기간 / 무위험수익률 / 출력경로)
모든 분석 스크립트가 여기서 설정을 가져갑니다. 한 곳만 바꾸면 전체에 반영돼요.
"""
import os
import platform
import warnings

warnings.filterwarnings("ignore")

# ── 백테스트 공통 파라미터 ───────────────────────────────
BACKTEST_YEARS = 5            # 기본 백테스트 기간 (최소 5년). 스크립트별로 override 가능
RISK_FREE_RATE = 0.02         # 무위험수익률(연 2%). Cash Drag · Sharpe 등에 사용
TRADING_DAYS = 252

# ── 벤치마크 ─────────────────────────────────────────────
PRIMARY_BENCHMARK = "SPY"     # 상대지표(베타/알파/캡처) 기준
BENCHMARKS = ["QQQ", "SPY", "SCHD"]

# ── 출력 경로 ────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_korean_font():
    """OS별 한글 폰트 설정. 리눅스에선 NanumGothic을 명시적으로 등록."""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        # 리눅스: NanumGothic 경로를 직접 등록 (없으면 기본 폰트로 폴백)
        path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def out(filename):
    """출력 파일 전체 경로를 돌려줍니다."""
    return os.path.join(OUTPUT_DIR, filename)
