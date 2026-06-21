"""
portfolio.py — 포트폴리오 단일 진실 공급원 (Single Source of Truth)

기존 18개 스크립트는 비중을 3가지 다른 방식으로 적어서 서로 안 맞았어요:
  - 정수 capital_allocation(16/14/5/8) + 내부비중  → LLY 가중치 0.0992
  - 금액 4000/43000 직접 표기                       → LLY 가중치 0.0930
  - 일부 파일은 Kiwoom 비중이 2가 아니라 1 (총 34)  → 아예 다른 포트폴리오

→ 여기서는 '하루에 각 종목에 넣는 실제 금액(원)'을 유일한 기준으로 삼고,
   비중·슬리브비중·티커목록을 전부 여기서 파생합니다. 한 곳만 고치면 전체 반영.
"""

# ── 유일한 기준: 일일 DCA 금액(원). 총합 43,000원 ──────────────
AMOUNTS = {
    "NH (기술/헬스)": {
        "LLY": 4000, "NVDA": 2000, "AVGO": 2000, "MSFT": 1000, "GOOG": 1000,
        "ASML": 1000, "TSM": 1000, "ABBV": 1000, "TMO": 1000, "AMAT": 1000,
    },
    "Toss (가치/방어)": {
        "JPM": 1500, "V": 1500, "COST": 1500, "CVX": 1500, "BX": 1500,
        "WM": 1500, "UNH": 1500, "RTX": 1500, "BRK-B": 1500, "PEP": 1500,
    },
    "Mirae (혁신 벤처)": {
        "SYM": 1000, "ISRG": 1000, "FTNT": 1000, "ETN": 1000, "RKLB": 1000,
    },
    "Kiwoom (전통 인프라)": {
        "NEE": 2000, "PLD": 2000, "CEG": 2000, "LIN": 2000,
    },
}

# 슬리브별 고유 색상 (시각화 일관성용)
SLEEVE_COLORS = {
    "NH (기술/헬스)": "#1f77b4",
    "Toss (가치/방어)": "#ff7f0e",
    "Mirae (혁신 벤처)": "#d62728",
    "Kiwoom (전통 인프라)": "#2ca02c",
}


def get_sleeves():
    """슬리브 이름 리스트."""
    return list(AMOUNTS.keys())


def get_tickers():
    """29개 종목 티커 리스트 (정의 순서 유지)."""
    return [t for sleeve in AMOUNTS.values() for t in sleeve]


def total_amount():
    return sum(v for sleeve in AMOUNTS.values() for v in sleeve.values())


def get_weights():
    """종목별 최종 비중 dict (전체 합 = 1). 금액 기준에서 직접 파생."""
    tot = total_amount()
    w = {}
    for sleeve in AMOUNTS.values():
        for t, amt in sleeve.items():
            w[t] = w.get(t, 0) + amt / tot
    return w


def get_sleeve_weights():
    """슬리브별 자본 비중 dict (합 = 1)."""
    tot = total_amount()
    return {s: sum(d.values()) / tot for s, d in AMOUNTS.items()}


def get_sleeve_map():
    """티커 → 소속 슬리브 dict."""
    return {t: s for s, d in AMOUNTS.items() for t in d}


def inner_weights(sleeve):
    """특정 슬리브 내부의 종목별 비중 dict (슬리브 내 합 = 1)."""
    d = AMOUNTS[sleeve]
    s = sum(d.values())
    return {t: a / s for t, a in d.items()}


def sleeve_returns(returns):
    """일간 수익률 DataFrame을 받아 4개 슬리브의 일간 수익률 DataFrame을 만듭니다."""
    import pandas as pd
    out = pd.DataFrame(index=returns.index)
    for sleeve in AMOUNTS:
        iw = inner_weights(sleeve)
        cols = [t for t in iw if t in returns.columns]
        out[sleeve] = sum(returns[t] * iw[t] for t in cols)
    return out


if __name__ == "__main__":
    # 자가 점검: 비중 합 == 1, 종목 수 == 29
    w = get_weights()
    print(f"종목 수: {len(get_tickers())}  (기대: 29)")
    print(f"비중 합: {sum(w.values()):.6f}  (기대: 1.0)")
    print(f"총 일일금액: {total_amount():,}원")
    print("\n슬리브 비중:")
    for s, sw in get_sleeve_weights().items():
        print(f"  {s:<18} {sw*100:5.1f}%")
