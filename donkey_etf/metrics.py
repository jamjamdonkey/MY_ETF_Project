"""
metrics.py — 퀀트 지표 통합 모듈

기존 QUANT.PY / QQQSPYSCHD.PY 등에 흩어져 중복 구현되던 지표 함수를 한 곳에.
모든 분석 스크립트가 동일한 정의로 지표를 계산하도록 통일합니다.
"""
import numpy as np
import pandas as pd
from scipy import stats

import config

RF = config.RISK_FREE_RATE
TD = config.TRADING_DAYS


# ── 기초 지표 ────────────────────────────────────────────
def cagr(returns):
    yrs = len(returns) / TD
    return ((1 + returns).prod() ** (1 / yrs) - 1) * 100


def volatility(returns):
    return returns.std() * np.sqrt(TD) * 100


def drawdown_series(returns):
    cum = (1 + returns).cumprod()
    return (cum / cum.cummax() - 1) * 100


def mdd(returns):
    return drawdown_series(returns).min()


def sharpe(returns):
    ann = returns.mean() * TD
    vol = returns.std() * np.sqrt(TD)
    return (ann - RF) / vol if vol else np.nan


def sortino(returns):
    ann = returns.mean() * TD
    dvol = returns[returns < 0].std() * np.sqrt(TD)
    return (ann - RF) / dvol if dvol else np.nan


def calmar(returns):
    m = mdd(returns)
    return cagr(returns) / abs(m) if m else np.nan


def win_rate(returns):
    return (returns > 0).mean() * 100


def var(returns, level=5):
    return np.percentile(returns, level) * 100


def cvar(returns, level=5):
    cut = np.percentile(returns, level)
    return returns[returns <= cut].mean() * 100


def ulcer_index(returns):
    dd = drawdown_series(returns) / 100
    return np.sqrt(np.mean(dd ** 2)) * 100


# ── 벤치마크 상대 지표 ───────────────────────────────────
def beta(returns, bm):
    cov = np.cov(returns, bm)[0, 1]
    v = np.var(bm)
    return cov / v if v else np.nan


def jensen_alpha(returns, bm):
    b = beta(returns, bm)
    ann, bm_ann = returns.mean() * TD, bm.mean() * TD
    return (ann - (RF + b * (bm_ann - RF))) * 100


def info_ratio(returns, bm):
    active = returns - bm
    te = active.std() * np.sqrt(TD)
    return ((returns.mean() - bm.mean()) * TD) / te if te else np.nan


def up_capture(returns, bm):
    up = bm > 0
    d = bm[up].mean()
    return (returns[up].mean() / d) * 100 if d else np.nan


def down_capture(returns, bm):
    dn = bm < 0
    d = bm[dn].mean()
    return (returns[dn].mean() / d) * 100 if d else np.nan


# ── 통합 티어시트 ────────────────────────────────────────
def tearsheet(returns, bm=None, name="Portfolio"):
    """단일 수익률 시계열의 종합 지표 Series."""
    m = {
        "CAGR(%)": cagr(returns),
        "Vol(%)": volatility(returns),
        "MDD(%)": mdd(returns),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "Calmar": calmar(returns),
        "WinRate(%)": win_rate(returns),
        "VaR95(%)": var(returns),
        "CVaR95(%)": cvar(returns),
        "Ulcer": ulcer_index(returns),
        "Skew": stats.skew(returns),
        "Kurt": stats.kurtosis(returns),
    }
    if bm is not None:
        m.update({
            "Beta": beta(returns, bm),
            "Alpha(%)": jensen_alpha(returns, bm),
            "InfoRatio": info_ratio(returns, bm),
            "UpCapture(%)": up_capture(returns, bm),
            "DownCapture(%)": down_capture(returns, bm),
        })
    return pd.Series(m, name=name).round(2)


def rolling_cagr(returns, years=5):
    """N년 롤링 연평균 수익률(%) 시계열."""
    window = int(TD * years)
    wealth = (1 + returns).cumprod()
    roll = (wealth / wealth.shift(window)) ** (1 / years) - 1
    return (roll * 100).dropna()


def risk_contribution(returns, weights):
    """종목별 컴포넌트 위험 기여도(%) — 합이 100%. 비중 vs 리스크 불균형 진단용."""
    w = pd.Series(weights)
    cols = [c for c in w.index if c in returns.columns]
    w = w[cols]
    cov = returns[cols].cov() * TD
    port_vol = np.sqrt(w.values @ cov.values @ w.values)
    marginal = cov.values @ w.values / port_vol
    comp = (marginal * w.values) / port_vol * 100
    return pd.Series(comp, index=cols)
