"""
data.py — 데이터 계층 (다운로드 / 수익률 / Cash Drag)

기존 18개 파일에 복붙돼 있던 'yfinance 다운로드 + Adj Close/멀티인덱스 추출'을
여기 한 곳으로 통합. 또한 야후 접속이 막힌 환경에서도 파이프라인이 돌도록
합성(synthetic) 데이터 폴백을 넣었습니다(오프라인 데모용, 콘솔에 경고 출력).
"""
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

import config


def _extract_adj_close(raw):
    """yfinance 반환물에서 Adj Close(없으면 Close)를 안전하게 추출."""
    if isinstance(raw.columns, pd.MultiIndex):
        for lvl in range(raw.columns.nlevels):
            if "Adj Close" in raw.columns.get_level_values(lvl):
                return raw.xs("Adj Close", level=lvl, axis=1)
        return raw.xs("Close", level=0, axis=1)
    return raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]


def _synthetic_prices(tickers, years, seed=42):
    """야후 접속 불가 시 GBM 기반 합성 가격(데모용). 실제 수치 아님."""
    rng = np.random.default_rng(seed)
    n = int(years * config.TRADING_DAYS)
    idx = pd.bdate_range(end=datetime.today().date(), periods=n)
    n = len(idx)  # 실제 생성된 길이에 맞춤 (불일치 방지)
    # 종목마다 약간씩 다른 드리프트/변동성 부여 (그럴듯한 분산)
    out = {}
    for i, t in enumerate(tickers):
        mu = rng.uniform(0.05, 0.30) / config.TRADING_DAYS
        sig = rng.uniform(0.15, 0.45) / np.sqrt(config.TRADING_DAYS)
        shocks = rng.normal(mu, sig, n)
        out[t] = 100 * np.exp(np.cumsum(shocks))
    return pd.DataFrame(out, index=idx)


def download_prices(tickers, years=None, end=None):
    """가격 시계열(DataFrame) 반환. 실패 시 합성 데이터로 폴백.

    실제 사용 환경(본인 PC)에서는 yfinance로 실데이터를 받고,
    상장 전 구간은 NaN으로 남아 to_returns()의 Cash Drag가 처리합니다.
    """
    years = years or config.BACKTEST_YEARS
    end = end or datetime.today()
    start = end - relativedelta(years=years)
    try:
        import yfinance as yf
        raw = yf.download(
            tickers, start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"), auto_adjust=False, progress=False,
        )
        prices = _extract_adj_close(raw).dropna(how="all")
        if prices.empty or prices.shape[0] < 30:
            raise ValueError("empty/insufficient data")
        return prices
    except Exception as e:
        print(f"  [오프라인 모드] yfinance 실패({type(e).__name__}) → 합성 데이터 사용. "
              f"실제 백테스트는 본인 PC에서 실행하세요.")
        return _synthetic_prices(tickers, years)


def to_returns(prices, fill_cash=True):
    """일간 수익률. fill_cash=True면 상장 전(NaN) 구간을 무위험수익률로 채움(Cash Drag).

    '0으로 채우기'는 현금 파킹과 달라 가중합을 왜곡하므로, 일관되게
    일간 무위험수익률로 채워 '상장 전엔 현금 보유' 가정을 정확히 반영합니다.
    """
    ret = prices.pct_change()
    if fill_cash:
        daily_rf = (1 + config.RISK_FREE_RATE) ** (1 / config.TRADING_DAYS) - 1
        ret = ret.fillna(daily_rf)
    ret.iloc[0] = ret.iloc[0].fillna(0) if hasattr(ret.iloc[0], "fillna") else 0
    return ret.dropna(how="all")


def portfolio_returns(returns, weights):
    """종목 수익률 + 비중(dict/Series) → 포트폴리오 일간 수익률 Series."""
    w = pd.Series(weights)
    cols = [c for c in w.index if c in returns.columns]
    return (returns[cols] * w[cols]).sum(axis=1)
