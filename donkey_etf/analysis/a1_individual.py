"""
a1_individual.py — [레벨 1] 개별 29종목 시각화

백서 대응: MYPORT(개별 종목 배수 추이) + HEATMAP(종목 간 상관계수)
산출물: 01_individual_multiples.png, 01_individual_corr.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import config, data, portfolio

config.set_korean_font()


def run(years=None):
    years = years or config.BACKTEST_YEARS
    tickers = portfolio.get_tickers()
    bms = ["QQQ", "SPY"]
    prices = data.download_prices(tickers + bms, years=years)

    # ── (1) 개별 종목 시초가 대비 배수 (로그 스케일) ──────────────
    mult = prices / prices.apply(lambda c: c.dropna().iloc[0])
    mult = mult.fillna(1.0)  # 상장 전 구간은 현금(1배)으로

    fig, ax = plt.subplots(figsize=(15, 8))
    cmap = plt.cm.get_cmap("hsv", len(tickers))
    for i, t in enumerate(tickers):
        if t in mult.columns:
            ax.plot(mult.index, mult[t], color=cmap(i), lw=1.0, alpha=0.35, zorder=2)
    for bm, c in zip(bms, ["darkred", "black"]):
        if bm in mult.columns:
            ax.plot(mult.index, mult[bm], color=c, lw=3.5, zorder=5, label=bm)
    ax.axhline(1, color="black", lw=1.2, zorder=4)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0f}배" if x >= 1 else f"{x:.1f}배"))
    ax.set_title(f"[레벨1] 개별 29종목 vs 벤치마크 — 시초가 대비 배수 ({years}년)",
                 fontsize=16, fontweight="bold")
    ax.set_ylabel("배수 (Log Scale)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.out("01_individual_multiples.png"), dpi=150)
    plt.close()

    # ── (2) 종목 간 상관계수 히트맵 ─────────────────────────────
    rets = prices[tickers].pct_change().dropna(how="all")
    corr = rets.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", vmin=0, vmax=1, center=0.5,
                linewidths=0.5, square=True, cbar_kws={"shrink": .8})
    plt.title("[레벨1] 29종목 상관계수 히트맵 (0~1)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.out("01_individual_corr.png"), dpi=150)
    plt.close()

    # ── 콘솔: 상관 최고/최저 쌍 ─────────────────────────────────
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).unstack().dropna()
    print("[레벨1] 가장 같이 움직이는 5쌍 (중복 리스크):")
    for (a, b), v in upper.sort_values(ascending=False).head(5).items():
        print(f"   {a:<5} & {b:<5} : {v:.3f}")
    print("[레벨1] 가장 따로 노는 5쌍 (분산 효과):")
    for (a, b), v in upper.sort_values().head(5).items():
        print(f"   {a:<5} & {b:<5} : {v:.3f}")
    print("→ 저장: 01_individual_multiples.png, 01_individual_corr.png")


if __name__ == "__main__":
    run()
