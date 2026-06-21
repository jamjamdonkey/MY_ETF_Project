"""
a2_sleeves.py — [레벨 2] 4개 ETF(슬리브) 시각화

백서 대응: ETF(슬리브 누적수익) + MPT(효율적 투자선) + GROUP(PCA 슬리브 분리)
산출물: 02_sleeves_cumulative.png, 02_sleeves_frontier.png, 02_sleeves_pca.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config, data, metrics, portfolio

config.set_korean_font()
TD = config.TRADING_DAYS


def run(years=None):
    years = years or max(config.BACKTEST_YEARS, 5)
    tickers = portfolio.get_tickers()
    prices = data.download_prices(tickers + ["QQQ", "SPY"], years=years)
    rets = data.to_returns(prices)
    sleeves = portfolio.sleeve_returns(rets)
    sleeve_w = portfolio.get_sleeve_weights()
    colors = portfolio.SLEEVE_COLORS

    # ── (1) 슬리브별 누적수익 ───────────────────────────────────
    plt.figure(figsize=(14, 7))
    for s in sleeves.columns:
        cum = (1 + sleeves[s]).cumprod()
        plt.plot(cum.index, (cum - 1) * 100, lw=2.5, color=colors[s],
                 label=f"{s} ({(cum.iloc[-1]-1)*100:.0f}%)")
    for bm, c in zip(["QQQ", "SPY"], ["gray", "black"]):
        if bm in rets.columns:
            cum = (1 + rets[bm]).cumprod()
            plt.plot(cum.index, (cum - 1) * 100, lw=1.5, ls="--", color=c, label=bm)
    plt.title(f"[레벨2] 4개 슬리브 누적수익률 ({years}년)", fontsize=16, fontweight="bold")
    plt.ylabel("Cumulative Return (%)")
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(config.out("02_sleeves_cumulative.png"), dpi=150)
    plt.close()

    # ── (2) MPT 효율적 투자선 (4 슬리브 자산) ───────────────────
    mu = sleeves.mean() * TD
    cov = sleeves.cov() * TD
    n = 8000
    rng = np.random.default_rng(42)
    res = np.zeros((3, n))
    for i in range(n):
        w = rng.random(len(sleeves.columns)); w /= w.sum()
        r = w @ mu.values
        vol = np.sqrt(w @ cov.values @ w)
        res[:, i] = [vol, r, (r - config.RISK_FREE_RATE) / vol]
    cur_w = np.array([sleeve_w[s] for s in sleeves.columns])
    cur_r, cur_vol = cur_w @ mu.values, np.sqrt(cur_w @ cov.values @ cur_w)
    opt = res[2].argmax()

    plt.figure(figsize=(13, 8))
    sc = plt.scatter(res[0] * 100, res[1] * 100, c=res[2], cmap="YlGnBu", s=8, alpha=0.4)
    plt.colorbar(sc, label="Sharpe Ratio")
    plt.scatter(res[0, opt] * 100, res[1, opt] * 100, marker="*", color="red", s=450,
                label="최적 (Max Sharpe)", zorder=5)
    plt.scatter(cur_vol * 100, cur_r * 100, marker="X", color="black", s=220,
                label="현재 비중", zorder=5)
    for i, s in enumerate(sleeves.columns):
        plt.scatter(np.sqrt(cov.iloc[i, i]) * 100, mu.iloc[i] * 100, marker="D",
                    color=colors[s], s=110, edgecolor="k", zorder=4)
    plt.title("[레벨2] 효율적 투자선 — 4슬리브 비중 최적화", fontsize=16, fontweight="bold")
    plt.xlabel("연환산 변동성 (%)"); plt.ylabel("연환산 기대수익 (%)")
    plt.legend(loc="upper left"); plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.out("02_sleeves_frontier.png"), dpi=150)
    plt.close()

    # ── (3) PCA 슬리브 구조 분리 ────────────────────────────────
    smap = portfolio.get_sleeve_map()
    feat = pd.DataFrame(index=tickers)
    feat["Return"] = rets[tickers].mean() * TD
    feat["Vol"] = rets[tickers].std() * np.sqrt(TD)
    cum = (1 + rets[tickers]).cumprod()
    feat["MDD"] = (cum / cum.cummax() - 1).min()
    mkt_var = rets["SPY"].var() if "SPY" in rets else rets[tickers].mean(axis=1).var()
    feat["Beta"] = [np.cov(rets[t], rets.get("SPY", rets[tickers].mean(axis=1)))[0, 1] / mkt_var
                    for t in tickers]
    feat["Sleeve"] = [smap[t] for t in tickers]

    scaled = StandardScaler().fit_transform(feat[["Return", "Vol", "MDD", "Beta"]])
    pcs = PCA(n_components=2).fit(scaled)
    feat[["PC1", "PC2"]] = pcs.transform(scaled)

    plt.figure(figsize=(13, 9))
    sns.scatterplot(data=feat, x="PC1", y="PC2", hue="Sleeve", palette=colors,
                    s=200, alpha=0.85, edgecolor="black")
    for t, r in feat.iterrows():
        plt.text(r["PC1"] + 0.05, r["PC2"] + 0.05, t, fontsize=9, alpha=0.8)
    plt.axhline(0, color="gray", ls="--", alpha=0.5); plt.axvline(0, color="gray", ls="--", alpha=0.5)
    plt.title("[레벨2] PCA 차원축소 — 4슬리브 구조 분리", fontsize=16, fontweight="bold")
    plt.xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]*100:.0f}%)")
    plt.ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]*100:.0f}%)")
    plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.out("02_sleeves_pca.png"), dpi=150)
    plt.close()

    print("[레벨2] 현재 vs 최적 슬리브 비중:")
    for i, s in enumerate(sleeves.columns):
        print(f"   {s:<18} 현재 {cur_w[i]*100:4.1f}%")
    print(f"   현재 Sharpe {(cur_r-config.RISK_FREE_RATE)/cur_vol:.2f} / 최적 Sharpe {res[2,opt]:.2f}")
    print("→ 저장: 02_sleeves_cumulative.png, 02_sleeves_frontier.png, 02_sleeves_pca.png")


if __name__ == "__main__":
    run()
