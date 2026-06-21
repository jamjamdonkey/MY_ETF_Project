"""
a3_portfolio.py — [레벨 3] 전체 29종목 포트폴리오 종합 검증

백서 대응: QUANT/QQQSPYSCHD(티어시트) + STRESSTEST(위기 MDD) +
          ROLLING(롤링 CAGR) + UPDONW(업/다운 캡처)
산출물: 03_port_tearsheet.png, 03_port_underwater.png,
        03_port_rolling.png, 03_port_capture.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config, data, metrics, portfolio

config.set_korean_font()

# 역사적 위기 구간 (실데이터일 때만 적용; 데이터 범위 밖이면 자동 skip)
CRISES = {
    "2008 금융위기": ("2007-10-01", "2009-03-31"),
    "2020 코로나": ("2020-02-15", "2020-04-30"),
    "2022 금리인상": ("2022-01-01", "2022-12-31"),
}


def run(years=15):
    tickers = portfolio.get_tickers()
    bms = ["QQQ", "SPY", "SCHD"]
    prices = data.download_prices(tickers + bms, years=years)
    rets = data.to_returns(prices)
    pr = data.portfolio_returns(rets, portfolio.get_weights())
    spy = rets["SPY"]

    # ── (1) 종합 티어시트 (포트 vs 벤치마크) ────────────────────
    cols = {"DONKEY (29종목)": pr}
    for bm in bms:
        if bm in rets.columns:
            cols[bm] = rets[bm]
    table = pd.DataFrame({name: metrics.tearsheet(r, spy, name) for name, r in cols.items()})
    print("[레벨3] 종합 퀀트 티어시트:\n" + table.to_string())

    fig, ax = plt.subplots(figsize=(12, 8)); ax.axis("off")
    tbl = ax.table(cellText=table.values, rowLabels=table.index,
                   colLabels=table.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for j in range(len(table.columns)):  # 헤더 강조
        tbl[0, j].set_facecolor("#1f77b4"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("[레벨3] DONKEY ETF 종합 퀀트 티어시트", fontsize=15, fontweight="bold", pad=20)
    plt.savefig(config.out("03_port_tearsheet.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── (2) 언더워터(낙폭) + 위기 구간 ─────────────────────────
    dd_p, dd_s = metrics.drawdown_series(pr), metrics.drawdown_series(spy)
    plt.figure(figsize=(15, 7))
    plt.fill_between(dd_p.index, dd_p, 0, color="royalblue", alpha=0.25)
    plt.plot(dd_p.index, dd_p, color="navy", lw=2.2, label=f"DONKEY (MDD {dd_p.min():.1f}%)")
    plt.plot(dd_s.index, dd_s, color="red", lw=1.3, ls="--", label=f"SPY (MDD {dd_s.min():.1f}%)")
    lines = []
    for name, (s, e) in CRISES.items():
        seg = dd_p.loc[s:e]
        if len(seg) > 5:
            plt.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="darkred", alpha=0.07)
            lines.append(f"{name}: DONKEY {seg.min():.1f}% / SPY {dd_s.loc[s:e].min():.1f}%")
    plt.axhline(0, color="black", lw=1.2)
    plt.title("[레벨3] 언더워터(낙폭) & 역사적 스트레스 테스트", fontsize=16, fontweight="bold")
    plt.ylabel("Drawdown (%)"); plt.grid(True, ls=":", alpha=0.6); plt.legend(loc="lower left")
    if lines:
        plt.gca().text(0.5, 0.05, "\n".join(lines), transform=plt.gca().transAxes,
                       ha="center", va="bottom", fontsize=11, fontweight="bold",
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    plt.tight_layout()
    plt.savefig(config.out("03_port_underwater.png"), dpi=150)
    plt.close()

    # ── (3) 롤링 CAGR (운/타이밍 배제) ──────────────────────────
    rw = min(5, max(1, int(len(pr) / config.TRADING_DAYS) - 1))
    rp = metrics.rolling_cagr(pr, rw)
    plt.figure(figsize=(15, 7))
    for bm, c in zip(["SPY", "QQQ"], ["gray", "crimson"]):
        if bm in rets.columns:
            rb = metrics.rolling_cagr(rets[bm], rw)
            plt.plot(rb.index, rb, color=c, lw=1.5, ls="--", alpha=0.7, label=bm)
    plt.plot(rp.index, rp, color="navy", lw=2.5, label=f"DONKEY ({rw}년 롤링)")
    plt.fill_between(rp.index, rp, 0, where=(rp >= 0), color="royalblue", alpha=0.2)
    plt.fill_between(rp.index, rp, 0, where=(rp < 0), color="red", alpha=0.4)
    plt.axhline(0, color="black", lw=1.5)
    win = (rp > 0).mean() * 100
    plt.title(f"[레벨3] {rw}년 롤링 연평균 수익률 (원금보존 확률 {win:.0f}%)",
              fontsize=16, fontweight="bold")
    plt.ylabel("Rolling CAGR (%)"); plt.grid(True, ls=":", alpha=0.6); plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(config.out("03_port_rolling.png"), dpi=150)
    plt.close()

    # ── (4) 업/다운 캡처 + Hit Ratio (월간, vs SPY) ─────────────
    mret = pd.DataFrame({"DONKEY": pr, **{b: rets[b] for b in bms if b in rets}})
    mret = (1 + mret).resample("ME").prod() - 1
    up, dn = mret["SPY"] > 0, mret["SPY"] < 0
    assets = ["DONKEY"] + [b for b in ["QQQ", "SCHD"] if b in mret.columns]
    uc = [mret.loc[up, a].mean() / mret.loc[up, "SPY"].mean() * 100 for a in assets]
    dc = [mret.loc[dn, a].mean() / mret.loc[dn, "SPY"].mean() * 100 for a in assets]
    hit = [(mret[a] > mret["SPY"]).mean() * 100 for a in assets]

    x = np.arange(len(assets)); w = 0.25
    plt.figure(figsize=(13, 7))
    plt.bar(x - w, uc, w, label="Up Capture (상승장 흡수, ↑좋음)", color="#4CAF50")
    plt.bar(x, dc, w, label="Down Capture (하락장 방어, ↓좋음)", color="#E93B57")
    plt.bar(x + w, hit, w, label="Hit Ratio (SPY 대비 승률)", color="#3b82f6")
    plt.axhline(100, color="gray", ls="--", lw=1.5, label="SPY 기준 100%")
    for i, a in enumerate(assets):
        for off, v in zip([-w, 0, w], [uc[i], dc[i], hit[i]]):
            plt.text(i + off, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, assets, fontweight="bold")
    plt.title("[레벨3] 상승/하락 비대칭성 & 승률 (월간, vs SPY)", fontsize=16, fontweight="bold")
    plt.grid(axis="y", ls=":", alpha=0.6); plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(config.out("03_port_capture.png"), dpi=150)
    plt.close()

    print("→ 저장: 03_port_tearsheet.png, 03_port_underwater.png, "
          "03_port_rolling.png, 03_port_capture.png")


if __name__ == "__main__":
    run()
