"""
run_all.py — 전체 분석 실행 (레벨1 → 레벨2 → 레벨3)

사용법:
    python run_all.py
개별 실행:
    python analysis/a1_individual.py
    python analysis/a2_sleeves.py
    python analysis/a3_portfolio.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))

import config
import a1_individual, a2_sleeves, a3_portfolio

if __name__ == "__main__":
    print("=" * 60)
    print(" DONKEY ETF 백테스트 파이프라인 실행")
    print("=" * 60)
    print("\n■ [레벨1] 개별 29종목")
    a1_individual.run()
    print("\n■ [레벨2] 4개 슬리브(ETF)")
    a2_sleeves.run()
    print("\n■ [레벨3] 전체 포트폴리오")
    a3_portfolio.run()
    print(f"\n완료. 모든 차트는 → {config.OUTPUT_DIR}")
