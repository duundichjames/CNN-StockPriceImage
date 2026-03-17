"""
매매 전략 시뮬레이션 및 성과 평가 스크립트 (9단계)

매매 전략.
  - 20거래일마다 리밸런싱
  - Jiang, Kelly, and Xiu (2023) 방식 long-short 10분위 포트폴리오
  - CNN 상승 확률 기준 종목을 10분위로 정렬하여
    상위 분위(D10) long, 하위 분위(D1) short, 동일가중
  - 비교 기준. 매수보유(buy-and-hold, 전종목 동일가중)

거래 비용.
  - 매수 시 수수료. KRX 0.015%
  - 매도 시 수수료. KRX 0.015% + 제세금 0.20% = 0.215%

성과 지표.
  - 분류. 정확도, AUC
  - 수익률. 연율화 수익률, 연율화 샤프비율, 최대 낙폭(MDD)
  - 시장 구간별 성과 (하락장 + 상승장)

사용법.
    python evaluate_strategy_s9.py

입력.
    results_s9/stage{1-9}_predictions.npz
    data/sample_yahoo/kospi_6000/*.csv
    data/sample_yahoo/kospi200_tickers.csv

출력.
    results_s9/performance_summary.csv
    results_s9/cumulative_returns.csv
    results_s9/market_period_analysis.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
결과경로 = "results_s9"
OHLCV경로 = "data/sample_yahoo/kospi_6000"
# [수정] KOSPI 200 전종목 목록 파일로 변경
종목목록파일 = "data/sample_yahoo/kospi200_tickers.csv"

예측지평 = 20       # 리밸런싱 주기 (거래일)
연거래일 = 252      # 연율화 기준
분위수 = 10          # Jiang et al. (2023) 방식 10분위

# 거래 비용
매수수수료 = 0.00015     # 0.015%
매도수수료 = 0.00015     # 0.015%
매도제세금 = 0.0020      # 0.20%
매도총비용 = 매도수수료 + 매도제세금  # 0.215%

전체단계 = [f"stage{i}" for i in range(1, 10)]
단계설명 = {
    "stage1": "종가선",
    "stage2": "OHLC",
    "stage3": "종가선+거래량",
    "stage4": "OHLC+거래량",
    "stage5": "종가선+MA20",
    "stage6": "OHLC+MA20",
    "stage7": "종가선+MA20+MA60",
    "stage8": "OHLC+거래량+MA20",
    "stage9": "OHLC+거래량+MA20+MA60",
}

# 시장 구간 정의 (하락장 + 상승장)
시장구간 = {
    "코로나19 급락 (2020.01-2020.03)": ("2020-01-01", "2020-03-31"),
    "코로나19 반등 (2020.04-2021.06)": ("2020-04-01", "2021-06-30"),
    "글로벌 긴축 (2022.01-2022.10)": ("2022-01-01", "2022-10-31"),
    "AI 슈퍼사이클 (2024.01-2025.12)": ("2024-01-01", "2025-12-31"),
}


# ══════════════════════════════════════════════════════
# 1. 종목별 일별 수익률 로드
# ══════════════════════════════════════════════════════
def 종목별수익률로드():
    """전체 종목의 일별 종가 수익률을 로드한다."""
    종목df = pd.read_csv(종목목록파일)
    수익률dict = {}
    누락수 = 0
    for _, row in 종목df.iterrows():
        ticker = row["ticker"]
        파일 = os.path.join(OHLCV경로, f"{ticker}.csv")
        # [수정] 데이터 파일 존재 여부 확인
        if not os.path.exists(파일):
            누락수 += 1
            continue
        df = pd.read_csv(파일, parse_dates=["Date"])
        df = df[df["Volume"] > 0].sort_values("Date").reset_index(drop=True)
        df["수익률"] = df["Close"].pct_change()
        수익률dict[ticker] = df[["Date", "Close", "수익률"]].copy()
    if 누락수 > 0:
        print(f"  데이터 파일 누락. {누락수}개 종목 건너뜀")
    return 수익률dict


# ══════════════════════════════════════════════════════
# 2. 분류 성과 지표
# ══════════════════════════════════════════════════════
def 분류성과(stage):
    """정확도와 AUC를 산출한다."""
    파일 = os.path.join(결과경로, f"{stage}_predictions.npz")
    data = np.load(파일, allow_pickle=True)

    확률 = data["probabilities"]
    예측 = data["predictions"]
    레이블 = data["labels"]

    정확도 = (예측 == 레이블).mean()

    if len(np.unique(레이블)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(레이블, 확률)

    return 정확도, auc


# ══════════════════════════════════════════════════════
# 3. 매매 전략 시뮬레이션
# ══════════════════════════════════════════════════════
def 매매전략시뮬레이션(stage, 수익률dict):
    """
    Jiang, Kelly, and Xiu (2023) 방식 long-short 10분위 포트폴리오.

    전략.
      매 리밸런싱 시점에서 CNN 상승 확률 기준으로 종목을 10분위로 정렬한다.
      상위 분위(D10)를 동일가중 매수(long), 하위 분위(D1)를 동일가중 매도(short)한다.
      전략 수익률은 H-L spread, 즉 long 수익률 - short 수익률이다.

    거래 비용.
      [수정] long/short 각 레그에서 포지션 변경 시에만 비용 발생.
      long 진입(매수). 매수수수료
      long 청산(매도). 매도총비용
      short 진입(공매도). 매도총비용
      short 청산(환매). 매수수수료
    """
    파일 = os.path.join(결과경로, f"{stage}_predictions.npz")
    data = np.load(파일, allow_pickle=True)

    확률 = data["probabilities"]
    날짜 = pd.to_datetime(data["dates"].astype(str))
    티커 = data["tickers"].astype(str)

    예측df = pd.DataFrame({
        "date": 날짜, "ticker": 티커, "prob": 확률
    })

    전체날짜 = sorted(예측df["date"].unique())
    리밸런싱시점 = 전체날짜[::예측지평]

    일별기록 = []
    # [수정] 이전 리밸런싱의 long/short 종목 집합 (포지션 변경 판별용)
    이전long종목 = set()
    이전short종목 = set()

    for idx in range(len(리밸런싱시점)):
        시점 = 리밸런싱시점[idx]
        다음시점 = 리밸런싱시점[idx + 1] if idx + 1 < len(리밸런싱시점) else None

        시점예측 = 예측df[예측df["date"] == 시점].copy()

        # ── 10분위 정렬 후 상위/하위 분위 선정 ──────────
        # Jiang et al. (2023) Table I 방식. 확률 기준 정렬, 동일가중
        시점예측 = 시점예측.sort_values("prob")
        종목수 = len(시점예측)
        분위크기 = 종목수 // 분위수  # 200종목 / 10분위 ≈ 20종목씩

        # D1(하위 분위) = short 레그, D10(상위 분위) = long 레그
        short종목 = set(시점예측.iloc[:분위크기]["ticker"].values)
        long종목 = set(시점예측.iloc[-분위크기:]["ticker"].values)

        # ── [수정] 포지션 변경에 따른 거래 비용 산출 ──────────
        # long 레그. 신규 진입 종목은 매수수수료, 청산 종목은 매도총비용
        long신규 = long종목 - 이전long종목
        long청산 = 이전long종목 - long종목
        # short 레그. 신규 진입 종목은 매도총비용, 청산 종목은 매수수수료
        short신규 = short종목 - 이전short종목
        short청산 = 이전short종목 - short종목

        리밸런싱비용 = (
            len(long신규) * 매수수수료 / 분위크기
            + len(long청산) * 매도총비용 / 분위크기
            + len(short신규) * 매도총비용 / 분위크기
            + len(short청산) * 매수수수료 / 분위크기
        ) if 분위크기 > 0 else 0.0

        # ── 보유기간 일별 수익률 산출 ────────────────────────
        if 다음시점 is not None:
            보유기간날짜 = [d for d in 전체날짜
                       if d >= 시점 and d < 다음시점]
        else:
            보유기간날짜 = [d for d in 전체날짜 if d >= 시점]

        for 일idx, 날 in enumerate(보유기간날짜):
            # [수정] long 레그 일별 수익률 (동일가중 평균)
            long수익률목록 = []
            for ticker in long종목:
                if ticker in 수익률dict:
                    종목df = 수익률dict[ticker]
                    행 = 종목df[종목df["Date"] == 날]
                    if len(행) > 0 and not pd.isna(행.iloc[0]["수익률"]):
                        long수익률목록.append(행.iloc[0]["수익률"])

            # [수정] short 레그 일별 수익률 (동일가중 평균)
            short수익률목록 = []
            for ticker in short종목:
                if ticker in 수익률dict:
                    종목df = 수익률dict[ticker]
                    행 = 종목df[종목df["Date"] == 날]
                    if len(행) > 0 and not pd.isna(행.iloc[0]["수익률"]):
                        short수익률목록.append(행.iloc[0]["수익률"])

            # BH 수익률 (전종목 동일가중)
            BH수익률목록 = []
            for ticker in 시점예측["ticker"].unique():
                if ticker in 수익률dict:
                    종목df = 수익률dict[ticker]
                    행 = 종목df[종목df["Date"] == 날]
                    if len(행) > 0 and not pd.isna(행.iloc[0]["수익률"]):
                        BH수익률목록.append(행.iloc[0]["수익률"])

            if len(long수익률목록) == 0 or len(short수익률목록) == 0:
                continue

            # [수정] H-L spread. long 평균 수익률 - short 평균 수익률
            long평균 = np.mean(long수익률목록)
            short평균 = np.mean(short수익률목록)
            전략수익률 = long평균 - short평균

            BH평균 = np.mean(BH수익률목록) if len(BH수익률목록) > 0 else 0.0

            일별기록.append({
                "date": 날,
                "전략수익률": 전략수익률,
                "BH수익률": BH평균,
                "long수익률": long평균,
                "short수익률": short평균,
                "long종목수": len(long수익률목록),
                "short종목수": len(short수익률목록),
                # [수정] 리밸런싱 첫날에만 거래 비용 부과
                "거래비용": 리밸런싱비용 if 일idx == 0 else 0.0,
                "CNN확률평균": 시점예측["prob"].mean(),
            })

        # [수정] 다음 리밸런싱을 위해 현재 포지션 저장
        이전long종목 = long종목
        이전short종목 = short종목

    기록df = pd.DataFrame(일별기록)
    if len(기록df) == 0:
        return None

    # ── [수정] 거래 비용 반영 및 누적 수익률 ──────────────────
    기록df["전략수익률_비용후"] = 기록df["전략수익률"] - 기록df["거래비용"]
    기록df["전략누적"] = (1 + 기록df["전략수익률_비용후"]).cumprod()
    기록df["BH누적"] = (1 + 기록df["BH수익률"]).cumprod()

    # [수정] 하위 호환성을 위해 포지션 및 포지션변경 열 추가
    기록df["포지션"] = "long-short"
    기록df["포지션변경"] = 기록df["거래비용"] > 0

    return 기록df


# ══════════════════════════════════════════════════════
# 4. 성과 지표 산출
# ══════════════════════════════════════════════════════
def 성과지표산출(기록df):
    """수익률 기반 성과 지표를 산출한다."""
    if 기록df is None or len(기록df) == 0:
        return {}

    전략수익률 = 기록df["전략수익률_비용후"]
    BH수익률 = 기록df["BH수익률"]
    n일 = len(기록df)
    n년 = n일 / 연거래일

    전략총수익 = 기록df["전략누적"].iloc[-1]
    BH총수익 = 기록df["BH누적"].iloc[-1]
    전략연율수익률 = 전략총수익 ** (1 / n년) - 1
    BH연율수익률 = BH총수익 ** (1 / n년) - 1

    전략샤프 = (전략수익률.mean() / 전략수익률.std() * np.sqrt(연거래일)
               if 전략수익률.std() > 0 else 0)
    BH샤프 = (BH수익률.mean() / BH수익률.std() * np.sqrt(연거래일)
             if BH수익률.std() > 0 else 0)

    전략고점 = 기록df["전략누적"].cummax()
    전략낙폭 = (기록df["전략누적"] - 전략고점) / 전략고점
    전략MDD = 전략낙폭.min()

    BH고점 = 기록df["BH누적"].cummax()
    BH낙폭 = (기록df["BH누적"] - BH고점) / BH고점
    BH_MDD = BH낙폭.min()

    # [수정] long-short 전략은 항상 포지션을 유지하므로 매수비율 대신 리밸런싱 횟수 보고
    리밸런싱횟수 = 기록df["포지션변경"].sum()

    return {
        "전략연율수익률": 전략연율수익률,
        "BH연율수익률": BH연율수익률,
        "전략샤프": 전략샤프,
        "BH샤프": BH샤프,
        "전략MDD": 전략MDD,
        "BH_MDD": BH_MDD,
        "리밸런싱횟수": int(리밸런싱횟수),
        "총거래일": n일,
    }


# ══════════════════════════════════════════════════════
# 5. 시장 구간별 성과
# ══════════════════════════════════════════════════════
def 시장구간성과(기록df):
    """사전 정의된 시장 구간(하락장 + 상승장)의 누적 수익률을 산출한다."""
    if 기록df is None or len(기록df) == 0:
        return {}

    결과 = {}
    for 구간명, (시작, 종료) in 시장구간.items():
        마스크 = (기록df["date"] >= 시작) & (기록df["date"] <= 종료)
        구간df = 기록df[마스크]
        if len(구간df) == 0:
            continue
        전략구간수익 = (1 + 구간df["전략수익률_비용후"]).prod() - 1
        BH구간수익 = (1 + 구간df["BH수익률"]).prod() - 1
        결과[구간명] = {
            "전략수익률": 전략구간수익,
            "BH수익률": BH구간수익,
            "초과수익률": 전략구간수익 - BH구간수익,
            "거래일수": len(구간df),
        }
    return 결과


# ══════════════════════════════════════════════════════
# 6. 메인
# ══════════════════════════════════════════════════════
def 메인():
    print("종목별 수익률 로드 중...")
    수익률dict = 종목별수익률로드()
    print(f"  {len(수익률dict)}개 종목 로드 완료")
    print()

    요약기록 = []
    시장구간기록 = []
    누적수익률저장 = {}

    for stage in 전체단계:
        print(f"{'='*60}")
        print(f"{stage} ({단계설명[stage]}) 평가")
        print(f"{'='*60}")

        # ── 분류 성과 ────────────────────────────────
        정확도, auc = 분류성과(stage)
        print(f"  정확도. {정확도:.4f}")
        print(f"  AUC.    {auc:.4f}")

        # ── 매매 전략 시뮬레이션 ──────────────────────
        기록df = 매매전략시뮬레이션(stage, 수익률dict)
        성과 = 성과지표산출(기록df)

        if 성과:
            # [수정] long-short H-L spread 수익률 표시
            print(f"  H-L 연율수익률. {성과['전략연율수익률']:.2%}")
            print(f"  BH 연율수익률.   {성과['BH연율수익률']:.2%}")
            print(f"  H-L 샤프비율.   {성과['전략샤프']:.3f}")
            print(f"  BH 샤프비율.     {성과['BH샤프']:.3f}")
            print(f"  H-L MDD.       {성과['전략MDD']:.2%}")
            print(f"  BH MDD.         {성과['BH_MDD']:.2%}")
            print(f"  리밸런싱 횟수.   {성과['리밸런싱횟수']}회")

        # ── 시장 구간별 성과 ──────────────────────────
        구간결과 = 시장구간성과(기록df)
        for 구간명, 구간성과 in 구간결과.items():
            # [수정] H-L spread 표시
            print(f"  {구간명}. "
                  f"H-L {구간성과['전략수익률']:.2%}, "
                  f"BH {구간성과['BH수익률']:.2%}, "
                  f"초과 {구간성과['초과수익률']:.2%}")
            시장구간기록.append({
                "stage": stage,
                "설명": 단계설명[stage],
                "구간": 구간명,
                **구간성과,
            })

        # ── 기록 저장 ────────────────────────────────
        요약기록.append({
            "stage": stage,
            "설명": 단계설명[stage],
            "정확도": 정확도,
            "AUC": auc,
            **성과,
        })

        if 기록df is not None:
            누적수익률저장[stage] = 기록df[
                ["date", "전략누적", "BH누적", "포지션"]
            ].copy()
            누적수익률저장[stage]["stage"] = stage

        print()

    # ── 결과 저장 ────────────────────────────────────
    요약df = pd.DataFrame(요약기록)
    요약df.to_csv(os.path.join(결과경로, "performance_summary.csv"), index=False)
    print(f"성과 요약 저장. {결과경로}/performance_summary.csv")

    if 누적수익률저장:
        누적df = pd.concat(누적수익률저장.values(), ignore_index=True)
        누적df.to_csv(os.path.join(결과경로, "cumulative_returns.csv"), index=False)
        print(f"누적 수익률 저장. {결과경로}/cumulative_returns.csv")

    if 시장구간기록:
        시장구간df = pd.DataFrame(시장구간기록)
        시장구간df.to_csv(os.path.join(결과경로, "market_period_analysis.csv"), index=False)
        print(f"시장 구간 분석 저장. {결과경로}/market_period_analysis.csv")

    # ── 최종 요약 표 ─────────────────────────────────
    print(f"\n{'='*80}")
    print("최종 성과 요약")
    print(f"{'='*80}")
    print(f"  {'단계':<8} {'설명':<20} {'정확도':>7} {'AUC':>7} "
          f"{'H-L수익':>8} {'BH수익':>8} {'H-L샤프':>8} {'BH샤프':>8} "
          f"{'H-L MDD':>8} {'BH MDD':>8}")
    print(f"  {'-'*76}")
    for row in 요약기록:
        print(
            f"  {row['stage']:<8} {row['설명']:<20} "
            f"{row['정확도']:>7.4f} {row['AUC']:>7.4f} "
            f"{row.get('전략연율수익률', 0):>7.2%} "
            f"{row.get('BH연율수익률', 0):>7.2%} "
            f"{row.get('전략샤프', 0):>8.3f} "
            f"{row.get('BH샤프', 0):>8.3f} "
            f"{row.get('전략MDD', 0):>7.2%} "
            f"{row.get('BH_MDD', 0):>7.2%}"
        )
    print(f"{'='*80}")


if __name__ == "__main__":
    메인()