"""
CNN 학습용 차트 이미지 생성 스크립트 (9단계)

9단계 이미지 (서술 순서에 따라 번호 배정).
  기본.
    stage1. 종가선
    stage2. OHLC
  거래량 추가.
    stage3. 종가선 + 거래량           ← 거래량의 순수 효과 분리
    stage4. OHLC + 거래량
  단일 이동평균.
    stage5. 종가선 + MA20
    stage6. OHLC + MA20
  이중 이동평균.
    stage7. 종가선 + MA20 + MA60
  복합.
    stage8. OHLC + 거래량 + MA20
    stage9. OHLC + 거래량 + MA20 + MA60

사용법.
    python generate_images_s9.py

입력.
    data/sample_yahoo/sample10_tickers.csv
    data/sample_yahoo/kospi_6000/*.csv

출력.
    data/images_s9/stage{1-9}_train.npz, stage{1-9}_test.npz
    data/images_s9/image_meta.csv
"""

import os
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
기본경로 = "data/sample_yahoo"
OHLCV경로 = os.path.join(기본경로, "kospi_6000")
# [수정] KOSPI 200 전종목 목록 파일로 변경
종목목록파일 = os.path.join(기본경로, "kospi200_tickers.csv")
출력경로 = "data/images_s9"
os.makedirs(출력경로, exist_ok=True)

윈도우 = 20
예측지평 = 20
이미지너비 = 윈도우 * 3   # 60
이미지높이 = 64

가격높이 = 51   # 거래량 포함 시 상단 4/5
거래량높이 = 13  # 거래량 포함 시 하단 1/5

# [수정] 분할일 하드코딩. 기존에는 분할일.json에서 로드하였으나,
# 실행 환경 의존성을 제거한다.
분할일 = pd.Timestamp("2018-12-31")

# [수정] 최소 데이터 수. MA60(60) + 윈도우(20) + 예측지평(20) = 100이
# 이론적 최소이나, 여유를 두어 120으로 설정한다.
최소데이터수 = 120

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


# ══════════════════════════════════════════════════════
# 1. 파이프라인에서 생성한 파일 로드
# ══════════════════════════════════════════════════════
def 종목로드():
    if not os.path.exists(종목목록파일):
        print(f"오류. {종목목록파일} 파일이 없습니다.")
        raise SystemExit(1)
    df = pd.read_csv(종목목록파일)
    print(f"종목 목록 로드. {len(df)}개 종목")
    for _, row in df.iterrows():
        print(f"  {row['ticker']}  {row['name']}")
    return df


# ══════════════════════════════════════════════════════
# 2. 데이터 로드 및 전처리
# ══════════════════════════════════════════════════════
def 데이터로드(ticker):
    파일 = os.path.join(OHLCV경로, f"{ticker}.csv")
    df = pd.read_csv(파일, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    원래행수 = len(df)
    df = df[df["Volume"] > 0].reset_index(drop=True)
    제거행수 = 원래행수 - len(df)

    df["MA20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["MA60"] = df["Close"].rolling(window=60, min_periods=60).mean()

    df["미래종가"] = df["Close"].shift(-예측지평)
    df["레이블"] = (df["미래종가"] / df["Close"] - 1 > 0).astype(int)

    return df, 제거행수


# ══════════════════════════════════════════════════════
# 3. 픽셀 좌표 변환 유틸리티
# ══════════════════════════════════════════════════════
def 가격을행으로(가격, 최저, 최고, 높이):
    if 최고 == 최저:
        return 높이 // 2
    비율 = (가격 - 최저) / (최고 - 최저)
    행 = int(round((1 - 비율) * (높이 - 1)))
    return max(0, min(높이 - 1, 행))


def 수직선그리기(img, 열, 행시작, 행끝):
    r0 = max(0, min(행시작, 행끝))
    r1 = min(img.shape[0] - 1, max(행시작, 행끝))
    img[r0:r1+1, 열] = 255


def 브레젠햄선(img, 열1, 행1, 열2, 행2):
    dx = abs(열2 - 열1)
    dy = abs(행2 - 행1)
    sx = 1 if 열1 < 열2 else -1
    sy = 1 if 행1 < 행2 else -1
    err = dx - dy
    while True:
        if 0 <= 행1 < img.shape[0] and 0 <= 열1 < img.shape[1]:
            img[행1, 열1] = 255
        if 열1 == 열2 and 행1 == 행2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            열1 += sx
        if e2 < dx:
            err += dx
            행1 += sy


# ══════════════════════════════════════════════════════
# 4. 이미지 구성 요소 함수
# ══════════════════════════════════════════════════════
def 종가선그리기(img, 윈도우df, 최저, 최고, 높이):
    이전열, 이전행 = None, None
    for d in range(윈도우):
        열 = 3 * d + 1
        행 = 가격을행으로(윈도우df.iloc[d]["Close"], 최저, 최고, 높이)
        img[행, 열] = 255
        if 이전열 is not None:
            브레젠햄선(img, 이전열, 이전행, 열, 행)
        이전열, 이전행 = 열, 행


def OHLC바그리기(img, 윈도우df, 최저, 최고, 높이):
    for d in range(윈도우):
        o = 윈도우df.iloc[d]["Open"]
        h = 윈도우df.iloc[d]["High"]
        l = 윈도우df.iloc[d]["Low"]
        c = 윈도우df.iloc[d]["Close"]
        좌열 = 3 * d
        중열 = 3 * d + 1
        우열 = 3 * d + 2
        행고 = 가격을행으로(h, 최저, 최고, 높이)
        행저 = 가격을행으로(l, 최저, 최고, 높이)
        행시 = 가격을행으로(o, 최저, 최고, 높이)
        행종 = 가격을행으로(c, 최저, 최고, 높이)
        수직선그리기(img, 중열, 행고, 행저)
        img[행시, 좌열] = 255
        img[행종, 우열] = 255


def 거래량바그리기(img, 윈도우df):
    최대거래량 = 윈도우df["Volume"].max()
    if 최대거래량 <= 0:
        return
    for d in range(윈도우):
        v = 윈도우df.iloc[d]["Volume"]
        if v <= 0:
            continue
        중열 = 3 * d + 1
        바높이 = max(1, int(round((v / 최대거래량) * (거래량높이 - 1))))
        바하단 = 이미지높이 - 1
        바상단 = 바하단 - 바높이 + 1
        수직선그리기(img, 중열, 바상단, 바하단)


def MA선그리기(img, MA값, 최저, 최고, 높이):
    이전열, 이전행 = None, None
    for d in range(윈도우):
        ma = MA값[d]
        if np.isnan(ma):
            이전열, 이전행 = None, None
            continue
        열 = 3 * d + 1
        행 = 가격을행으로(ma, 최저, 최고, 높이)
        if 이전열 is not None:
            브레젠햄선(img, 이전열, 이전행, 열, 행)
        img[행, 열] = 255
        이전열, 이전행 = 열, 행


# ══════════════════════════════════════════════════════
# 5. 9단계 이미지 생성
# ══════════════════════════════════════════════════════
def 가격범위산출(윈도우df, MA20값, MA60값, stage):
    """단계별로 적절한 가격 범위를 산출한다."""
    OHLC최저 = 윈도우df["Low"].min()
    OHLC최고 = 윈도우df["High"].max()

    # stage 1-4. OHLC 범위만 사용
    if stage in ("stage1", "stage2", "stage3", "stage4"):
        return OHLC최저, OHLC최고

    MA20유효 = MA20값[~np.isnan(MA20값)]
    MA60유효 = MA60값[~np.isnan(MA60값)]

    # stage 5, 6, 8. OHLC + MA20
    if stage in ("stage5", "stage6", "stage8"):
        if len(MA20유효) > 0:
            return min(OHLC최저, MA20유효.min()), max(OHLC최고, MA20유효.max())
        return OHLC최저, OHLC최고

    # stage 7, 9. OHLC + MA20 + MA60
    모든MA = np.concatenate([MA20유효, MA60유효])
    if len(모든MA) > 0:
        return min(OHLC최저, 모든MA.min()), max(OHLC최고, 모든MA.max())
    return OHLC최저, OHLC최고


def 단일이미지생성(stage, 윈도우df, MA20값, MA60값):
    """한 윈도우에 대해 지정된 단계의 이미지 1개를 생성한다."""
    최저, 최고 = 가격범위산출(윈도우df, MA20값, MA60값, stage)
    img = np.zeros((이미지높이, 이미지너비), dtype=np.uint8)

    if stage == "stage1":
        종가선그리기(img, 윈도우df, 최저, 최고, 이미지높이)

    elif stage == "stage2":
        OHLC바그리기(img, 윈도우df, 최저, 최고, 이미지높이)

    elif stage == "stage3":
        종가선그리기(img, 윈도우df, 최저, 최고, 가격높이)
        거래량바그리기(img, 윈도우df)

    elif stage == "stage4":
        OHLC바그리기(img, 윈도우df, 최저, 최고, 가격높이)
        거래량바그리기(img, 윈도우df)

    elif stage == "stage5":
        종가선그리기(img, 윈도우df, 최저, 최고, 이미지높이)
        MA선그리기(img, MA20값, 최저, 최고, 이미지높이)

    elif stage == "stage6":
        OHLC바그리기(img, 윈도우df, 최저, 최고, 이미지높이)
        MA선그리기(img, MA20값, 최저, 최고, 이미지높이)

    elif stage == "stage7":
        종가선그리기(img, 윈도우df, 최저, 최고, 이미지높이)
        MA선그리기(img, MA20값, 최저, 최고, 이미지높이)
        MA선그리기(img, MA60값, 최저, 최고, 이미지높이)

    elif stage == "stage8":
        OHLC바그리기(img, 윈도우df, 최저, 최고, 가격높이)
        거래량바그리기(img, 윈도우df)
        MA선그리기(img, MA20값, 최저, 최고, 가격높이)

    elif stage == "stage9":
        OHLC바그리기(img, 윈도우df, 최저, 최고, 가격높이)
        거래량바그리기(img, 윈도우df)
        MA선그리기(img, MA20값, 최저, 최고, 가격높이)
        MA선그리기(img, MA60값, 최저, 최고, 가격높이)

    return img


# ══════════════════════════════════════════════════════
# 6. 종목별 데이터 전처리 (1회만 수행)
# ══════════════════════════════════════════════════════
def 전체종목전처리(종목df):
    """전 종목의 OHLCV + MA + 레이블을 미리 로드한다.
    이미지 픽셀 배열이 아니라 가격 DataFrame만 보관하므로 메모리 부담이 작다."""
    종목데이터 = {}
    for _, row in 종목df.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        파일 = os.path.join(OHLCV경로, f"{ticker}.csv")
        if not os.path.exists(파일):
            print(f"  {ticker} {name}. 데이터 파일 없음, 건너뜀")
            continue
        df, 제거수 = 데이터로드(ticker)
        if len(df) < 최소데이터수:
            print(f"  {ticker} {name}. 데이터 {len(df)}행 < {최소데이터수}행, 건너뜀")
            continue
        종목데이터[ticker] = df
    print(f"  전처리 완료. {len(종목데이터)}개 종목 유효\n")
    return 종목데이터


# ══════════════════════════════════════════════════════
# 7. 메인 (단계별 순차 처리)
# ══════════════════════════════════════════════════════
def 메인():
    종목df = 종목로드()
    print(f"분할일. {분할일.date()}")
    print(f"  훈련. ~{분할일.year}년 말")
    print(f"  테스트. {분할일.year + 1}년 초~\n")

    # ── [수정] 종목 데이터를 1회만 로드 ──────────────────────
    print("종목 데이터 전처리 중...")
    종목데이터 = 전체종목전처리(종목df)
    유효종목 = list(종목데이터.keys())

    메타기록 = []

    # ── [수정] 단계별 순차 처리. 한 단계 완료 후 저장하고 메모리 해제 ──
    # 9단계를 동시에 메모리에 유지하면 200종목 × 120만 이미지 × 9 ≈ 41GB.
    # 단계별로 처리하면 한 번에 약 4.5GB만 필요하다.
    for stage in 전체단계:
        print(f"{'='*60}")
        print(f"{stage} ({단계설명[stage]}) 이미지 생성")
        print(f"{'='*60}")

        images = []
        labels = []
        dates = []
        tickers = []

        for ticker in 유효종목:
            df = 종목데이터[ticker]
            총일수 = len(df)
            n_before = len(images)

            for t in range(윈도우, 총일수 - 예측지평):
                윈도우시작 = t - 윈도우
                윈도우df = df.iloc[윈도우시작:t]

                if len(윈도우df) != 윈도우:
                    continue
                if pd.isna(df.iloc[t]["레이블"]):
                    continue

                MA20값 = df.iloc[윈도우시작:t]["MA20"].values
                MA60값 = df.iloc[윈도우시작:t]["MA60"].values

                img = 단일이미지생성(stage, 윈도우df, MA20값, MA60값)
                images.append(img)
                labels.append(int(df.iloc[t]["레이블"]))
                dates.append(df.iloc[t]["Date"])
                tickers.append(ticker)

            n_생성 = len(images) - n_before
            if n_생성 > 0:
                print(f"  {ticker}. {n_생성:,}개")

        # ── 훈련/테스트 분할 및 저장 ──────────────────────
        images_arr = np.array(images, dtype=np.uint8)
        labels_arr = np.array(labels, dtype=np.int8)
        dates_arr = np.array(dates, dtype="datetime64[D]")
        tickers_arr = np.array(tickers)

        훈련마스크 = dates_arr <= np.datetime64(분할일)
        테스트마스크 = ~훈련마스크

        훈련파일 = os.path.join(출력경로, f"{stage}_train.npz")
        테스트파일 = os.path.join(출력경로, f"{stage}_test.npz")

        np.savez_compressed(훈련파일,
                            images=images_arr[훈련마스크],
                            labels=labels_arr[훈련마스크],
                            dates=dates_arr[훈련마스크],
                            tickers=tickers_arr[훈련마스크])
        np.savez_compressed(테스트파일,
                            images=images_arr[테스트마스크],
                            labels=labels_arr[테스트마스크],
                            dates=dates_arr[테스트마스크],
                            tickers=tickers_arr[테스트마스크])

        n_train = int(훈련마스크.sum())
        n_test = int(테스트마스크.sum())
        print(f"  → 저장 완료. 훈련 {n_train:,}개, 테스트 {n_test:,}개\n")

        메타기록.append({
            "stage": stage, "description": 단계설명[stage],
            "n_train": n_train, "n_test": n_test,
            "n_total": n_train + n_test
        })

        # [수정] 메모리 해제
        del images, labels, dates, tickers
        del images_arr, labels_arr, dates_arr, tickers_arr

    # ── 메타 정보 저장 ──────────────────────────────────
    메타df = pd.DataFrame(메타기록)
    메타df["split_date"] = str(분할일.date())
    메타df["n_tickers"] = len(유효종목)
    메타df["window"] = 윈도우
    메타df["horizon"] = 예측지평
    메타df["img_height"] = 이미지높이
    메타df["img_width"] = 이미지너비
    메타df.to_csv(os.path.join(출력경로, "image_meta.csv"), index=False)

    print(f"이미지 생성 완료. {출력경로}/")


if __name__ == "__main__":
    메인()