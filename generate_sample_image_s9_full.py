"""
9단계 샘플 이미지 생성 (논문 그림용)
KOSPI 200 종목 중 임의 1개를 추출하여 3×3 패널 그림을 생성한다.
"""
import matplotlib.pyplot as plt
import json
import numpy as np

from generate_images_s9_ma_adjusted import (
    종목로드, 데이터로드, 단일이미지생성, 가격범위산출,
    윈도우, 전체단계
)

plt.rcParams['font.family'] = ['WooriBatang', 'DejaVu Sans']

# ── 시드 설정 및 임의 종목 추출 ─────────────────────
np.random.seed(42)

종목df = 종목로드()
선택행 = np.random.randint(0, len(종목df))
ticker = 종목df.iloc[선택행]["ticker"]
name   = 종목df.iloc[선택행]["name"]
print(f"\n테스트 종목. {ticker} {name} (행 {선택행})")

df, 제거수 = 데이터로드(ticker)
print(f"총 거래일. {len(df)}, 거래량 0 제거. {제거수}일")
print(f"기간. {df['Date'].iloc[0].date()} - {df['Date'].iloc[-1].date()}")

# ── t 선택. MA60이 완전한 구간에서 임의 1개 ──────────
유효시작 = 윈도우 + 59
유효범위 = range(유효시작, len(df) - 5)
t = np.random.choice(유효범위)

윈도우시작 = t - 윈도우
윈도우df = df.iloc[윈도우시작:t].copy()
MA20값 = df.iloc[윈도우시작:t]["MA20"].values
MA60값 = df.iloc[윈도우시작:t]["MA60"].values
레이블 = int(df.iloc[t]["레이블"])
날짜 = df.iloc[t]["Date"].date()

# ── 9단계 이미지 생성 ───────────────────────────────
이미지목록 = []
for stage in 전체단계:
    img = 단일이미지생성(stage, 윈도우df, MA20값, MA60값)
    이미지목록.append(img)

# ── 3행 3열 배치 ────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5),
                         gridspec_kw={"wspace": 0.08, "hspace": 0.3})
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)

제목 = [
    "1: 종가선",         "2: OHLC",            "3: 종가선+거래량",
    "4: OHLC+거래량",     "5: 종가선+MA20",      "6: OHLC+MA20",
    "7: 종가선+MA20+MA60", "8: OHLC+거래량+MA20", "9: OHLC+거래량+MA20+MA60",
]

for ax, img, title in zip(axes.flat, 이미지목록, 제목):
    ax.imshow(img, cmap="gray", aspect="equal")
    ax.set_title(title, fontsize=14, pad=4)
    ax.axis("off")

fig.savefig("figs/이미지예시_s9.png", dpi=600, bbox_inches="tight")
plt.show()

print(f"  t={t}, 날짜={날짜}, 레이블={레이블}")
print(f"  윈도우 기간. {윈도우df['Date'].iloc[0].date()} - {윈도우df['Date'].iloc[-1].date()}")
print(f"  가격 범위. {윈도우df['Low'].min():,.0f} - {윈도우df['High'].max():,.0f}")
print(f"  MA20 NaN 수. {sum(np.isnan(MA20값))}/{len(MA20값)}")
print(f"  MA60 NaN 수. {sum(np.isnan(MA60값))}/{len(MA60값)}")

# ── 캡션 메타데이터 저장 ────────────────────────────
캡션정보 = {
    "종목명": name,
    "종목코드": ticker.replace(".KS", ""),
    "윈도우시작": str(윈도우df['Date'].iloc[0].date()),
    "윈도우종료": str(윈도우df['Date'].iloc[-1].date())
}
with open("figs/이미지예시_s9_meta.json", "w", encoding="utf-8") as f:
    json.dump(캡션정보, f, ensure_ascii=False)