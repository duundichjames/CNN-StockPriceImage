"""
Grad-CAM MA20 주목도 분석 스크립트

Stage 4(OHLC+거래량)와 Stage 8(OHLC+거래량+MA20)의 이미지 차분으로
MA20 픽셀 마스크를 생성하고, Stage 8의 Grad-CAM 히트맵에서
MA20 영역과 비MA20 영역의 주목도를 비교한다.

실행 방법.
  python gradcam_ma20_attention_full.py

입력.
  data/images_s9/stage4_test.npz
  data/images_s9/stage8_test.npz
  results_s9/stage8_predictions.npz
  models_s9/stage8_run{1-5}.pt

출력.
  results_gradcam/ma20_attention.csv
  results_gradcam/ma20_attention_summary.png
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
모형경로 = "models_s9"
이미지경로 = "data/images_s9"
예측경로 = "results_s9"
출력경로 = "results_gradcam"
os.makedirs(출력경로, exist_ok=True)

이미지높이 = 64
이미지너비 = 60
가격높이 = 51
거래량높이 = 13
독립훈련횟수 = 5
최대개수 = 500

if torch.backends.mps.is_available():
    디바이스 = torch.device("mps")
elif torch.cuda.is_available():
    디바이스 = torch.device("cuda")
else:
    디바이스 = torch.device("cpu")


# ══════════════════════════════════════════════════════
# 1. CNN 아키텍처
# ══════════════════════════════════════════════════════
class ChartCNN(nn.Module):
    def __init__(self, 입력높이=64, 입력너비=60):
        super().__init__()
        self.블록1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.01), nn.MaxPool2d(2, 2))
        self.블록2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.01), nn.MaxPool2d(2, 2))
        self.블록3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.01), nn.MaxPool2d(2, 2))
        flat = 128 * (입력높이 // 8) * (입력너비 // 8)
        self.분류기 = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 128),
            nn.LeakyReLU(0.01), nn.Dropout(0.5), nn.Linear(128, 2))

    def forward(self, x):
        return self.분류기(self.블록3(self.블록2(self.블록1(x))))


# ══════════════════════════════════════════════════════
# 2. Grad-CAM
# ══════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, 모형):
        self.모형 = 모형
        self.모형.eval()
        self.특징맵 = None
        self.그래디언트 = None
        self.모형.블록3.register_forward_hook(self._특징맵저장)
        self.모형.블록3.register_full_backward_hook(self._그래디언트저장)

    def _특징맵저장(self, module, input, output):
        self.특징맵 = output.detach()

    def _그래디언트저장(self, module, grad_input, grad_output):
        self.그래디언트 = grad_output[0].detach()

    def __call__(self, 입력, 타겟클래스=None):
        입력.requires_grad_(True)
        출력 = self.모형(입력)
        if 타겟클래스 is None:
            타겟클래스 = 출력.argmax(dim=1).item()
        self.모형.zero_grad()
        출력[0, 타겟클래스].backward()
        가중치 = self.그래디언트.mean(dim=[2, 3], keepdim=True)
        cam = (가중치 * self.특징맵).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=(이미지높이, 이미지너비), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


# ══════════════════════════════════════════════════════
# 3. 데이터 로드 및 MA20 마스크 생성
# ══════════════════════════════════════════════════════
def 데이터로드():
    print("데이터 로드 중...")

    # stage8 이미지 및 예측
    s8 = np.load(os.path.join(이미지경로, "stage8_test.npz"), allow_pickle=True)
    이미지8 = s8["images"].astype(np.float32)
    레이블 = s8["labels"].astype(np.int64)
    날짜 = pd.to_datetime(s8["dates"].astype(str))

    예측 = np.load(os.path.join(예측경로, "stage8_predictions.npz"), allow_pickle=True)
    확률 = 예측["probabilities"]
    예측레이블 = 예측["predictions"]

    # stage4 이미지 (MA20 마스크 생성용)
    s4 = np.load(os.path.join(이미지경로, "stage4_test.npz"), allow_pickle=True)
    이미지4 = s4["images"].astype(np.float32)

    print(f"  stage8 테스트 이미지. {len(이미지8):,}개")
    print(f"  stage4 테스트 이미지. {len(이미지4):,}개")

    # ── MA20 마스크 생성 ──
    # 가격 영역(상단 51행)에서 stage8에만 존재하는 픽셀 = MA20 선
    # stage8 > 0 이고 stage4 == 0 인 픽셀
    마스크 = np.zeros((len(이미지8), 이미지높이, 이미지너비), dtype=bool)
    가격영역마스크 = np.zeros_like(마스크)

    일치수 = min(len(이미지8), len(이미지4))
    for i in range(일치수):
        # 가격 영역에서만 차분
        s8가격 = 이미지8[i, :가격높이, :]
        s4가격 = 이미지4[i, :가격높이, :]
        ma20픽셀 = (s8가격 > 0) & (s4가격 == 0)
        마스크[i, :가격높이, :] = ma20픽셀

    ma20비율 = 마스크[:일치수].sum() / (일치수 * 가격높이 * 이미지너비) * 100
    print(f"  MA20 마스크 생성 완료. 가격 영역 대비 MA20 픽셀 비율. {ma20비율:.2f}%")

    # 정규화 (CNN 입력용)
    이미지8_norm = 이미지8 / 255.0

    return 이미지8_norm, 레이블, 날짜, 확률, 예측레이블, 마스크


# ══════════════════════════════════════════════════════
# 4. MA20 주목도 분석
# ══════════════════════════════════════════════════════
def MA20주목도분석(이미지, 인덱스목록, 마스크, 타겟클래스=None):
    """
    지정된 인덱스의 이미지에 대해 Grad-CAM을 계산하고,
    MA20 영역, 비MA20 가격 영역, 거래량 영역의 평균 주목도를 산출한다.
    """
    if len(인덱스목록) > 최대개수:
        인덱스목록 = np.random.choice(인덱스목록, 최대개수, replace=False)

    ma20주목 = []
    비ma20가격주목 = []
    거래량주목 = []

    for run in range(1, 독립훈련횟수 + 1):
        가중치파일 = os.path.join(모형경로, f"stage8_run{run}.pt")
        모형 = ChartCNN().to(디바이스)
        모형.load_state_dict(torch.load(가중치파일, map_location=디바이스,
                                       weights_only=True))
        모형.eval()
        gradcam = GradCAM(모형)

        for idx in 인덱스목록:
            텐서 = torch.FloatTensor(이미지[idx][None, None, :, :]).to(디바이스)
            cam = gradcam(텐서, 타겟클래스=타겟클래스)

            # MA20 마스크가 있는 픽셀의 주목도
            ma20_mask = 마스크[idx]
            if ma20_mask.sum() > 0:
                ma20주목.append(cam[ma20_mask].mean())

            # 비MA20 가격 영역 (가격 영역에서 MA20을 제외한 나머지)
            비ma20_mask = ~ma20_mask.copy()
            비ma20_mask[가격높이:, :] = False  # 거래량 영역 제외
            if 비ma20_mask.sum() > 0:
                비ma20가격주목.append(cam[비ma20_mask].mean())

            # 거래량 영역
            거래량주목.append(cam[가격높이:, :].mean())

    return {
        "MA20주목도": np.mean(ma20주목) if ma20주목 else np.nan,
        "비MA20가격주목도": np.mean(비ma20가격주목) if 비ma20가격주목 else np.nan,
        "거래량주목도": np.mean(거래량주목) if 거래량주목 else np.nan,
        "표본수": len(인덱스목록),
    }


# ══════════════════════════════════════════════════════
# 5. 시각화
# ══════════════════════════════════════════════════════
def 결과그림저장(결과목록):
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 19,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })
    
    범주 = [r["분류"] for r in 결과목록]
    ma20 = [r["MA20주목도"] for r in 결과목록]
    비ma20 = [r["비MA20가격주목도"] for r in 결과목록]
    거래량 = [r["거래량주목도"] for r in 결과목록]

    x = np.arange(len(범주))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.bar(x - w, ma20, w, color='#E41A1C', alpha=0.85, label='MA20 영역')
    ax.bar(x, 비ma20, w, color='#4472C4', alpha=0.85, label='비MA20 가격 영역')
    ax.bar(x + w, 거래량, w, color='#A5A5A5', alpha=0.85, label='거래량 영역')

    ax.set_xticks(x)
    ax.set_xticklabels(범주, fontsize=14)
    ax.set_ylabel('평균 주목도', fontsize=15)
    ax.set_title('분류 범주별 영역 주목도 (Stage 8)', fontsize=16)
    ax.legend(fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(출력경로, "ma20_attention_summary.png"),
                dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  저장. ma20_attention_summary.png")


# ══════════════════════════════════════════════════════
# 6. 메인
# ══════════════════════════════════════════════════════
def 메인():
    print("=" * 60)
    print("Grad-CAM MA20 주목도 분석 (Stage 8)")
    print("=" * 60)

    np.random.seed(42)

    이미지, 레이블, 날짜, 확률, 예측레이블, 마스크 = 데이터로드()

    # ── 분류 결과별 인덱스 ──
    TP = np.where((예측레이블 == 1) & (레이블 == 1))[0]
    TN = np.where((예측레이블 == 0) & (레이블 == 0))[0]
    FP = np.where((예측레이블 == 1) & (레이블 == 0))[0]
    FN = np.where((예측레이블 == 0) & (레이블 == 1))[0]

    코로나마스크 = (날짜 >= "2020-01-01") & (날짜 <= "2020-03-31")
    코로나인덱스 = np.where(코로나마스크)[0]

    print(f"\n  TP. {len(TP):,}개")
    print(f"  TN. {len(TN):,}개")
    print(f"  FP. {len(FP):,}개")
    print(f"  FN. {len(FN):,}개")
    print(f"  코로나 급락기. {len(코로나인덱스):,}개")

    # ── 범주별 MA20 주목도 분석 ──
    결과목록 = []

    print(f"\nMA20 주목도 분석 중...")

    print(f"  TP (정분류 상승)...")
    r = MA20주목도분석(이미지, TP, 마스크, 타겟클래스=1)
    r["분류"] = "TP"
    결과목록.append(r)

    print(f"  TN (정분류 하락)...")
    r = MA20주목도분석(이미지, TN, 마스크, 타겟클래스=0)
    r["분류"] = "TN"
    결과목록.append(r)

    print(f"  FP (거짓 양성)...")
    r = MA20주목도분석(이미지, FP, 마스크, 타겟클래스=1)
    r["분류"] = "FP"
    결과목록.append(r)

    print(f"  FN (거짓 음성)...")
    r = MA20주목도분석(이미지, FN, 마스크, 타겟클래스=0)
    r["분류"] = "FN"
    결과목록.append(r)

    if len(코로나인덱스) > 0:
        print(f"  코로나 급락기...")
        r = MA20주목도분석(이미지, 코로나인덱스, 마스크)
        r["분류"] = "COVID crash"
        결과목록.append(r)

    # ── 결과 저장 ──
    결과df = pd.DataFrame(결과목록)
    결과df = 결과df[["분류", "MA20주목도", "비MA20가격주목도", "거래량주목도", "표본수"]]
    결과df.to_csv(os.path.join(출력경로, "ma20_attention.csv"), index=False)

    # ── 시각화 ──
    결과그림저장(결과목록)

    # ── 콘솔 출력 ──
    print(f"\n{'='*70}")
    print("MA20 주목도 분석 결과")
    print(f"{'='*70}")
    print(f"  {'분류':<16} {'MA20':>8} {'비MA20':>8} {'거래량':>8} "
          f"{'MA20/비MA20':>12} {'표본수':>8}")
    print(f"  {'-'*62}")
    for r in 결과목록:
        비율 = r['MA20주목도'] / r['비MA20가격주목도'] if r['비MA20가격주목도'] > 0 else float('inf')
        print(f"  {r['분류']:<16} {r['MA20주목도']:>8.4f} {r['비MA20가격주목도']:>8.4f} "
              f"{r['거래량주목도']:>8.4f} {비율:>12.2f} {r['표본수']:>8}")
    print(f"{'='*70}")
    print(f"\n결과 저장. {출력경로}/")


if __name__ == "__main__":
    메인()
