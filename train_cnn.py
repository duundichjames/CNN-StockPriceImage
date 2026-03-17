"""
CNN 학습 및 평가 스크립트
Jiang, Kelly, Xiu (2023) / Gu, Kelly, Xiu (2020) 의 훈련 절차를 따른다.

아키텍처.
  3개 빌딩 블록 (Conv2d → BatchNorm → LeakyReLU → MaxPool2d)
  블록별 필터 수. 32, 64, 128
  완전 연결 레이어 → 드롭아웃(70%) → 소프트맥스(2클래스)

훈련 절차.
  - 훈련 데이터의 70%를 학습용, 30%를 검증용으로 무작위 분할
  - 교차 엔트로피 손실함수
  - Adam 최적화 (학습률 1e-5, 배치 크기 128)
  - Xavier 초기화
  - 조기 중단 (검증 손실 연속 2 에포크 미개선 시 중단)
  - CNN 최적화의 확률적 특성을 감안하여 5회 독립 훈련 후 예측 평균

입력.
    data/images/stage{1-4}_train.npz
    data/images/stage{1-4}_test.npz

출력.
    models/
      stage{1-4}_run{1-5}.pt          (학습된 모형 가중치)
    results/
      stage{1-4}_predictions.npz       (테스트 예측 확률, 레이블, 날짜, 티커)
      training_log.csv                 (에포크별 손실/정확도 기록)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
이미지경로 = "data/images"
모형경로 = "models"
결과경로 = "results"
os.makedirs(모형경로, exist_ok=True)
os.makedirs(결과경로, exist_ok=True)

# 하이퍼파라미터
학습률 = 1e-5
배치크기 = 128
최대에포크 = 100
조기중단인내 = 2        # 검증 손실 연속 미개선 허용 에포크 수
드롭아웃비율 = 0.5
훈련검증비율 = 0.7      # 훈련 데이터 중 학습용 비율 (나머지 30%는 검증)
독립훈련횟수 = 5         # CNN 확률적 특성 감안, 5회 독립 훈련 후 예측 평균
시드기본값 = 42

# 디바이스 설정 (MPS > CUDA > CPU)
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
    """
    3개 빌딩 블록으로 구성된 CNN.
    각 블록. Conv2d → BatchNorm2d → LeakyReLU → MaxPool2d
    블록별 필터 수. 32, 64, 128
    최종. Flatten → Linear → Dropout(70%) → Linear(2) → Softmax
    """
    def __init__(self, 입력높이=64, 입력너비=60):
        super().__init__()

        # ── 빌딩 블록 1 (필터 32개) ──
        # 단순한 국소 패턴 탐지. OHLC 바 길이, 바 간 간격, 거래량 높낮이
        self.블록1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── 빌딩 블록 2 (필터 64개) ──
        # 단순 패턴의 조합 탐지
        self.블록2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── 빌딩 블록 3 (필터 128개) ──
        # 복합 다층 패턴 탐지 (추세 전환 신호 등)
        self.블록3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── 완전 연결 레이어 크기 산출 ──
        # 3번의 MaxPool2d(2,2)를 거치면 공간 차원이 1/8로 축소
        풀링후높이 = 입력높이 // 8   # 64 → 8
        풀링후너비 = 입력너비 // 8   # 60 → 7 (60//2=30, 30//2=15, 15//2=7)
        평탄화크기 = 128 * 풀링후높이 * 풀링후너비

        # ── 분류 헤드 ──
        self.분류기 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(평탄화크기, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=드롭아웃비율),
            nn.Linear(128, 2),
        )

        # ── Xavier 초기화 ──
        self._가중치초기화()

    def _가중치초기화(self):
        """Xavier 초기화를 모든 Conv2d와 Linear 레이어에 적용한다."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.블록1(x)
        x = self.블록2(x)
        x = self.블록3(x)
        x = self.분류기(x)
        return x


# ══════════════════════════════════════════════════════
# 2. 데이터 로드 유틸리티
# ══════════════════════════════════════════════════════
def 데이터로드(stage, split):
    """
    npz 파일을 로드하여 이미지와 레이블을 반환한다.
    이미지는 [0, 255] → [0, 1] 정규화하고, (N, 1, H, W) 형태로 변환한다.
    """
    파일 = os.path.join(이미지경로, f"{stage}_{split}.npz")
    data = np.load(파일, allow_pickle=True)

    images = data["images"].astype(np.float32) / 255.0  # [0, 1] 정규화
    images = images[:, np.newaxis, :, :]                 # (N, 1, H, W) 채널 차원 추가
    labels = data["labels"].astype(np.int64)

    return images, labels, data["dates"], data["tickers"]


def 데이터셋생성(images, labels):
    """numpy 배열을 PyTorch TensorDataset으로 변환한다."""
    X = torch.from_numpy(images)
    y = torch.from_numpy(labels)
    return TensorDataset(X, y)


# ══════════════════════════════════════════════════════
# 3. 단일 훈련 실행
# ══════════════════════════════════════════════════════
def 단일훈련(stage, run_id, images, labels):
    """
    한 번의 독립적 CNN 훈련을 수행한다.

    훈련 데이터를 70:30으로 무작위 분할하여 학습/검증 세트를 구성한다.
    @jiang2023reimagining 에 따르면 무작위 분할은 상승/하락 레이블의
    균형을 맞추는 데 도움이 되며, 장기 강세/약세 시장 변동으로 인한
    분류 편향을 완화한다.
    """
    print(f"    Run {run_id}/{독립훈련횟수} 시작...")

    # 재현성을 위한 시드 설정 (run마다 다른 시드)
    torch.manual_seed(시드기본값 + run_id)
    np.random.seed(시드기본값 + run_id)

    # 데이터셋 생성 및 훈련/검증 분할
    전체셋 = 데이터셋생성(images, labels)
    훈련크기 = int(len(전체셋) * 훈련검증비율)
    검증크기 = len(전체셋) - 훈련크기
    훈련셋, 검증셋 = random_split(전체셋, [훈련크기, 검증크기])

    훈련로더 = DataLoader(훈련셋, batch_size=배치크기, shuffle=True)
    검증로더 = DataLoader(검증셋, batch_size=배치크기, shuffle=False)

    # 모형 초기화
    모형 = ChartCNN().to(디바이스)
    손실함수 = nn.CrossEntropyLoss()
    최적화기 = optim.Adam(모형.parameters(), lr=학습률)

    # ── 훈련 루프 ────────────────────────────────────
    최선검증손실 = float("inf")
    미개선횟수 = 0
    훈련기록 = []

    for 에포크 in range(1, 최대에포크 + 1):
        # --- 학습 ---
        모형.train()
        훈련손실합 = 0
        훈련정답수 = 0
        훈련총수 = 0

        for X_batch, y_batch in 훈련로더:
            X_batch = X_batch.to(디바이스)
            y_batch = y_batch.to(디바이스)

            최적화기.zero_grad()
            출력 = 모형(X_batch)
            손실 = 손실함수(출력, y_batch)
            손실.backward()
            최적화기.step()

            훈련손실합 += 손실.item() * len(y_batch)
            훈련정답수 += (출력.argmax(dim=1) == y_batch).sum().item()
            훈련총수 += len(y_batch)

        훈련손실 = 훈련손실합 / 훈련총수
        훈련정확도 = 훈련정답수 / 훈련총수

        # --- 검증 ---
        모형.eval()
        검증손실합 = 0
        검증정답수 = 0
        검증총수 = 0

        with torch.no_grad():
            for X_batch, y_batch in 검증로더:
                X_batch = X_batch.to(디바이스)
                y_batch = y_batch.to(디바이스)

                출력 = 모형(X_batch)
                손실 = 손실함수(출력, y_batch)

                검증손실합 += 손실.item() * len(y_batch)
                검증정답수 += (출력.argmax(dim=1) == y_batch).sum().item()
                검증총수 += len(y_batch)

        검증손실 = 검증손실합 / 검증총수
        검증정확도 = 검증정답수 / 검증총수

        훈련기록.append({
            "stage": stage, "run": run_id, "epoch": 에포크,
            "train_loss": 훈련손실, "train_acc": 훈련정확도,
            "val_loss": 검증손실, "val_acc": 검증정확도,
        })

        # --- 조기 중단 ---
        if 검증손실 < 최선검증손실:
            최선검증손실 = 검증손실
            미개선횟수 = 0
            # 최선 모형 저장
            최선가중치 = {k: v.clone() for k, v in 모형.state_dict().items()}
        else:
            미개선횟수 += 1

        if 에포크 % 5 == 0 or 미개선횟수 >= 조기중단인내:
            print(
                f"      에포크 {에포크}. "
                f"훈련 손실={훈련손실:.4f} 정확도={훈련정확도:.3f}, "
                f"검증 손실={검증손실:.4f} 정확도={검증정확도:.3f}"
            )

        if 미개선횟수 >= 조기중단인내:
            print(f"      조기 중단 (에포크 {에포크})")
            break

    # 최선 가중치 복원 및 저장
    모형.load_state_dict(최선가중치)
    모형파일 = os.path.join(모형경로, f"{stage}_run{run_id}.pt")
    torch.save(최선가중치, 모형파일)

    return 모형, 훈련기록


# ══════════════════════════════════════════════════════
# 4. 테스트 예측
# ══════════════════════════════════════════════════════
def 테스트예측(모형, images):
    """
    학습된 모형으로 테스트 이미지의 상승 확률을 산출한다.
    소프트맥스 출력의 클래스 1(상승) 확률을 반환한다.
    """
    모형.eval()
    X = torch.from_numpy(images).to(디바이스)
    데이터셋 = TensorDataset(X)
    로더 = DataLoader(데이터셋, batch_size=배치크기, shuffle=False)

    확률목록 = []
    with torch.no_grad():
        for (X_batch,) in 로더:
            출력 = 모형(X_batch)
            확률 = torch.softmax(출력, dim=1)[:, 1]  # 클래스 1(상승) 확률
            확률목록.append(확률.cpu().numpy())

    return np.concatenate(확률목록)


# ══════════════════════════════════════════════════════
# 5. 메인
# ══════════════════════════════════════════════════════
def 메인():
    print(f"디바이스. {디바이스}")
    print()

    전체훈련기록 = []

    for stage in ["stage1", "stage2", "stage3", "stage4"]:
        print(f"{'='*60}")
        print(f"{stage} 학습 시작")
        print(f"{'='*60}")

        # ── 데이터 로드 ──────────────────────────────
        훈련이미지, 훈련레이블, _, _ = 데이터로드(stage, "train")
        테스트이미지, 테스트레이블, 테스트날짜, 테스트티커 = 데이터로드(stage, "test")

        print(f"  훈련. {len(훈련레이블):,}개 (상승 {훈련레이블.mean():.1%})")
        print(f"  테스트. {len(테스트레이블):,}개 (상승 {테스트레이블.mean():.1%})")

        # ── 5회 독립 훈련 ────────────────────────────
        # CNN 최적화의 확률적 특성을 감안하여 5회 독립 훈련 후
        # 예측 확률을 평균한다 (@gu2020empirical 을 따름).
        예측확률목록 = []

        for run in range(1, 독립훈련횟수 + 1):
            모형, 기록 = 단일훈련(stage, run, 훈련이미지, 훈련레이블)
            전체훈련기록.extend(기록)

            확률 = 테스트예측(모형, 테스트이미지)
            예측확률목록.append(확률)

        # ── 5회 예측 평균 ────────────────────────────
        평균확률 = np.mean(예측확률목록, axis=0)
        예측레이블 = (평균확률 >= 0.5).astype(int)
        정확도 = (예측레이블 == 테스트레이블).mean()
        print(f"\n  {stage} 테스트 정확도 (5회 평균). {정확도:.4f}")

        # ── 예측 결과 저장 ───────────────────────────
        np.savez_compressed(
            os.path.join(결과경로, f"{stage}_predictions.npz"),
            probabilities=평균확률,           # 상승 확률 (5회 평균)
            predictions=예측레이블,            # 이진 예측 (0 또는 1)
            labels=테스트레이블,               # 실제 레이블
            dates=테스트날짜,
            tickers=테스트티커,
            run_probabilities=np.array(예측확률목록),  # 개별 run 확률
        )
        print(f"  저장. {결과경로}/{stage}_predictions.npz")
        print()

    # ── 훈련 기록 저장 ───────────────────────────────
    기록df = pd.DataFrame(전체훈련기록)
    기록df.to_csv(os.path.join(결과경로, "training_log.csv"), index=False)
    print(f"훈련 기록 저장. {결과경로}/training_log.csv")

    # ── 최종 요약 ────────────────────────────────────
    print(f"\n{'='*60}")
    print("최종 요약")
    print(f"{'='*60}")
    for stage in ["stage1", "stage2", "stage3", "stage4"]:
        data = np.load(os.path.join(결과경로, f"{stage}_predictions.npz"))
        정확도 = (data["predictions"] == data["labels"]).mean()
        상승비율 = data["predictions"].mean()
        print(f"  {stage}. 정확도 {정확도:.4f}, 상승 예측 비율 {상승비율:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    메인()
