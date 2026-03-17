"""
CNN 학습 및 평가 스크립트 (9단계)

9단계.
  stage1. 종가선              stage6. OHLC+MA20
  stage2. OHLC                stage7. 종가선+MA20+MA60
  stage3. 종가선+거래량         stage8. OHLC+거래량+MA20
  stage4. OHLC+거래량          stage9. OHLC+거래량+MA20+MA60
  stage5. 종가선+MA20

사용법.
    python train_cnn_s9.py

입력.   data/images_s9/stage{1-9}_train.npz, stage{1-9}_test.npz
출력.   models_s9/, results_s9/
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
이미지경로 = "data/images_s9"
모형경로 = "models_s9"
결과경로 = "results_s9"
os.makedirs(모형경로, exist_ok=True)
os.makedirs(결과경로, exist_ok=True)

학습률 = 1e-5
배치크기 = 128
최대에포크 = 100
조기중단인내 = 2
드롭아웃비율 = 0.5
훈련검증비율 = 0.7
독립훈련횟수 = 5
시드기본값 = 42

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
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.블록2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.블록3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        풀링후높이 = 입력높이 // 8
        풀링후너비 = 입력너비 // 8
        평탄화크기 = 128 * 풀링후높이 * 풀링후너비
        self.분류기 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(평탄화크기, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=드롭아웃비율),
            nn.Linear(128, 2),
        )
        self._가중치초기화()

    def _가중치초기화(self):
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
# 2. 데이터 로드
# ══════════════════════════════════════════════════════
def 데이터로드(stage, split):
    파일 = os.path.join(이미지경로, f"{stage}_{split}.npz")
    data = np.load(파일, allow_pickle=True)
    images = data["images"].astype(np.float32) / 255.0
    images = images[:, np.newaxis, :, :]
    labels = data["labels"].astype(np.int64)
    return images, labels, data["dates"], data["tickers"]


def 데이터셋생성(images, labels):
    return TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))


# ══════════════════════════════════════════════════════
# 3. 단일 훈련 실행
# ══════════════════════════════════════════════════════
def 단일훈련(stage, run_id, images, labels):
    print(f"    Run {run_id}/{독립훈련횟수} 시작...")
    torch.manual_seed(시드기본값 + run_id)
    np.random.seed(시드기본값 + run_id)

    전체셋 = 데이터셋생성(images, labels)
    훈련크기 = int(len(전체셋) * 훈련검증비율)
    검증크기 = len(전체셋) - 훈련크기
    훈련셋, 검증셋 = random_split(전체셋, [훈련크기, 검증크기])

    훈련로더 = DataLoader(훈련셋, batch_size=배치크기, shuffle=True)
    검증로더 = DataLoader(검증셋, batch_size=배치크기, shuffle=False)

    모형 = ChartCNN().to(디바이스)
    손실함수 = nn.CrossEntropyLoss()
    최적화기 = optim.Adam(모형.parameters(), lr=학습률)

    최선검증손실 = float("inf")
    미개선횟수 = 0
    최선가중치 = None
    훈련기록 = []

    for 에포크 in range(1, 최대에포크 + 1):
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

        if 검증손실 < 최선검증손실:
            최선검증손실 = 검증손실
            미개선횟수 = 0
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

    모형.load_state_dict(최선가중치)
    torch.save(최선가중치, os.path.join(모형경로, f"{stage}_run{run_id}.pt"))
    return 모형, 훈련기록


# ══════════════════════════════════════════════════════
# 4. 테스트 예측
# ══════════════════════════════════════════════════════
def 테스트예측(모형, images):
    모형.eval()
    로더 = DataLoader(TensorDataset(torch.from_numpy(images).to(디바이스)),
                     batch_size=배치크기, shuffle=False)
    확률목록 = []
    with torch.no_grad():
        for (X_batch,) in 로더:
            출력 = 모형(X_batch)
            확률 = torch.softmax(출력, dim=1)[:, 1]
            확률목록.append(확률.cpu().numpy())
    return np.concatenate(확률목록)


# ══════════════════════════════════════════════════════
# 5. 메인
# ══════════════════════════════════════════════════════
def 메인():
    print(f"디바이스. {디바이스}")
    print()

    전체훈련기록 = []

    for stage in 전체단계:
        print(f"{'='*60}")
        print(f"{stage} ({단계설명[stage]}) 학습 시작")
        print(f"{'='*60}")

        훈련이미지, 훈련레이블, _, _ = 데이터로드(stage, "train")
        테스트이미지, 테스트레이블, 테스트날짜, 테스트티커 = 데이터로드(stage, "test")

        print(f"  훈련. {len(훈련레이블):,}개 (상승 {훈련레이블.mean():.1%})")
        print(f"  테스트. {len(테스트레이블):,}개 (상승 {테스트레이블.mean():.1%})")

        예측확률목록 = []
        for run in range(1, 독립훈련횟수 + 1):
            모형, 기록 = 단일훈련(stage, run, 훈련이미지, 훈련레이블)
            전체훈련기록.extend(기록)
            확률 = 테스트예측(모형, 테스트이미지)
            예측확률목록.append(확률)

        평균확률 = np.mean(예측확률목록, axis=0)
        예측레이블 = (평균확률 >= 0.5).astype(int)
        정확도 = (예측레이블 == 테스트레이블).mean()
        print(f"\n  {stage} ({단계설명[stage]}) 테스트 정확도 (5회 평균). {정확도:.4f}")

        np.savez_compressed(
            os.path.join(결과경로, f"{stage}_predictions.npz"),
            probabilities=평균확률,
            predictions=예측레이블,
            labels=테스트레이블,
            dates=테스트날짜,
            tickers=테스트티커,
            run_probabilities=np.array(예측확률목록),
        )
        print(f"  저장. {결과경로}/{stage}_predictions.npz")
        print()

    기록df = pd.DataFrame(전체훈련기록)
    기록df.to_csv(os.path.join(결과경로, "training_log.csv"), index=False)
    print(f"훈련 기록 저장. {결과경로}/training_log.csv")

    print(f"\n{'='*60}")
    print("최종 요약")
    print(f"{'='*60}")
    print(f"  {'단계':<10} {'설명':<24} {'정확도':>8} {'상승예측비율':>12}")
    print(f"  {'-'*54}")
    for stage in 전체단계:
        data = np.load(os.path.join(결과경로, f"{stage}_predictions.npz"))
        정확도 = (data["predictions"] == data["labels"]).mean()
        상승비율 = data["predictions"].mean()
        print(f"  {stage:<10} {단계설명[stage]:<24} {정확도:>8.4f} {상승비율:>12.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    메인()
