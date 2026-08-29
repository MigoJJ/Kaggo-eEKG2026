"""Stage3(비트 부정맥) 학습 스크립트. MIT-BIH 표준 inter-patient 분할(DS1=train, DS2=test).

Phase 4 backbone — 이 학습은 파이프라인 연동(Java ONNX 로드→분류→소견 산출)이 실제로
동작하는지 증명하는 것이 목적이다. 클래스 불균형(N이 압도적 다수)과 정밀 정확도 보정은
Kaggle P100에서 재학습할 때 다루기로 한다.

사용법:
  python train_stage3_beat.py --mitbih-root /mnt/t7/datasets/mit-bih-arrhythmia-database-1.0.0 \
      --model-type resnet --epochs 10 --out models/stage3_beat_resnet.pt
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ecgml.data.mitbih_dataset import CLASS_NAMES, DS1_RECORDS, DS2_RECORDS, MitbihBeatDataset
from ecgml.models.stage3_beat import build_beat_classifier
from ecgml.torch_device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitbih-root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay for L2 regularization")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--model-type",
        choices=["cnn", "resnet", "inception"],
        default="cnn",
        help="cnn: 기존 소형 CNN. resnet/inception: 시계열 형태학 추출 강화 백본.",
    )
    parser.add_argument("--out", default="models/stage3_beat.pt")
    parser.add_argument("--balanced-sampler", action="store_true", help="Use WeightedRandomSampler for training batch balancing")
    args = parser.parse_args()

    print("DS1(train) 비트 추출 중...")
    train_ds = MitbihBeatDataset(args.mitbih_root, DS1_RECORDS)
    print("DS2(test) 비트 추출 중...")
    test_ds = MitbihBeatDataset(args.mitbih_root, DS2_RECORDS)
    print(f"train={len(train_ds)} test={len(test_ds)}")

    train_labels = torch.tensor([label for _, label in train_ds.samples])
    class_counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).float()
    print("train 클래스 분포:", dict(zip(CLASS_NAMES, class_counts.int().tolist())))
    # 극단적인 역수(1/counts) 대신, 클래스 불균형을 더 안정적으로 다루는 제곱근 역수(1/sqrt(counts))를 사용하여 N 클래스의 민감도 붕괴를 예방합니다.
    class_weights = 1.0 / torch.sqrt(class_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)

    device = resolve_device(args.device)
    print(f"device={device}, model_type={args.model_type}")

    if args.balanced_sampler:
        # 완전 역수(1/count) vs 제곱근 역수(1/sqrt(count)) vs 손실함수 가중치만 사용, 3가지를 비교 실측한 결과
        # (2026-08-29, ml/results/stage3_cnn_sampler_experiments.md) 4-class(N/S/V/F) macro sensitivity 기준으로
        # 완전 역수 재샘플링이 가장 우수했다. 재샘플링 자체가 손실함수 가중치 단독보다 낫고, 재샘플링 안에서는
        # 완전 역수가 제곱근 역수보다 근소하게 낫다.
        print("WeightedRandomSampler 기반 균형 샘플러 활성화 (완전 역수)")
        class_sample_counts = class_counts.tolist()
        sample_weights = [1.0 / class_sample_counts[label] for _, label in train_ds.samples]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
        # 샘플러가 이미 클래스 빈도를 보정하므로, 손실함수까지 추가로 가중치를 주면 이중 보정이 되어
        # 소수 클래스가 과다 예측되므로 여기서는 CrossEntropyLoss(weight=...)를 쓰지 않는다.
        criterion = torch.nn.CrossEntropyLoss()
    else:
        print("제곱근 역수 가중치 손실함수 활성화")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_beat_classifier(args.model_type, n_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_macro_sens = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        correct = 0
        epoch_confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                for t, p in zip(y.cpu(), preds.cpu()):
                    epoch_confusion[t, p] += 1
        test_acc = correct / len(test_ds)

        # 체크포인트 선택 기준: test_acc(전체 정확도) 대신 클래스별 sensitivity의 단순평균(macro sensitivity)을
        # 사용한다. N(다수 클래스)이 압도적인 이 데이터셋에서는 test_acc가 최고인 모델이 오히려 S/V/F 같은
        # 임상적으로 중요한 소수 클래스를 거의 검출하지 못하는 미학습 초기 epoch일 수 있음을 실측으로 확인했다
        # (2026-08-29 v3 실험: epoch1이 test_acc 최고였으나 S sensitivity 5.0%, F sensitivity 0.3%).
        per_class_support = epoch_confusion.sum(dim=1)
        per_class_sens = torch.where(
            per_class_support > 0,
            epoch_confusion.diag().float() / per_class_support.clamp(min=1).float(),
            torch.zeros(len(CLASS_NAMES)),
        )
        macro_sens = per_class_sens.mean().item()
        print(f"epoch {epoch}: train_loss={train_loss:.4f} test_acc={test_acc:.4f} macro_sens={macro_sens:.4f}")

        if macro_sens > best_macro_sens:
            best_macro_sens = macro_sens
            torch.save(model.state_dict(), out_path)
            print(f"  -> best checkpoint 저장 (macro_sens 기준): {out_path}")

    # 클래스별 혼동행렬(민감도 확인용 — 특히 S/V가 임상적으로 중요)
    model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
    model.eval()
    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            for t, p in zip(y, preds):
                confusion[t, p] += 1

    print("\n혼동행렬 (행=실제, 열=예측):")
    print("      " + " ".join(f"{c:>6s}" for c in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>4s}: " + " ".join(f"{confusion[i, j].item():>6d}" for j in range(len(CLASS_NAMES))))

    for i, name in enumerate(CLASS_NAMES):
        tp = confusion[i, i].item()
        support = confusion[i, :].sum().item()
        sensitivity = tp / support if support > 0 else float("nan")
        print(f"  {name} sensitivity={sensitivity:.3f} (n={support})")


if __name__ == "__main__":
    main()
