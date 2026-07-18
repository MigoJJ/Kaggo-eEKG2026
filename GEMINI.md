# EKG-GDS EMR 2026 — Project Instructions & Constraints

## Global Constraints (의료 AI 리드 엔지니어 핵심 원칙)

1. **No Autonomy (자율 진단 배제)**
   - 모델은 진단을 최종 확정하지 않습니다.
   - 1단계 AI의 `P(NORM)` 점수는 오직 **의사 검토 큐의 우선순위 정렬용(Triage)** 메타데이터로만 사용합니다. 어떠한 임계값을 초과하더라도 후속 규칙 및 검증 파이프라인 단계를 건너뛰지(gating) 않습니다.

2. **Dataset-Honest (데이터셋 고유성 존중)**
   - MIT-BIH, PTB-XL, CPSC 등 데이터셋별 고유의 라벨링 체계와 리드 구성을 왜곡하거나 혼합하지 않습니다.
   - 예: MIT-BIH의 부정맥 비트 정보를 전도장애(RBBB/LBBB) 검증 용도로 교차 오용하지 않습니다.

3. **Contract-First (인터페이스 불변)**
   - Java 런타임(0~4단계)과 Python 학습 모듈 간의 유일한 약속(Contract)인 **`.onnx` 모델 파일** 및 **JSON 규격** 내에서만 구조를 수정합니다. train/serve skew 방지를 위한 규칙을 엄격히 준수합니다.

4. **Performance Constraint (연산 성능 유지)**
   - Median Filter 기선보정 등 핵심 연산에서 $O(nw)$ 증분 정렬 최적화를 훼손하는 $O(nw \log w)$ 방식의 알고리즘(예: 매 샘플마다 윈도우 복사 및 정렬)은 절대 제안하거나 복구하지 않습니다.
