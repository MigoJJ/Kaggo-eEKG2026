# Phase 7 외부기관 교차검증 결과 (CinC2020: CPSC2018, Chapman-Shaoxing+Ningbo)

측정만 수행 — 재보정 없음. RuleThresholds/Stage1 임계값은 이 결과로 자동 변경되지 않음.


## Track A: Stage1 P(NORM) vs SNOMED CT 426783006(Sinus rhythm) 단독

| 소스 | n | NORM수 | 비정상수 | AUROC | F1@0.5 | best F1 | best F1 threshold |
|---|---|---|---|---|---|---|---|
| 전체 | 51034 | 6700 | 44334 | 0.717 | 0.348 | 0.348 | 0.500 |
| chapman_shaoxing_ningbo | 44159 | 5782 | 38377 | 0.707 | 0.343 | 0.343 | 0.500 |
| cpsc2018 | 6875 | 918 | 5957 | 0.780 | 0.383 | 0.393 | 0.690 |

## Track B: 개별 finding 참고용 F1 (⚠️ 진단 세분화 수준 차이로 참고용, 재보정에 쓰지 않음)

| 소스 | finding | 양성수 | precision | recall | F1 |
|---|---|---|---|---|---|
| 전체 | RBBB | 2506 | 0.089 | 0.068 | 0.077 |
| 전체 | LBBB | 476 | 0.364 | 0.008 | 0.016 |
| 전체 | AVB1 | 1856 | 0.113 | 0.855 | 0.200 |
| chapman_shaoxing_ningbo | RBBB | 649 | 0.045 | 0.109 | 0.064 |
| chapman_shaoxing_ningbo | LBBB | 240 | 0.167 | 0.004 | 0.008 |
| chapman_shaoxing_ningbo | AVB1 | 1134 | 0.078 | 0.822 | 0.143 |
| cpsc2018 | RBBB | 1857 | 0.289 | 0.054 | 0.091 |
| cpsc2018 | LBBB | 236 | 0.600 | 0.013 | 0.025 |
| cpsc2018 | AVB1 | 722 | 0.308 | 0.907 | 0.460 |
