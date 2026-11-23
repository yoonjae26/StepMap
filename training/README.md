# training — SegFormer-B1 학습 · 파인튜닝 · 평가

경로는 환경변수로 지정합니다 (데이터는 저장소에 포함하지 않음).

| 변수 | 내용 |
|---|---|
| `STEPMAP_AIHUB` | AI Hub 「인도보행 영상」 서피스마스킹 (이미지 + 마스크) |
| `STEPMAP_FIELD` | Go2 · 스마트폰 현장 데이터 (이미지, 픽셀 라벨, 인덱스 json) |
| `STEPMAP_WORK` | 결과 폴더 (`runs/…`) |

| 스크립트 | 내용 |
|---|---|
| `train_seg.py` | AI Hub 1차 학습 (6 클래스) |
| `finetune_robot.py` | 현장 데이터 파인튜닝 → 손상 탐지 모델 |
| `finetune_types.py` | 손상 채널 → 파손 · 결손 · 이탈·분리 분리 (8 클래스), 장소 기준 5-fold 평가 |
| `eval_seg.py`, `eval_before_after_labels.py` | 평가 (재현율 · 정밀도, 유형별) |
| `eval_system.py` | Go2와 같은 방식(탐지 → 유형 → 깊이)으로 전체 시스템 평가 |
| `replay_ai.py` | 녹화된 컬러 + 깊이에 로봇 AI · 오버레이를 다시 적용 |
| `export_onnx.py`, `export_types_onnx.py` | ONNX 내보내기 |

학습 설정 요약: AdamW (인코더 3e-5, 디코더 3e-4), warmup 200 step, batch 16 (AI Hub 50% + 현장 50%), 1,000 step 체크포인트,
손실 = 픽셀 CE (손상 유형 가중 ×4) + Dice + 이미지 단위 BCE.
