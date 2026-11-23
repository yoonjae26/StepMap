# 모델 (Go2 탑재)

모델 파일은 용량(각 약 55 MB) 때문에 저장소에 넣지 않고 **GitHub Releases** 에 첨부합니다.
받은 파일을 이 폴더에 두면 `training/eval_system.py`, `training/replay_ai.py` 가 사용합니다.
로봇에서는 `~/terrain_dataset/models/` 에 두고 TensorRT 엔진으로 변환합니다.

| 파일 | 역할 | 입력 → 출력 | SHA-256 |
|---|---|---|---|
| `damage_seg.onnx` | 손상 탐지 (현장 데이터 파인튜닝) | `image [1,3,480,640]` → 6 클래스 | `15332d83fb67341b522444ec87b6da1a1f1448aac52862d6b29926dbf11efd8a` |
| `damage_types.onnx` | 손상 유형 (파손 · 결손 · 이탈·분리) | `image [1,3,480,640]` → `probs [1,8,120,160]` | `d6aee876bcd5d97f841f6cafbf2db362bd5fa7421d855ed00e58a531fc3785af` |
| `damage_seg_b1.onnx` | 비교용: AI Hub만 학습 (파인튜닝 전) | `image [1,3,480,640]` → 6 클래스 | `54f87ceb747bf62a4c0a6a31a646740074aab1b63a4a9158ce66214fc33c30f5` |

```bash
# Jetson (Go2): ONNX -> TensorRT
/usr/src/tensorrt/bin/trtexec --onnx=damage_seg.onnx   --fp16 --saveEngine=damage_seg.engine
/usr/src/tensorrt/bin/trtexec --onnx=damage_types.onnx --fp16 --saveEngine=damage_types.engine
```

클래스: 0 배경 · 1 정상 노면 · 2 손상(파손) · 3 점자블록 · 4 시설물 · 5 주의구역 · 6 결손 · 7 이탈·분리
