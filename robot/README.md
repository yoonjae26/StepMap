# robot — Go2에서 실행되는 노드

Go2 EDU (Jetson, ROS 2 Foxy) 의 `~/terrain_dataset/` 에 복사해 실행합니다.

| 파일 | 역할 |
|---|---|
| `sidewalk_measure.launch.py` | 전체 실행 (센서 · 측정 노드 · 대시보드) |
| `sensors.launch.py` | RealSense D435i, static TF, odometry → TF |
| `surface_seg_node.py` | 깊이 → 함몰·돌출·요철, AI 손상 탐지 + 유형 + 단차 결합 |
| `damage_ai.py` | 모델 실행(TensorRT), 손상 마스크, 깊이 결합(`fuse_damage`) |
| `surface_seg.py` | 노면 평면 · 높이 계산, 카메라 화면 오버레이 |
| `slope_core.py` | 평면 추정 · 경사 계산 |
| `terrain_geometry_node.py` | 경사 · 단차 · 장애물 |
| `inspect_node.py` | 같은 위치의 탐지를 하나의 지점으로 확정, 사진 저장 |
| `stepmap_bridge.py` | 지점 · 경로 · 배터리 · 사진 → 대시보드 |
| `foot_node.py` | (선택, 기본 꺼짐) 발 접촉으로 흔들리는 블록 |
| `stepmap_live.sh`, `setup_clock_sync.sh` | 대시보드 실행, 로봇 시계 동기화 |

```bash
ros2 launch ~/terrain_dataset/sidewalk_measure.launch.py session:=20261113_ui location:="제주시 ..."
```

모델은 [`../models`](../models), 한글 오버레이 글꼴은 `fonts/NotoSansCJKkr-Bold.otf` (Noto Sans CJK, SIL OFL) 를 받아 두세요.
