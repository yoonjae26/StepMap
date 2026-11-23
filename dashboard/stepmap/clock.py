"""로봇 시계를 노트북 시각으로 맞춘다 (LAN 케이블로 연결했을 때).

Jetson은 RTC 배터리가 없어 인터넷이 없으면 1970년부터 센다. 노트북이 ssh로 로봇 시각을 읽고, 차이가 크면
`sudo -n /bin/date -u -s @<노트북 시각>`으로 바꾼다 (왕복 시간의 절반을 더한다). 로봇에는 이 명령만 비밀번호 없이
허용하는 sudoers 파일이 한 번 필요하다: robot/setup_clock_sync.sh (로봇에서 사용자가 sudo로 실행).

  stepmap sync-clock --host go2              # 한 번
  stepmap sync-clock --host go2 --watch 30   # 30초마다 확인 (케이블을 꽂으면 자동으로 맞춘다)
  stepmap live -d ... --pull ... --sync-clock go2   # 실시간 복제와 함께

맞춘 뒤의 시각은 live 서버의 1970 보정(clock_offset)을 거치지 않는다 (CLOCK_VALID보다 큰 시각).
녹화·측정을 시작하기 전에 맞추는 것이 가장 좋다 (세션 이름·사진 파일 이름도 실제 날짜가 된다).
"""
from __future__ import annotations

import subprocess
import time

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=4"]
MAX_SKEW_S = 1.0                 # 이보다 많이 다르면 맞춘다
MAX_RTT_S = 1.5                  # 왕복이 이보다 길면 정확하지 않으니 다시 잰다


def _ssh(host: str, cmd: str, timeout: float = 8.0) -> str:
    r = subprocess.run(["ssh", *SSH_OPTS, host, cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or f"ssh {host} 실패 ({r.returncode})")
    return r.stdout.strip()


def robot_skew(host: str) -> tuple[float, float]:
    """(로봇 시각 - 노트북 시각 [s], 왕복 [s]). 왕복 가운데 시점의 노트북 시각과 비교한다."""
    t0 = time.time()
    robot = float(_ssh(host, "date +%s.%N"))
    t1 = time.time()
    return robot - (t0 + t1) / 2, t1 - t0


def sync_clock(host: str, max_skew: float = MAX_SKEW_S) -> dict:
    """로봇 시계를 확인하고 필요하면 맞춘다. {"skew_before", "skew_after", "set": 바꿨는지, "rtt"}."""
    skew, rtt = robot_skew(host)
    if rtt > MAX_RTT_S:
        skew, rtt = robot_skew(host)
    out = {"skew_before": round(skew, 3), "rtt": round(rtt, 3), "set": False, "skew_after": round(skew, 3)}
    if abs(skew) <= max_skew:
        return out
    target = time.time() + rtt / 2                 # 명령이 로봇에 닿는 시점
    try:
        _ssh(host, f"sudo -n /bin/date -u -s @{target:.3f} >/dev/null")
    except RuntimeError as e:
        if "password" in str(e) or "sudo" in str(e):
            raise RuntimeError("로봇에서 date 권한이 없습니다. 로봇에서 한 번 실행: "
                               "sudo bash ~/stepmap_live/stepmap/robot/setup_clock_sync.sh") from e
        raise
    out["set"] = True
    out["skew_after"] = round(robot_skew(host)[0], 3)
    return out


def watch_clock(host: str, period: float = 30.0, log=print, stop=None) -> None:
    """period초마다 확인해서 맞춘다 (연결이 없으면 조용히 기다린다). stop: threading.Event."""
    connected = None
    while not (stop and stop.is_set()):
        try:
            r = sync_clock(host)
            if r["set"] or connected is not True:
                log(f"로봇 시계 {host}: 차이 {r['skew_before']:+.3f} s" +
                    (f" → 맞춤 ({r['skew_after']:+.3f} s)" if r["set"] else " (맞음)"))
            connected = True
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as e:
            if connected is not False:
                log(f"로봇 시계 {host}: 확인 못 함 ({e})")
            connected = False
        if stop:
            stop.wait(period)
        else:
            time.sleep(period)
