import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from stepmap import demo, live
from stepmap.events import (EVENT_COLUMNS, GeoRef, merge_events, read_events, read_foot_traces, write_events)


@pytest.fixture(scope="module")
def split_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("split")
    demo.generate(d, split=True)
    return d


def items_from_split(d):
    """기능별 결과 파일을 live 항목으로 (측정 노드가 하나씩 보내는 것과 같은 내용)."""
    items = []
    for name in demo.RAW_FILES:
        for i, r in enumerate(read_events(d / "raw" / name, raw=True)):
            items.append({"type": "event", "key": f"{name}:{i}", "row": {c: r[c] for c in EVENT_COLUMNS}})
    steps = {}
    for r in read_foot_traces(d / "raw" / demo.RAW_TRACES, raw=True):
        steps.setdefault((r["leg"], r["t_event"]), []).append(r)
    for (leg, t_event), rows in steps.items():
        items.append({"type": "trace", "key": f"{leg}:{t_event}", "rows": rows})
    return items


def test_live_equals_offline_merge(split_dir, tmp_path):
    items = items_from_split(split_dir)
    store = live.LiveStore(tmp_path / "s", {})
    seq, errors = store.ingest(items)
    assert errors == [] and seq == len(items)
    assert store.export(force=True)
    rows = [r for name in demo.RAW_FILES for r in read_events(split_dir / "raw" / name, raw=True)]
    offline, _ = merge_events(rows)
    write_events(tmp_path / "offline.csv", offline)
    assert (tmp_path / "s" / "events.csv").read_bytes() == (tmp_path / "offline.csv").read_bytes()
    assert (tmp_path / "s" / "foot_traces.csv").is_file()
    page = (tmp_path / "s" / "live" / "dashboard.html").read_text(encoding="utf-8")
    assert "stepmap-live" in page and "api/version" in page
    assert store.summary["events"] == len(offline)


def test_replay_and_upsert(split_dir, tmp_path):
    items = items_from_split(split_dir)
    a = live.LiveStore(tmp_path / "s", {})
    a.ingest(items)
    first = items[0]
    changed = {**first, "row": {**first["row"], "value": 9.9}}
    a.ingest([changed])                                     # 같은 key: 덮어쓰기
    b = live.LiveStore(tmp_path / "s", {})                 # 다시 켜면 stream.jsonl에서 복원
    assert b.seq == a.seq == len(items) + 1
    assert b.events[first["key"]]["value"] == 9.9
    assert len(b.events) == len({i["key"] for i in items if i["type"] == "event"})


def test_bad_items_rejected(tmp_path):
    s = live.LiveStore(tmp_path / "s", {})
    bad = {"type": "event", "key": "k", "row": {"t": 1, "x": 0, "y": 0, "kind": "step", "value": 2, "unit": "mm",
                                               "source": "camera", "confidence": 0.9}}
    seq, errors = s.ingest([bad, {"type": "nope"}, {"type": "traj", "row": {"t": "a"}}])
    assert seq == 0 and len(errors) == 3
    assert not (tmp_path / "s" / live.STREAM).exists()
    ok = {**bad, "row": {**bad["row"], "unit": "cm"}}
    seq, errors = s.ingest([ok])
    assert seq == 1 and errors == [] and s.events["k"]["session"] == "s"   # session 빈칸이면 세션 폴더 이름


def test_redraw_throttled(tmp_path, monkeypatch):
    """다시 그리기 간격: 새 측정 지점은 REDRAW_NEW_S, 값·사진만 바뀌면 REDRAW_UPDATE_S 뒤에 (화면 깜빡임 방지)."""
    now = [1000.0]
    monkeypatch.setattr(live.time, "time", lambda: now[0])
    s = live.LiveStore(tmp_path / "s", {})
    row = {"t": 1, "x": 0, "y": 0, "kind": "step", "value": 2.5, "unit": "cm", "source": "camera", "confidence": 0.9}
    s.ingest([{"type": "event", "key": "k", "row": row}])
    assert s.export() and s.version == 1
    s.ingest([{"type": "event", "key": "k", "row": {**row, "value": 2.6}}])      # 값만 바뀜
    now[0] += 5
    assert not s.export() and s.version == 1
    now[0] += live.REDRAW_UPDATE_S
    assert s.export() and s.version == 2                                         # 미뤄 둔 변경이 그려진다
    s.ingest([{"type": "event", "key": "k2", "row": {**row, "x": 3}}])             # 새 지점
    now[0] += 1
    assert not s.export()
    now[0] += live.REDRAW_NEW_S
    assert s.export() and s.version == 3


def test_trajectory_only_does_not_redraw(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "REDRAW_NEW_S", 0.0)
    s = live.LiveStore(tmp_path / "s", {})
    s.ingest([{"type": "event", "key": "k", "row": {"t": 1, "x": 0, "y": 0, "kind": "step", "value": 2.5,
                                                    "unit": "cm", "source": "camera", "confidence": 0.9}}])
    assert s.export() and s.version == 1
    s.ingest([{"type": "traj", "row": {"t": 2, "x": 0.5, "y": 0}}])
    assert not s.export() and s.version == 1                # 궤적만 늘면 TRAJ_REFRESH_S 뒤에 다시 그린다
    s.ingest([{"type": "event", "key": "k2", "row": {"t": 3, "x": 1, "y": 0, "kind": "step", "value": 0.5,
                                                     "unit": "cm", "source": "camera", "confidence": 0.9}}])
    assert s.export() and s.version == 2


def test_robot_clock_offset(tmp_path, monkeypatch):
    s = live.LiveStore(tmp_path / "s", {})
    s.ingest([{"type": "event", "key": "k", "row": {"t": 1000.0, "x": 0, "y": 0, "kind": "slope", "value": 3.0,
                                                    "unit": "deg", "source": "camera", "confidence": 0.9}}])
    monkeypatch.setattr(live.time, "time", lambda: 1100.0)       # 로봇 시계: 1970년 근처
    s.note_remote_clock(1_792_454_400.0)                          # 노트북 시각
    monkeypatch.undo()
    s.export(force=True)
    t = read_events(tmp_path / "s" / "events.csv")[0]["t"]
    assert t == pytest.approx(1000.0 + 1_792_454_400.0 - 1100.0)
    assert json.loads((tmp_path / "s" / live.META).read_text())["clock_offset"] > 0


def test_http_ingest_and_pull_replication(split_dir, tmp_path):
    robot = live.LiveStore(tmp_path / "robot", {}, GeoRef(33.4860373, 126.4866073, 0.0))
    live._Handler.store = robot
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), live._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        items = items_from_split(split_dir)
        half = len(items) // 2
        body = json.dumps({"items": items[:half]}).encode()
        with urllib.request.urlopen(urllib.request.Request(url + "/api/ingest", body, method="POST")) as r:
            assert json.loads(r.read())["ok"]
        img = b"\xff\xd8fake-jpeg"
        req = urllib.request.Request(url + "/api/file?path=img/a.jpg", img, method="POST")
        urllib.request.urlopen(req).read()

        laptop = live.LiveStore(tmp_path / "laptop", {}, GeoRef(33.4860373, 126.4866073, 0.0))
        stop = threading.Event()
        threading.Thread(target=live._puller, args=(laptop, url, stop), daemon=True).start()
        for _ in range(100):
            if laptop.meta.get("pull_seq") == robot.seq:
                break
            time.sleep(0.05)
        robot.ingest(items[half:])                               # 연결 중에 들어온 나머지도 이어받는다
        for _ in range(100):
            if laptop.meta.get("pull_seq") == robot.seq:
                break
            time.sleep(0.05)
        stop.set()
        assert laptop.meta["pull_seq"] == robot.seq
        assert (tmp_path / "laptop" / "img" / "a.jpg").read_bytes() == img
        robot.export(force=True)
        laptop.export(force=True)
        assert (tmp_path / "laptop" / "events.csv").read_bytes() == (tmp_path / "robot" / "events.csv").read_bytes()
        with urllib.request.urlopen(url + "/api/version") as r:
            v = json.loads(r.read())
        assert v["events"] == robot.summary["events"] and v["seq"] == robot.seq
        with urllib.request.urlopen(url + "/dashboard.html") as r:
            assert b"stepmap-live" in r.read()
    finally:
        httpd.shutdown()


def test_pull_fails_over_to_next_address(split_dir, tmp_path):
    robot = live.LiveStore(tmp_path / "robot", {}, GeoRef(33.4860373, 126.4866073, 0.0))
    robot.ingest(items_from_split(split_dir))
    live._Handler.store = robot
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), live._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    good = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        laptop = live.LiveStore(tmp_path / "laptop", {}, GeoRef(33.4860373, 126.4866073, 0.0))
        stop = threading.Event()
        dead = "http://127.0.0.1:9"                             # 닿지 않는 첫 주소 (예: 케이블을 뺀 유선 LAN)
        threading.Thread(target=live._puller, args=(laptop, f"{dead},{good}", stop), daemon=True).start()
        for _ in range(200):
            if laptop.meta.get("pull_seq") == robot.seq:
                break
            time.sleep(0.05)
        stop.set()
        assert laptop.meta["pull_seq"] == robot.seq
        assert laptop.pull_state["url"] == good and laptop.pull_state["ok"]
    finally:
        httpd.shutdown()


def test_camera_mjpeg_stream(tmp_path, monkeypatch):
    cam = tmp_path / "live.jpg"
    cam.write_bytes(b"\xff\xd8frame1\xff\xd9")
    store = live.LiveStore(tmp_path / "s", {})
    store.camera = str(cam)
    monkeypatch.setattr(live, "CAMERA_IDLE_S", 1.0)
    live._Handler.store = store
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), live._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/camera.mjpg", timeout=5) as r:
            assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            body = r.read()                      # 새 사진이 없으면 CAMERA_IDLE_S 뒤 서버가 닫는다
    finally:
        httpd.shutdown()
    assert body.startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 10\r\n\r\n\xff\xd8frame1\xff\xd9\r\n")


def test_camera_info(tmp_path):
    cam = tmp_path / "live.jpg"
    cam.write_bytes(b"\xff\xd8x\xff\xd9")
    (tmp_path / "live.json").write_text(json.dumps({"written": time.time(), "fps": 15.0, "latency_ms": 80.0}))
    store = live.LiveStore(tmp_path / "s", {})
    store.camera = str(cam)
    live._Handler.store = store
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), live._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/api/camera", timeout=5) as r:
            d = json.loads(r.read())
    finally:
        httpd.shutdown()
    assert d["fps"] == 15.0 and d["latency_ms"] == 80.0 and 0 <= d["age_s"] < 5


def test_clock_offset_only_before_sync(tmp_path, monkeypatch):
    """세션 중에 로봇 시계를 맞추면(sync-clock) 그 뒤 시각에는 보정값을 더하지 않는다."""
    s = live.LiveStore(tmp_path / "s", {})
    row = {"x": 0, "y": 0, "kind": "slope", "value": 3.0, "unit": "deg", "source": "camera", "confidence": 0.9}
    s.ingest([{"type": "event", "key": "a", "row": {**row, "t": 1000.0}},
              {"type": "event", "key": "b", "row": {**row, "t": 1_792_454_500.0, "x": 5}}])
    monkeypatch.setattr(live.time, "time", lambda: 1100.0)
    s.note_remote_clock(1_792_454_400.0)
    monkeypatch.undo()
    s.export(force=True)
    ts = sorted(e["t"] for e in read_events(tmp_path / "s" / "events.csv"))
    assert ts == [pytest.approx(1000.0 + 1_792_454_400.0 - 1100.0), 1_792_454_500.0]


def test_start_point_sets_map_position(tmp_path):
    """지도에서 고른 출발점: 첫 궤적 점 = 고른 곳, 처음 걸어간 방향 = 고른 방향 → 지도 위치 (GPS 없을 때)."""
    s = live.LiveStore(tmp_path / "s", {})
    assert s.add_start(33.49, 126.49, 90.0) == "로봇 위치(odom)가 아직 없습니다"
    s.ingest([{"type": "traj", "row": {"t": float(i), "x": 10.0, "y": 5.0 + i}} for i in range(2)])
    assert "움직인 뒤" in s.add_start(33.49, 126.49, 90.0)
    s.ingest([{"type": "traj", "row": {"t": float(i), "x": 10.0, "y": 5.0 + i}} for i in range(2, 6)])
    assert s.add_start(33.49, 126.49, 90.0) is None           # odom +y로 걸어간 길 = 동쪽
    s.export(force=True)
    assert s.geo_source == "start"
    lat0, lon0 = s.geo.to_latlon(10.0, 5.0)
    assert lat0 == pytest.approx(33.49, abs=1e-7) and lon0 == pytest.approx(126.49, abs=1e-7)
    lat5, lon5 = s.geo.to_latlon(10.0, 10.0)                  # 5 m 더 가면 동쪽 (경도만 커진다)
    assert lat5 == pytest.approx(33.49, abs=1e-6) and lon5 > 126.49
    again = live.LiveStore(tmp_path / "s", {})                # 다시 켜도 stream에서 복원
    again.export(force=True)
    assert again.geo_source == "start" and again.geo == s.geo


def test_sync_clock(monkeypatch):
    """로봇 시계가 1970이면 sudo date로 노트북 시각에 맞추고, 이미 맞으면 그대로 둔다 (ssh는 가짜)."""
    from stepmap import clock
    robot = {"t": 1000.0}
    sent = []

    def fake_ssh(host, cmd, timeout=8.0):
        if cmd.startswith("date"):
            return f"{robot['t']:.3f}"
        sent.append(cmd)
        robot["t"] = float(cmd.split("@")[1].split()[0])
        return ""
    monkeypatch.setattr(clock, "_ssh", fake_ssh)
    monkeypatch.setattr(clock.time, "time", lambda: 1_792_454_400.0)
    r = clock.sync_clock("go2")
    assert r["set"] and abs(r["skew_after"]) < 1 and sent and sent[0].startswith("sudo -n /bin/date -u -s @")
    sent.clear()
    assert not clock.sync_clock("go2")["set"] and not sent


def test_damage_type_subtypes_accepted(tmp_path):
    """2026-11-12: the bridge sends the damage type of the type model as subtype (파손 · 결손 · 이탈·분리)."""
    store = live.LiveStore(tmp_path / "s", {})
    rows = []
    for i, sub in enumerate(("파손", "결손", "이탈·분리")):
        rows.append({"type": "event", "key": f"inspect:t:{i}", "row": {
            "t": 1000.0 + i, "session": "s", "x": float(i), "y": 0.0, "kind": "step", "value": 1.5 + i, "unit": "cm",
            "source": "camera", "leg": "", "confidence": 0.8, "image": "", "note": "test", "subtype": sub}})
    _, err = store.ingest(rows)
    assert err == []
    store.export(force=True)
    csv = (tmp_path / "s" / "events.csv").read_text(encoding="utf-8")
    assert "결손" in csv and "이탈·분리" in csv
    _, err = store.ingest([dict(rows[0], key="inspect:t:9", row=dict(rows[0]["row"], subtype="흔들림"))])
    assert err                                     # a step event cannot be 흔들림


def test_shell_page_swaps_without_reload():
    """2026-11-13: / is a shell page: the dashboard is swapped in from a hidden iframe (no full-page flash on every new
    measurement) and the live camera lives in the shell (the stream is not cut by a swap)."""
    assert '<iframe class="fr" id="stepmap-frame" src="/dashboard.html"' in live.SHELL_PAGE
    assert "window.stepmapShell" in live.SHELL_PAGE and "camera.mjpg" in live.SHELL_PAGE
    assert "if (SHELL) SHELL.swap(window); else location.reload();" in live.LIVE_SCRIPT
    assert "SHELL.toggleCam()" in live.LIVE_SCRIPT
