"""StepMap live: 측정 결과를 받는 즉시 합쳐(merge) 대시보드를 몇 초마다 갱신하고, 다른 컴퓨터로 끊김 없이 복제한다.

  # 로봇(Jetson): 측정 노드 → stepmap_bridge가 POST /api/ingest → 이 서버
  stepmap live -d sessions/<세션> --port 8060
  # 노트북: 로봇 서버에서 끌어와(pull) 같은 대시보드를 띄운다. 네트워크가 끊겨도 다시 연결되면 빠진 것부터 이어받는다
  stepmap live -d sessions/<세션> --port 8060 --pull http://<로봇 주소>:8060
  # 경로가 여럿이면 쉼표로: 끊기면 다음 주소로 자동 전환 (유선 → Go2 WiFi 경유 → 핫스팟 → Tailscale)
  stepmap live -d sessions/<세션> --pull http://192.168.123.18:8060

저장: 받은 항목은 모두 순번(seq)을 붙여 <세션>/stream.jsonl에 한 줄씩 추가하고 디스크에 바로 쓴다 (fsync).
서버를 다시 켜면 stream.jsonl을 다시 읽어 그대로 복원한다. CSV(events·trajectory·foot_traces)는 stream에서
`stepmap merge`와 같은 함수(merge_events, link_foot_traces)로 다시 만든 결과이므로, 같은 입력이면 오프라인 merge와
같은 파일이 나온다.

항목 (JSON, ingest·since 공통):
  {"type": "event", "key": "<출처별 고유 키>", "row": {events.csv 열, id·level 빈칸 가능}}
      같은 key가 다시 오면 덮어쓴다 (측정이 쌓이며 값·사진이 바뀌는 지점)
  {"type": "traj", "row": {"t", "x", "y"}}                 이동 궤적 한 점 (t가 앞 점보다 늦을 때만)
  {"type": "trace", "key": "<걸음 키>", "rows": [foot_traces 열, id 빈칸 + t_event]}
  {"type": "status", "row": {"battery": 72, ...}}          로봇 상태 (CSV에는 넣지 않고 화면용)
  {"type": "file", "path": "img/xx.jpg"}                    사진이 도착했다는 표시 (복제 쪽이 GET /files/로 받아 간다)
  {"type": "gps", "row": {"t", "lat", "lon", "acc", "x", "y"}}  대시보드를 연 기기(휴대폰·노트북)의 GPS 위치와 그 순간의
      로봇 odom 위치. 짝이 쌓이면 odom → 위도·경도 변환(GeoRef)을 GPS로 맞춘다 (gps.py). config/--geo가 있으면 그것이 먼저
  {"type": "start", "row": {"t", "lat", "lon", "bearing", "origin_lat", "origin_lon", "heading"}}  지도에서 고른 출발점
      (GPS가 없을 때): 출발 위치 + 처음 걸어간 방향 → GeoRef. 우선순위 config > 출발점 > GPS

위치: 로봇에는 GPS가 없다. 대시보드의 "내 위치로 보정"을 누르면 브라우저가 기기 위치를 POST /api/gps로 보내고,
로봇 서버가 그때의 odom 위치와 짝지어 stream에 저장한다 (노트북 복제본은 받은 위치를 로봇으로 넘긴다).
브라우저 위치 기능은 https 또는 localhost에서만 되므로 휴대폰은 --https-port(자체 서명 인증서) 주소로 연다.

시계: Jetson은 RTC 배터리가 없어 인터넷이 없으면 1970년부터 다시 센다. 로봇 쪽 시각이 CLOCK_VALID보다 작으면,
노트북이 /api/since?now=<노트북 시각>으로 처음 연결될 때 차이(clock_offset)를 기록하고, CSV·대시보드에는
보정한 시각을 쓴다 (stream에는 원래 값 그대로). 복제본(--pull)은 로봇이 알려 주는 같은 보정값을 쓴다.
보정값은 CLOCK_VALID보다 작은 시각에만 더한다: 세션 중에 로봇 시계를 노트북 시각으로 맞추면(stepmap sync-clock,
LAN 연결) 그 뒤의 시각은 이미 맞으므로 그대로 쓴다. GET /api/time = 이 컴퓨터의 시각 (시계 맞추기용).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .gps import fit_geo, shift_m
from .events import (_NUMERIC_DIGITS, EVENT_COLUMNS, FOOT_TRACE_COLUMNS, FormatError, GeoRef, georef_from_config,
                     link_foot_traces, load_config, merge_events, merge_from_config, thresholds_from_config,
                     validate_events, validate_foot_traces, vworld_key_from_config, write_events, write_foot_traces,
                     write_trajectory)

STREAM = "stream.jsonl"
META = "live_meta.json"
OUT_DIR = "live"                 # <세션>/live/: dashboard.html, report.html, photos/
CLOCK_VALID = 1.6e9              # 2020-09-13. 이보다 작은 시각 = 시계가 맞지 않은 로봇
EXPORT_MIN_S = 2.0               # 내보내기 확인 간격
TRAJ_REFRESH_S = 60.0            # 측정 지점이 그대로이고 궤적만 늘었을 때 대시보드를 다시 만드는 간격
# 대시보드를 다시 만들면 브라우저가 페이지 전체를 새로 읽어 화면이 깜빡인다. 로봇이 움직이는 동안에는 사진·값이
# 계속 바뀌므로 간격을 둔다: 측정 지점 수나 등급이 바뀌면 REDRAW_NEW_S, 값·사진만 바뀌면 REDRAW_UPDATE_S 뒤에.
REDRAW_NEW_S = 3.0
REDRAW_UPDATE_S = 20.0
PULL_PERIOD_S = 1.0
# config.yaml의 geo가 아직 <...>이면 지도에 필요한 위도·경도를 이 기준점으로 만든다 (위치는 출발점 기준 상대값).
# 실제 지도 위치가 필요하면 config.yaml geo 또는 --geo LAT,LON,HEADING
FALLBACK_GEO = GeoRef(33.4860373, 126.4866073, 0.0)
GEO_SHIFT_M = 2.0                # GPS 보정이 이만큼 이상 바뀔 때만 지도를 다시 그린다 (위치가 올 때마다 새로고침하지 않게)
GPS_MIN_GAP_S = 1.0              # 이보다 자주 오는 위치는 버린다
START_DIR_M = 3.0                # 지도에서 출발점을 고를 때 처음 걸어간 방향을 정하는 거리

def _latest(events: list[dict]) -> str | None:
    """새 측정이 들어왔을 때 화면에 띄울 지점: 가장 최근의 위험·주의 지점 (없으면 가장 최근 지점)."""
    pick = [e for e in events if e["level"] in ("danger", "warn")] or events
    return max(pick, key=lambda e: e["t"])["id"] if pick else None


LIVE_SCRIPT = """
<style>
  .sm-live { position: fixed; right: 16px; top: 14px; z-index: 100000; display: flex; gap: 8px; align-items: center;
             font: 600 13px/1 "Pretendard Variable", Pretendard, "Malgun Gothic", system-ui, sans-serif; }
  .sm-live .pill { display: inline-flex; align-items: center; gap: 8px; height: 44px; padding: 0 16px; border: 1px solid rgb(255 255 255 / 0.6);
                   border-radius: 999px; background: rgb(255 255 255 / 0.78); color: #363236; box-shadow: 0 4px 16px rgb(54 50 54 / 0.1);
                   -webkit-backdrop-filter: blur(18px) saturate(1.4); backdrop-filter: blur(18px) saturate(1.4); white-space: nowrap; }
  .sm-live .dot { width: 8px; height: 8px; border-radius: 50%; background: #A5D48C; box-shadow: 0 0 0 4px rgb(165 212 140 / 0.3); }
  .sm-live .off .dot { background: #EF6461; box-shadow: 0 0 0 4px rgb(239 100 97 / 0.25); }
  .sm-live .stale .dot { background: #F7B558; box-shadow: 0 0 0 4px rgb(247 181 88 / 0.3); }   /* 15초 넘게 로봇 데이터 없음 */
  .sm-live .age { color: #75707A; font-weight: 500; font-variant-numeric: tabular-nums; }
  #stepmap-live-t { display: none; }   /* 수신 시각(HH:MM:SS)은 'n초 전'으로 대신한다 (위쪽 바가 좁아 카메라 버튼과 겹쳤다) */
  .sm-live .muted { color: #75707A; font-weight: 500; }
  .sm-live button.pill { cursor: pointer; font: inherit; }
  .sm-live button.pill[aria-pressed="true"] { background: #363236; color: #fff; }
  .sm-live svg { width: 16px; height: 16px; }
  #stepmap-start-btn[aria-pressed="false"] span { display: none; }   /* 고르는 중에만 안내 글자 (평소에는 깃발 아이콘) */
  .sm-cam { display: none; position: fixed; right: 16px; top: 68px; z-index: 100000; width: min(420px, 70vw); overflow: hidden;
            border-radius: 18px; background: #1F1C1F; box-shadow: 0 4px 16px rgb(54 50 54 / 0.2); }
  .sm-cam img { display: block; width: 100%; cursor: zoom-in; }
  .sm-cam.big { width: min(960px, calc(100vw - 32px)); }
  .sm-cam.big img { cursor: zoom-out; }
  .sm-cam span { position: absolute; left: 10px; top: 8px; padding: 4px 10px; border-radius: 999px; background: rgb(31 28 31 / 0.6);
                 color: #fff; font: 600 11px/1.2 "Pretendard Variable", "Malgun Gothic", system-ui, sans-serif; }
  /* 가운데 메뉴와 겹치지 않게: 화면이 넓지 않으면 위치 버튼은 아이콘만 (켜져 있을 때는 상태 글자), 수신 시각은 숨김 */
  @media (max-width: 1700px) { #stepmap-gps-btn[aria-pressed="false"] span { display: none; } .sm-live .muted { display: none; } }
  @media (max-width: 1100px) { .sm-live { top: 68px; } .sm-cam { top: 122px; } .sm-live .muted { display: none; } }
  @media (max-width: 640px) { .sm-live { left: 16px; flex-wrap: wrap; justify-content: flex-end; } .sm-live .pill { height: 36px; padding: 0 12px; } }
</style>
<div class="sm-cam" id="stepmap-live-cam"><img id="stepmap-live-img" alt="실시간 카메라"><span id="stepmap-live-camtxt">실시간 카메라</span></div>
<div class="sm-live">
 <button class="pill" id="stepmap-gps-btn" type="button" aria-pressed="false" title="이 기기의 GPS 위치로 로봇의 실제 위치를 맞춥니다 (기기를 로봇 가까이에 두고)"><svg
  viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path
  d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/></svg><span id="stepmap-gps">위치</span></button>
 <button class="pill" id="stepmap-start-btn" type="button" aria-pressed="false" title="GPS가 없을 때: 지도에서 출발 지점과 처음 걸어간 방향을 눌러 지도 위치를 맞춥니다"><svg
  viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path
  d="M5 21V4"/><path d="M5 4h11l-2 4 2 4H5"/></svg><span id="stepmap-start">출발점 지정</span></button>
 <div class="pill" id="stepmap-live-box"><i class="dot"></i><span id="stepmap-live">실시간 연결 중…</span><span class="age" id="stepmap-live-age"></span><span class="muted" id="stepmap-live-t"></span></div>
 <button class="pill" id="stepmap-live-cambtn" type="button" aria-pressed="false"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
  stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>카메라</button>
</div>
<script>
(function () {
  var box = document.getElementById("stepmap-live-box"), badge = document.getElementById("stepmap-live");
  var when = document.getElementById("stepmap-live-t");
  var loaded = __VERSION__, last = Date.now(), seen = null;
  // 껍데기 페이지 (live.py SHELL_PAGE): 새 판 바꿔 끼우기와 실시간 카메라는 껍데기가 맡는다
  var SHELL = null;
  try { if (window.parent !== window && window.parent.stepmapShell) SHELL = window.parent.stepmapShell; } catch (e) {}
  // 로봇 데이터 신선도: 마지막으로 새 항목(seq 증가: 궤적·상태·측정)이 들어온 뒤 지난 시간. 서 있어도 로봇은 몇 초마다 보낸다
  var ageEl = document.getElementById("stepmap-live-age"), lastSeq = null, seqAt = Date.now();
  function showAge() {
    var s = Math.max(0, Math.round((Date.now() - seqAt) / 1000));
    ageEl.textContent = lastSeq === null ? "" : "· " + (s < 60 ? s + "초 전" : Math.floor(s / 60) + "분 전");
    box.classList.toggle("stale", lastSeq !== null && s > 15 && !box.classList.contains("off"));
  }
  setInterval(showAge, 1000);
  function hhmmss(d) { return d.toTimeString().slice(0, 8); }
  // 새 측정 지점이 들어와 다시 불러올 때: 지점을 고르지 않은 상태면 가장 최근의 위험·주의 지점을 연다
  try {
    var want = sessionStorage.getItem("stepmap-open");
    sessionStorage.removeItem("stepmap-open");
    if (want && /^#?(home)?[/]?$/.test(location.hash)) location.hash = "#home/" + want;
  } catch (e) {}
  // 위치 보정: 이 기기의 GPS 위치를 2초마다 서버로 보낸다 (서버가 그 순간의 로봇 odom 위치와 짝지어 지도 위치를 맞춘다).
  // 브라우저는 https 또는 localhost에서만 위치를 알려 준다 → http로 열었으면 서버의 https 주소로 옮긴다
  var gbtn = document.getElementById("stepmap-gps-btn"), gtxt = document.getElementById("stepmap-gps");
  var gon = false, gwatch = null, glast = 0, gmsg = "", geo = null, httpsPort = null;
  try { gon = sessionStorage.getItem("stepmap-gps") === "1"; } catch (e) {}
  function gpsLabel() {
    var g = geo || {}, t;
    if (gmsg) t = gmsg;
    else if (g.source === "config") t = "위치: 설정값";
    else if (g.source === "start") t = "위치: 지정한 출발점 기준";
    else if (g.source === "gps") t = "GPS 위치" + (g.heading_ok ? "" : " · 방향 확인 중") + (g.acc != null ? " · ±" + Math.round(g.acc) + " m" : "");
    else t = gon ? "GPS 기다리는 중…" : "위치 미확인 · 내 위치로 보정";
    gtxt.textContent = t;
    gbtn.setAttribute("aria-pressed", gon ? "true" : "false");
  }
  function gpsSend(p) {
    if (Date.now() - glast < 2000) return;
    glast = Date.now();
    fetch("api/gps", {method: "POST", headers: {"Content-Type": "application/json"},
                      body: JSON.stringify({lat: p.coords.latitude, lon: p.coords.longitude, acc: p.coords.accuracy})})
      .then(function (r) { return r.json(); })
      .then(function (d) { gmsg = d.ok ? "" : (d.error || "위치 보정 실패"); gpsLabel(); })
      .catch(function () { gmsg = "위치를 보내지 못함"; gpsLabel(); });
  }
  function gpsStart(clicked) {
    if (!window.isSecureContext) {
      if (clicked && httpsPort) {
        location.href = "https://" + location.hostname + ":" + httpsPort + location.pathname + location.hash;
        return;
      }
      gmsg = "위치는 https 또는 localhost 주소에서만";
      return;
    }
    if (!navigator.geolocation) { gmsg = "이 브라우저는 위치 기능 없음"; return; }
    gmsg = "";
    gwatch = navigator.geolocation.watchPosition(gpsSend, function (err) {
      gmsg = err.code === 1 ? "위치 권한이 거부됨" : "위치를 받지 못함";
      gpsLabel();
    }, {enableHighAccuracy: true, maximumAge: 1000, timeout: 20000});
  }
  function gpsStop() {
    if (gwatch !== null) navigator.geolocation.clearWatch(gwatch);
    gwatch = null;
    gmsg = "";
  }
  gbtn.onclick = function () {
    gon = !gon;
    try { sessionStorage.setItem("stepmap-gps", gon ? "1" : "0"); } catch (e) {}
    if (gon) gpsStart(true); else gpsStop();
    gpsLabel();
  };
  if (gon) gpsStart(false);
  gpsLabel();
  // 출발점 지정 (GPS 없이 지도 위치 맞추기): ① 지도에서 로봇이 출발한 곳 ② 처음 걸어간 방향 쪽 한 곳을 누른다.
  // 서버가 odom의 첫 이동 방향과 짝지어 지도 위치를 정한다 (POST api/start). 인터넷 지도(Leaflet)가 있을 때만
  var sbtn = document.getElementById("stepmap-start-btn"), stxt = document.getElementById("stepmap-start");
  var spick = 0, sA = null, sMk = null;
  function startLabel(t) { stxt.textContent = t; sbtn.setAttribute("aria-pressed", spick ? "true" : "false"); }
  function startReset(t) {
    var map = window.stepmapHomeMap;
    spick = 0; sA = null;
    if (sMk && map) map.removeLayer(sMk);
    sMk = null;
    startLabel(t || ((geo || {}).source === "start" ? "출발점 지정됨 · 다시 지정" : "출발점 지정"));
  }
  function bearing(a, b) {
    var r = Math.PI / 180, p1 = a.lat * r, p2 = b.lat * r, dl = (b.lng - a.lng) * r;
    var y = Math.sin(dl) * Math.cos(p2), x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
    return (Math.atan2(y, x) / r + 360) % 360;
  }
  function startClick(ev) {
    var map = window.stepmapHomeMap;
    if (!spick) return;
    if (spick === 1) {
      sA = ev.latlng; spick = 2;
      sMk = window.L ? L.circleMarker(sA, {radius: 9, color: "#363236", weight: 3, fillColor: "#A5D48C", fillOpacity: 1}).addTo(map) : null;
      startLabel("② 처음 걸어간 방향 쪽을 누르세요 (Esc 취소)");
      map.once("click", startClick);
      return;
    }
    var body = {lat: sA.lat, lon: sA.lng, bearing: bearing(sA, ev.latlng)};
    startLabel("출발점 저장 중…");
    fetch("api/start", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)})
      .then(function (r) { return r.json(); })
      .then(function (d) { startReset(d.ok ? "출발점 지정됨 · 지도 다시 그리는 중" : (d.error || "출발점 지정 실패")); })
      .catch(function () { startReset("출발점을 보내지 못함"); });
  }
  sbtn.onclick = function () {
    var map = window.stepmapHomeMap;
    if (spick) { startReset(); return; }
    if (!map) { startLabel("인터넷 지도가 필요합니다 (평면 지도에서는 지정 불가)"); return; }
    if (!/^#?(home)?/.test(location.hash) || /^#(detail|history)/.test(location.hash)) location.hash = "#home";
    spick = 1;
    startLabel("① 지도에서 로봇이 출발한 곳을 누르세요 (Esc 취소)");
    setTimeout(function () { map.once("click", startClick); }, 0);
  };
  document.addEventListener("keydown", function (ev) { if (ev.key === "Escape" && spick) startReset(); });
  function tick() {
    fetch("api/version", {cache: "no-store"}).then(function (r) { return r.json(); }).then(function (v) {
      last = Date.now();
      if (seen === null) seen = v.latest;
      if (v.version > loaded) {
        if (v.latest && v.latest !== seen) { try { sessionStorage.setItem("stepmap-open", v.latest); } catch (e) {} }
        // 껍데기 페이지(/) 안이면: 껍데기가 새 판을 뒤에서 다 그린 뒤 바꿔 끼운다 (깜빡이지 않음). 직접 연 페이지는 예전처럼 새로고침
        if (SHELL) SHELL.swap(window); else location.reload();
        return;
      }
      box.classList.remove("off");
      geo = v.geo || null;
      httpsPort = v.https_port || null;
      gpsLabel();
      sbtn.hidden = (geo || {}).source === "config";
      if (!spick) startReset();
      // 로봇 마커: 다시 그리지 않고 지금 위치로 옮긴다 (대시보드가 window.stepmapMoveRobot을 열어 둔다)
      if (v.robot && window.stepmapMoveRobot) window.stepmapMoveRobot(v.robot);
      badge.textContent = "실시간 · 측정 지점 " + v.events + "개";
      if (v.seq !== lastSeq) { lastSeq = v.seq; seqAt = Date.now(); }
      showAge();
      when.textContent = hhmmss(new Date());
      // 로봇 카드 배터리: 다시 그리지 않고 바로 바꾼다
      var bat = document.getElementById("rc-battery"), batK = document.getElementById("rc-battery-k");
      if (bat && v.battery != null) { bat.textContent = v.battery + "%"; bat.hidden = false; if (batK) batK.hidden = false; }
    }).catch(function () {
      box.classList.add("off");
      badge.textContent = "연결 끊김";
      when.textContent = "마지막 수신 " + hhmmss(new Date(last));
    });
  }
  tick();
  setInterval(tick, 2000);
  // 실시간 카메라: 로봇은 inspect_node가 쓰는 최신 사진, 노트북은 로봇에서 받아 전달한다 (약 3장/초)
  var cam = document.getElementById("stepmap-live-cam"), img = document.getElementById("stepmap-live-img");
  var btn = document.getElementById("stepmap-live-cambtn");
  var txt = document.getElementById("stepmap-live-camtxt"), on = false, busy = false;
  if (SHELL) {          // 카메라 창은 껍데기에 있다 (새 판으로 바꿔도 영상이 끊기지 않는다). 버튼만 여기서 켜고 끈다
    cam.remove();
    var sync = function () { btn.setAttribute("aria-pressed", SHELL.camOn() ? "true" : "false"); };
    btn.onclick = function () { SHELL.toggleCam(); sync(); };
    SHELL.onCam(sync);
    sync();
    return;
  }
  try { on = sessionStorage.getItem("stepmap-cam") === "1"; } catch (e) {}
  function show() { cam.style.display = on ? "block" : "none"; btn.setAttribute("aria-pressed", on ? "true" : "false"); }
  btn.onclick = function () {
    on = !on; show();
    try { sessionStorage.setItem("stepmap-cam", on ? "1" : "0"); } catch (e) {}
  };
  // 연속 전송(camera.mjpg)으로 받는다. 끊기면(로봇 카메라 멈춤·네트워크) 3초 뒤 다시 연결
  var streaming = false;
  function camOpen() {
    if (!on || streaming) return;
    streaming = true;
    txt.textContent = "실시간 카메라 연결 중…";
    img.src = "camera.mjpg?t=" + Date.now();
  }
  function camClose() { streaming = false; img.removeAttribute("src"); }
  img.onload = function () { txt.textContent = "실시간 카메라"; };
  img.onclick = function () { cam.classList.toggle("big"); };   // 누르면 크게 / 다시 누르면 작게
  // 카메라 상태 (1초마다): 로봇 카메라 fps, 촬영 → 사진 지연, 사진이 멈췄는지
  setInterval(function () {
    if (!on) return;
    fetch("api/camera", {cache: "no-store"}).then(function (r) { return r.json(); }).then(function (c) {
      if (c.error) return;
      if (c.age_s != null && c.age_s > 3) { txt.textContent = "카메라 멈춤 · " + Math.round(c.age_s) + "초 전 사진"; return; }
      txt.textContent = "실시간 카메라 · " + Math.round(c.fps || 0) + " fps"
        + (c.latency_ms != null ? " · 지연 " + Math.round(c.latency_ms) + " ms" : "");
    }).catch(function () {});
  }, 1000);
  img.onerror = function () {
    streaming = false;
    txt.textContent = "카메라 없음 (inspect.sh 실행 중인지 확인) · 다시 연결 중";
    setTimeout(camOpen, 3000);
  };
  var showCam = show;
  show = function () { showCam(); if (on) camOpen(); else camClose(); };
  show();
})();
</script>
"""

# 껍데기 페이지 (/): 대시보드를 iframe에 넣고, 새 판이 나오면 보이지 않는 두 번째 iframe에서 다 그린 뒤(지도 타일까지)
# 부드럽게 바꿔 끼운다 → 측정이 들어올 때마다 화면 전체가 새로고침되며 깜빡이던 것을 없앤다 (데이터·대시보드 파일은 그대로 저장).
# 실시간 카메라 창도 껍데기에 두어 판을 바꿔도 영상이 끊기지 않는다. /dashboard.html을 직접 열면 예전처럼 동작한다.
SHELL_PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><title>StepMap 실시간</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #ECEBEA; }
  .fr { position: fixed; inset: 0; width: 100%; height: 100%; border: 0; background: #ECEBEA; transition: opacity 0.35s ease; }
  .fr.pre { opacity: 0; pointer-events: none; }
  /* 실시간 카메라 카드: 대시보드 유리 카드와 같은 모양 (흰 반투명 + 블러, 둥근 모서리 18px, 차콜 글자) */
  .smc { position: fixed; right: 16px; top: 72px; z-index: 20; width: min(360px, calc(100vw - 32px)); box-sizing: border-box;
         padding: 10px; border-radius: 18px; background: rgb(255 255 255 / 0.82); border: 1px solid rgb(255 255 255 / 0.7);
         box-shadow: 0 6px 24px rgb(54 50 54 / 0.16); -webkit-backdrop-filter: blur(18px) saturate(1.4); backdrop-filter: blur(18px) saturate(1.4);
         font: 600 13px/1.3 "Pretendard Variable", Pretendard, "Malgun Gothic", system-ui, sans-serif; color: #363236;
         transition: width 0.25s ease, top 0.25s ease, right 0.25s ease; }
  .smc[hidden] { display: none; }
  .smc .hd { display: flex; align-items: center; gap: 8px; padding: 2px 2px 8px; }
  .smc .live { display: inline-flex; align-items: center; gap: 5px; padding: 3px 8px; border-radius: 999px; background: #363236; color: #fff;
               font-size: 11px; letter-spacing: 0.02em; }
  .smc .live i { width: 6px; height: 6px; border-radius: 50%; background: #EF6461; animation: smc-blink 1.6s ease-in-out infinite; }
  .smc.stale .live i, .smc.off .live i { animation: none; background: #F7B558; }
  .smc.off .live i { background: #9A959C; }
  @keyframes smc-blink { 50% { opacity: 0.35; } }
  .smc .tt { flex: 1; min-width: 0; }
  .smc .meta { color: #75707A; font-weight: 500; font-size: 12px; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .smc button { width: 30px; height: 30px; border-radius: 50%; border: 1px solid rgb(54 50 54 / 0.12); background: #fff; color: #363236;
                display: inline-grid; place-items: center; cursor: pointer; padding: 0; flex: none; }
  .smc button:hover { background: #F4F3F2; }
  .smc button svg { width: 15px; height: 15px; }
  /* the live picture keeps its own shape (camera + info panel below it, e.g. 640x676): no crop */
  .smc .pic { position: relative; min-height: 180px; border-radius: 12px; overflow: hidden; background: #1F1C1F; }
  .smc .pic img { display: block; width: 100%; height: auto; max-height: calc(100vh - 170px); object-fit: contain; }
  .smc .pic .msg[hidden] { display: none; }
  .smc .pic .msg { position: absolute; inset: 0; display: grid; place-items: center; padding: 16px; text-align: center; color: #D9D6DA;
                   font-weight: 500; font-size: 12.5px; }
  .smc.big { width: min(960px, calc(100vw - 32px)); right: 50%; transform: translateX(50%); top: 80px; z-index: 30; }
  .smc-dim { position: fixed; inset: 0; z-index: 25; background: rgb(31 28 31 / 0.35); }
  .smc-dim[hidden] { display: none; }
  @media (max-width: 1100px) { .smc { top: 122px; } }
  @media (max-width: 640px) { .smc { top: auto; bottom: 16px; } .smc.big { top: 16px; bottom: auto; } }
</style></head><body>
<iframe class="fr" id="stepmap-frame" src="/dashboard.html" title="StepMap 대시보드" allow="geolocation"></iframe>
<div class="smc-dim" id="smc-dim" hidden></div>
<section class="smc" id="smc" aria-label="실시간 카메라" hidden>
  <div class="hd"><span class="live"><i></i>LIVE</span><div class="tt">실시간 카메라<div class="meta" id="smc-meta">연결 중…</div></div>
    <button type="button" id="smc-big" title="크게 보기 / 작게 보기" aria-label="크게 보기"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/></svg></button>
    <button type="button" id="smc-close" title="닫기" aria-label="카메라 닫기"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg></button></div>
  <div class="pic"><img id="smc-img" alt="로봇 실시간 카메라"><div class="msg" id="smc-msg">카메라 연결 중…</div></div>
</section>
<script>
(function () {
  var cur = document.getElementById("stepmap-frame"), pending = false, camListeners = [];
  var card = document.getElementById("smc"), img = document.getElementById("smc-img"), msg = document.getElementById("smc-msg");
  var meta = document.getElementById("smc-meta"), dim = document.getElementById("smc-dim"), on = false, streaming = false;
  try { on = sessionStorage.getItem("stepmap-cam") === "1"; } catch (e) {}
  function ready(f) {            // 새 판이 다 그려졌는지: 지도 타일이 한 장이라도 왔거나, 평면 지도이거나, 홈 화면이 아님
    try {
      var w = f.contentWindow, s = w.stepmap;
      if (!w.document || w.document.readyState !== "complete") return false;
      if (!s) return true;                                   // 기다림 페이지 등
      return s.mode === "plane" || (s.tiles && s.tiles.loaded > 0) || (s.page && s.page !== "home");
    } catch (e) { return true; }
  }
  window.stepmapShell = {
    swap: function (from) {
      if (pending || from !== cur.contentWindow) return;     // 지금 보이는 판이 부를 때만, 한 번에 하나
      pending = true;
      var w = cur.contentWindow, hash = "";
      try {
        hash = w.location.hash || "";
        if (w.stepmapUserMoved && w.stepmapHomeMap) {        // 사용자가 옮긴 지도 위치를 새 판에 넘긴다
          var c = w.stepmapHomeMap.getCenter();
          sessionStorage.setItem("stepmap-view", JSON.stringify({ lat: c.lat, lng: c.lng, z: w.stepmapHomeMap.getZoom(), at: Date.now() }));
        }
      } catch (e) {}
      var nf = document.createElement("iframe");
      nf.className = "fr pre";
      nf.title = cur.title;
      nf.setAttribute("allow", "geolocation");
      nf.src = "/dashboard.html?v=" + Date.now() + hash;
      document.body.insertBefore(nf, cur.nextSibling);       // 위에 겹쳐 두고(투명) 다 그려지면 나타나게
      var t0 = Date.now();
      (function wait() {
        if (!ready(nf) && Date.now() - t0 < 6000) { setTimeout(wait, 120); return; }
        setTimeout(function () {
          nf.classList.remove("pre");
          setTimeout(function () { cur.remove(); cur = nf; nf.id = "stepmap-frame"; pending = false; }, 380);
        }, 200);
      })();
    },
    camOn: function () { return on; },
    toggleCam: function () { on = !on; showCam(); },
    onCam: function (fn) { camListeners.push(fn); }
  };
  function notify() {
    camListeners = camListeners.filter(function (fn) { try { fn(); return true; } catch (e) { return false; } });
  }
  function camOpen() { if (!streaming) { streaming = true; msg.hidden = false; msg.textContent = "카메라 연결 중…"; img.src = "camera.mjpg?t=" + Date.now(); } }
  function camClose() { streaming = false; img.removeAttribute("src"); }
  function showCam() {
    card.hidden = !on;
    if (!on) { card.classList.remove("big"); dim.hidden = true; }
    try { sessionStorage.setItem("stepmap-cam", on ? "1" : "0"); } catch (e) {}
    if (on) camOpen(); else camClose();
    notify();
  }
  img.onload = function () { msg.hidden = true; };
  img.onerror = function () {
    streaming = false;
    msg.hidden = false;
    msg.textContent = "카메라 영상이 없습니다 (로봇 측정 노드 확인) · 다시 연결 중";
    card.classList.add("off");
    setTimeout(function () { if (on) camOpen(); }, 3000);
  };
  function toggleBig() { if (!on) return; var b = card.classList.toggle("big"); dim.hidden = !b; }
  img.onclick = toggleBig;
  document.getElementById("smc-big").onclick = toggleBig;
  dim.onclick = toggleBig;
  document.getElementById("smc-close").onclick = function () { on = false; showCam(); };
  document.addEventListener("keydown", function (ev) { if (ev.key === "Escape" && card.classList.contains("big")) toggleBig(); });
  // 카메라 상태 (1초마다): fps · 촬영→화면 지연 · 멈춤
  setInterval(function () {
    if (!on) return;
    fetch("api/camera", { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (c) {
      if (c.error) { meta.textContent = "카메라 정보 없음"; return; }
      var stale = c.age_s != null && c.age_s > 3;
      card.classList.toggle("stale", stale);
      card.classList.remove("off");
      if (!stale && streaming) msg.hidden = true;       // MJPEG는 브라우저가 load를 알려 주지 않을 때가 있다: 새 사진이 오면 안내 글자를 치운다
      meta.textContent = stale ? "영상 멈춤 · " + Math.round(c.age_s) + "초 전 사진"
        : Math.round(c.fps || 0) + " fps" + (c.latency_ms != null ? " · 지연 " + Math.round(c.latency_ms) + " ms" : "");
    }).catch(function () { meta.textContent = "로봇 서버 연결 끊김"; card.classList.add("off"); });
  }, 1000);
  showCam();
})();
</script></body></html>"""

WAITING_PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>StepMap 실시간</title></head>
<body style="margin:0;font:15px/1.6 system-ui,sans-serif;background:#f6f6f4;color:#333">
<div style="max-width:560px;margin:18vh auto;padding:0 20px;text-align:center">
<h1 style="font-size:22px">측정 데이터를 기다리는 중</h1>
<p>로봇에서 첫 이동 궤적이나 측정 지점이 도착하면 대시보드가 자동으로 열립니다.<br>
오래 걸리면 로봇의 odom(/utlidar/robot_odom)과 stepmap_bridge가 실행 중인지 확인하세요.</p></div>
""" + LIVE_SCRIPT + "</body></html>"


def _norm_event(row: dict) -> dict:
    """JSON으로 받은 행을 events.py의 내부 형식(숫자 열은 float/None, 나머지는 문자열)으로 맞춘다."""
    out = {}
    for c in EVENT_COLUMNS:
        v = row.get(c)
        if c in _NUMERIC_DIGITS:
            out[c] = None if v is None or v == "" else (float(v) if isinstance(v, (int, float)) else v)
        else:
            out[c] = "" if v is None else str(v)
    return out


def _norm_trace(row: dict) -> dict:
    out = {}
    for c in FOOT_TRACE_COLUMNS:
        v = row.get(c)
        if c in ("id", "leg"):
            out[c] = "" if v is None else str(v)
        else:
            out[c] = None if v is None or v == "" else (float(v) if isinstance(v, (int, float)) else v)
    return out


class LiveStore:
    """세션 하나의 실시간 저장소: stream.jsonl (추가 전용) + 메모리 상태 + CSV·대시보드 내보내기."""

    def __init__(self, folder: str | Path, cfg: dict | None = None, geo: GeoRef | None = None):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or {}
        self.geo = geo or georef_from_config(self.cfg)
        self.geo_is_fallback = self.geo is None
        self.geo_source = "none" if self.geo is None else "config"   # "config" | "gps" | "none"(기준점, 위치 미확인)
        self.geo = self.geo or FALLBACK_GEO
        self.gps: list[dict] = []                 # GPS 위치 + 그때의 odom 위치
        self.gps_heading_ok = False               # 방위각까지 GPS로 정했는지 (로봇이 MIN_SPREAD_M 이상 움직인 뒤)
        self.forward_gps = False                  # 복제본(--pull): 받은 위치·출발점을 로봇 서버로 넘긴다
        self.start: dict | None = None            # 지도에서 고른 출발점 (GPS가 없을 때의 위치)
        self.https_port: int | None = None
        self.lock = threading.RLock()
        self.seq = 0
        self.events: dict[str, dict] = {}
        self.traj: list[dict] = []
        self.traces: dict[str, list[dict]] = {}
        self.status: dict = {}
        self.items: list[dict] = []          # 모든 항목 (seq 순), since()용
        self.meta = {"clock_offset": 0.0, "pull_seq": 0}
        if (self.dir / META).is_file():
            self.meta.update(json.loads((self.dir / META).read_text(encoding="utf-8")))
        self.version = 0                      # 내보낸 결과가 바뀔 때마다 증가 (대시보드 새로고침 기준)
        self.robot_photo: Path | None = None  # 로봇 카드 사진 (serve가 config 옆 data/robot.png를 넣는다)
        self.summary = {"events": 0, "danger": 0, "warn": 0, "latest": None}
        self.dirty = False
        self.pull_state: dict = {}                # --pull: 지금 받는 주소와 상태
        self.camera: str | None = None            # 실시간 카메라 사진 파일 (로봇)
        self._sig = None                      # 마지막으로 그린 측정 지점·발끝 기록·상태
        self.file_seq = 0                     # 마지막 사진 항목의 seq (사진만 바뀌어도 다시 그리기)
        self._drawn_at = 0.0
        self._head = None                     # 마지막으로 그린 측정 지점 수·등급 (다시 그리기 간격 선택)
        self._replay()

    # ---- 저장 ----
    def _replay(self):
        path = self.dir / STREAM
        if not path.is_file():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._apply(json.loads(line))
        self.dirty = bool(self.items)

    def _save_meta(self):
        tmp = self.dir / (META + ".tmp")
        tmp.write_text(json.dumps(self.meta), encoding="utf-8")
        os.replace(tmp, self.dir / META)

    def _apply(self, item: dict):
        self.seq = max(self.seq, int(item["seq"]))
        self.items.append(item)
        kind = item["type"]
        if kind == "event":
            self.events[item["key"]] = _norm_event(item["row"])
        elif kind == "traj":
            r = item["row"]
            if not self.traj or float(r["t"]) > self.traj[-1]["t"]:   # 시간순만 받는다
                self.traj.append({"t": float(r["t"]), "x": float(r["x"]), "y": float(r["y"])})
        elif kind == "trace":
            self.traces[item["key"]] = [_norm_trace(r) for r in item["rows"]]
        elif kind == "status":
            self.status.update(item["row"])
        elif kind == "file":                       # 사진이 바뀌면 (같은 이름이어도) 대시보드를 다시 그린다
            self.file_seq = int(item["seq"])
        elif kind == "gps":
            self.gps.append({k: float(item["row"][k]) if item["row"].get(k) is not None else None
                             for k in ("t", "lat", "lon", "acc", "x", "y")})
        elif kind == "start":
            self.start = {k: float(v) for k, v in item["row"].items()}

    def check(self, item: dict) -> list[str]:
        """받기 전에 형식 검사 (잘못된 항목이 stream에 들어가 내보내기를 깨지 않게)."""
        kind = item.get("type")
        if kind == "event":
            if not item.get("key"):
                return ["event: key 필요"]
            row = _norm_event(item.get("row") or {})
            return validate_events([row], raw=True)
        if kind == "traj":
            r = item.get("row") or {}
            return [] if all(isinstance(r.get(c), (int, float)) for c in ("t", "x", "y")) else ["traj: t, x, y 숫자 필요"]
        if kind == "trace":
            if not item.get("key") or not item.get("rows"):
                return ["trace: key, rows 필요"]
            return validate_foot_traces([_norm_trace(r) for r in item["rows"]], raw=True)
        if kind in ("status", "file"):
            return []
        if kind == "gps":
            r = item.get("row") or {}
            if not all(isinstance(r.get(c), (int, float)) for c in ("t", "lat", "lon", "x", "y")):
                return ["gps: t, lat, lon, x, y 숫자 필요"]
            if not (-90 <= r["lat"] <= 90 and -180 <= r["lon"] <= 180):
                return ["gps: 위도·경도 범위 밖"]
            return []
        if kind == "start":
            r = item.get("row") or {}
            if not all(isinstance(r.get(c), (int, float)) for c in ("lat", "lon", "bearing", "origin_lat", "origin_lon", "heading")):
                return ["start: lat, lon, bearing, origin_lat, origin_lon, heading 숫자 필요"]
            if not (-90 <= r["lat"] <= 90 and -180 <= r["lon"] <= 180):
                return ["start: 위도·경도 범위 밖"]
            return []
        return [f"알 수 없는 type {kind!r}"]

    def ingest(self, items: list[dict], keep_src_seq: bool = False) -> tuple[int, list[str]]:
        """항목을 검사·저장한다. 잘못된 항목은 건너뛰고 오류를 돌려준다. 반환: (마지막 seq, 오류 목록)."""
        errors = []
        with self.lock:
            lines = []
            for item in items:
                if item.get("type") == "event" and isinstance(item.get("row"), dict) and not item["row"].get("session"):
                    item = {**item, "row": {**item["row"], "session": self.dir.name}}
                e = self.check(item)
                if e:
                    errors += e
                    continue
                self.seq += 1
                rec = {k: v for k, v in item.items() if k != "seq"}
                rec["seq"] = self.seq
                if keep_src_seq and "seq" in item:
                    rec["src_seq"] = item["seq"]
                lines.append(json.dumps(rec, ensure_ascii=False))
                self._apply(rec)
            if lines:
                with open(self.dir / STREAM, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                self.dirty = True
            return self.seq, errors

    def add_gps(self, lat: float, lon: float, acc: float | None) -> str | None:
        """로봇 서버: 기기 GPS 위치를 지금 로봇 odom 위치(마지막 궤적 점)와 짝지어 저장. 문제가 있으면 이유를 돌려준다."""
        with self.lock:
            if not self.traj:
                return "로봇 위치(odom)가 아직 없습니다"
            now = time.time()
            if self.gps and now - self.gps[-1]["t"] < GPS_MIN_GAP_S:
                return None
            p = self.traj[-1]
            row = {"t": round(now, 2), "lat": lat, "lon": lon, "acc": None if acc is None else round(acc, 1),
                   "x": p["x"], "y": p["y"]}
        _, errors = self.ingest([{"type": "gps", "row": row}])
        return errors[0] if errors else None

    def add_start(self, lat: float, lon: float, bearing: float) -> str | None:
        """지도에서 고른 출발점: 세션 첫 궤적 점 = (lat, lon), 처음 걸어간 방향 = bearing(북=0, 시계방향).
        odom의 첫 이동 방향(첫 점에서 START_DIR_M 이상 떨어진 점)과 짝지어 GeoRef를 만든다."""
        with self.lock:
            if not self.traj:
                return "로봇 위치(odom)가 아직 없습니다"
            p0 = self.traj[0]
            far = next((p for p in self.traj if math.hypot(p["x"] - p0["x"], p["y"] - p0["y"]) >= START_DIR_M), None)
            if far is None:
                return f"로봇이 {START_DIR_M:g} m 이상 움직인 뒤에 지정할 수 있습니다 (처음 걸어간 방향이 필요)"
            a = math.degrees(math.atan2(far["y"] - p0["y"], far["x"] - p0["x"]))  # odom 각도 (x축에서 반시계)
            heading = (bearing + a) % 360.0                                       # odom x축의 방위각
            origin = GeoRef(lat, lon, heading).to_latlon(-p0["x"], -p0["y"])      # odom (0, 0)의 위도·경도
            row = {"t": round(time.time(), 2), "lat": lat, "lon": lon, "bearing": round(bearing, 2),
                   "origin_lat": round(origin[0], 8), "origin_lon": round(origin[1], 8), "heading": round(heading, 3)}
        _, errors = self.ingest([{"type": "start", "row": row}])
        return errors[0] if errors else None

    def _update_geo(self):
        """출발점·GPS로 GeoRef를 다시 맞춘다 (config/--geo가 있으면 그대로, 출발점이 GPS보다 먼저).
        크게 바뀌면 지도를 다시 그리게 한다."""
        if self.geo_source == "config":
            return
        if self.start:
            new = GeoRef(self.start["origin_lat"], self.start["origin_lon"], self.start["heading"])
            if self.geo_source != "start" or new != self.geo:
                self.geo, self.geo_source, self.geo_is_fallback = new, "start", False
                self._sig = None
            return
        if not self.gps:
            return
        res = fit_geo(self.gps, self.geo.heading_deg if self.gps_heading_ok else None)
        if res is None:
            return
        new, heading_ok = res
        heading_ok = heading_ok or self.gps_heading_ok
        pts = [(0.0, 0.0)] + [(p["x"], p["y"]) for p in self.traj[:1] + self.traj[-1:]] +               [(e["x"], e["y"]) for e in self.events.values() if e.get("x") is not None]
        if self.geo_source != "gps" or heading_ok != self.gps_heading_ok or shift_m(self.geo, new, pts) > GEO_SHIFT_M:
            self.geo, self.geo_source, self.gps_heading_ok = new, "gps", heading_ok
            self.geo_is_fallback = False
            self._sig = None                     # 위치가 바뀌었으니 대시보드를 다시 그린다

    def robot_latlon(self) -> list[float] | None:
        with self.lock:
            if not self.traj:
                return None
            return [round(v, 7) for v in self.geo.to_latlon(self.traj[-1]["x"], self.traj[-1]["y"])]

    def since(self, seq: int, limit: int = 2000) -> list[dict]:
        with self.lock:
            lo, hi = 0, len(self.items)
            while lo < hi:                      # items는 seq 순
                mid = (lo + hi) // 2
                if self.items[mid]["seq"] <= seq:
                    lo = mid + 1
                else:
                    hi = mid
            return self.items[lo:lo + limit]

    def note_remote_clock(self, now: float):
        """복제하는 쪽 시각으로 로봇 시계 보정값을 정한다 (로봇 시계가 1970이고 아직 보정값이 없을 때 한 번)."""
        mine = time.time()
        with self.lock:
            if mine < CLOCK_VALID and now > CLOCK_VALID and not self.meta.get("clock_offset"):
                self.meta["clock_offset"] = now - mine
                self._save_meta()
                self.dirty = True

    # ---- 내보내기 ----
    def _rows(self):
        self._update_geo()
        off = float(self.meta.get("clock_offset") or 0.0)
        session = self.dir.name
        events = []
        fix = lambda t: t + off if t < CLOCK_VALID else t   # 시계를 맞춘 뒤의 시각은 그대로
        for r in self.events.values():
            r = dict(r)
            r["t"] = fix(r["t"])
            r["session"] = r["session"] or session
            events.append(r)
        traj = [{"t": fix(p["t"]), "x": p["x"], "y": p["y"], "lat": None, "lon": None} for p in self.traj]
        for p in traj:
            p["lat"], p["lon"] = self.geo.to_latlon(p["x"], p["y"])
        traces = []
        for rows in self.traces.values():
            for r in rows:
                r = dict(r)
                if r.get("t_event") is not None:
                    r["t_event"] = fix(r["t_event"])
                traces.append(r)
        return events, traj, traces

    def export(self, force: bool = False) -> bool:
        """events.csv·trajectory.csv·foot_traces.csv를 다시 쓰고, 측정 지점이 바뀌었으면(또는 궤적만 바뀐 지
        TRAJ_REFRESH_S가 지났으면) 대시보드도 다시 만든다. 대시보드를 새로 만들었으면 True."""
        with self.lock:
            if not self.dirty and not force:
                return False
            self.dirty = False
            raw_events, traj, traces = self._rows()
            status = dict(self.status)
        try:
            return self._export(raw_events, traj, traces, status, force)
        except Exception:
            with self.lock:
                self.dirty = True                # 다음 주기에 다시 시도
            raise

    def _export(self, raw_events, traj, traces, status, force) -> bool:
        from .dashboard import write_dashboard
        th = thresholds_from_config(self.cfg)
        events, _ = merge_events(raw_events, th, merge_from_config(self.cfg), self.geo)
        write_events(self.dir / "events.csv", events)
        if not traj:                             # 대시보드는 궤적이 한 점 이상 있어야 그린다
            traj = [{"t": e["t"], "x": e["x"], "y": e["y"], "lat": e["lat"], "lon": e["lon"]} for e in events[:1]]
        write_trajectory(self.dir / "trajectory.csv", traj)
        linked, _ = link_foot_traces(traces, events) if traces else ([], [])
        trace_path = self.dir / "foot_traces.csv"
        if linked:
            write_foot_traces(trace_path, linked)
        elif trace_path.exists():
            trace_path.unlink()
        out = self.dir / OUT_DIR
        sig = json.dumps([events, linked, self.file_seq], sort_keys=True, default=str)
        since = time.time() - self._drawn_at
        stale = since > TRAJ_REFRESH_S
        if not force and sig == self._sig and not stale:
            return False
        head = json.dumps(sorted(str(e.get("level")) for e in events))   # 지점 수·등급
        if not force and sig != self._sig:
            gap = REDRAW_NEW_S if head != self._head else REDRAW_UPDATE_S
            if since < gap:                      # 아직 이르다: 다음 주기에 다시 확인 (바뀐 내용은 남아 있다)
                with self.lock:
                    self.dirty = True
                return False
        self._sig, self._head, self._drawn_at = sig, head, time.time()
        if self.robot_photo and self.robot_photo.is_file() and not (self.dir / "robot.png").exists():
            shutil.copyfile(self.robot_photo, self.dir / "robot.png")   # write_dashboard는 events.csv 옆에서 찾는다
        if events or traj:
            meta = {**_meta(self.cfg, status), "geo_source": self.geo_source}
            dash = write_dashboard(self.dir / "events.csv", self.dir / "trajectory.csv", out / "dashboard.html",
                                   th, meta, trace_path if linked else None)
            with self.lock:
                self.version += 1
                page = dash.read_text(encoding="utf-8")
                page = page.replace("</body>", LIVE_SCRIPT.replace("__VERSION__", str(self.version)) + "</body>", 1)
                tmp = dash.with_suffix(".tmp")
                tmp.write_text(page, encoding="utf-8")
                os.replace(tmp, dash)
        with self.lock:
            self.summary = {"events": len(events), "danger": sum(e["level"] == "danger" for e in events),
                            "warn": sum(e["level"] == "warn" for e in events), "latest": _latest(events)}
        return True

    def report(self) -> Path:
        from .report import write_report
        self.export(force=True)
        out = self.dir / OUT_DIR / "report.html"
        write_report(self.dir / "events.csv", self.dir / "trajectory.csv", out, thresholds_from_config(self.cfg),
                     str((self.cfg.get("report") or {}).get("area") or ""))
        self.export(force=True)                  # 대시보드에 "신고서 열기" 버튼을 넣으려면 다시 만든다
        return out


def _meta(cfg: dict, status: dict) -> dict:
    robot = cfg.get("robot") or {}
    # live: only the battery the robot reports (bridge status). config robot.battery is a sample value for static /
    # demo dashboards and must not look like a measurement here; no report yet -> the line stays hidden.
    battery = status.get("battery")
    return {"vworld_key": vworld_key_from_config(cfg),
            "area": str((cfg.get("report") or {}).get("area") or ""),
            "robot": str(robot.get("name") or ""),
            "model": str(robot.get("model") or ""),
            "sensors": [str(x) for x in robot.get("sensors") or []],
            "battery": battery}


# ---------------------------------------------------------------------------
# HTTP 서버
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    store: LiveStore = None  # type: ignore[assignment]

    def log_message(self, fmt, *args):   # 요청마다 찍지 않는다
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _file(self, path: Path):
        if not path.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = {".html": "text/html; charset=utf-8", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                 ".csv": "text/csv; charset=utf-8", ".json": "application/json"}.get(path.suffix.lower(),
                                                                                   "application/octet-stream")
        self._send(200, path.read_bytes(), ctype)

    def _safe(self, base: Path, rel: str) -> Path | None:
        p = (base / urllib.parse.unquote(rel)).resolve()
        return p if str(p).startswith(str(base.resolve())) else None

    def do_GET(self):
        s = self.store
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            # 껍데기 페이지: 새 판을 깜빡임 없이 바꿔 끼우고 카메라 창을 유지한다 (/dashboard.html은 예전처럼 직접 열 수 있다)
            self._send(200, SHELL_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif u.path == "/api/version":
            with s.lock:
                self._json({"version": s.version, "seq": s.seq, **s.summary, "battery": s.status.get("battery"),
                            "status": s.status, "clock_offset": s.meta.get("clock_offset"),
                            "geo_fallback": s.geo_is_fallback, "pull": s.pull_state,
                            "geo": {"source": s.geo_source, "fixes": len(s.gps), "heading_ok": s.gps_heading_ok,
                                    "acc": s.gps[-1]["acc"] if s.gps else None},
                            "https_port": s.https_port, "robot": s.robot_latlon()})
        elif u.path == "/api/time":                 # 이 컴퓨터의 시각 (로봇 시계 맞추기)
            self._json({"t": time.time()})
        elif u.path == "/api/since":
            if "now" in q:
                s.note_remote_clock(float(q["now"][0]))
            items = s.since(int(q.get("seq", ["0"])[0]))
            self._json({"items": items, "last": s.seq, "clock_offset": s.meta.get("clock_offset") or 0.0})
        elif u.path == "/report":
            try:
                s.report()
                self.send_response(302)
                self.send_header("Location", "/report.html")
                self.end_headers()
            except (FormatError, ValueError) as e:
                self._send(500, str(e).encode("utf-8"), "text/plain; charset=utf-8")
        elif u.path == "/camera.jpg":
            self._camera()
        elif u.path == "/camera.mjpg":
            self._camera_stream()
        elif u.path == "/api/camera":
            self._camera_info()
        elif u.path == "/dashboard.html" and not (s.dir / OUT_DIR / "dashboard.html").is_file():
            self._send(200, WAITING_PAGE.replace("__VERSION__", "0").encode("utf-8"), "text/html; charset=utf-8")
        elif u.path.startswith("/files/"):
            p = self._safe(s.dir, u.path[len("/files/"):])
            self._file(p) if p else self._send(403, b"forbidden", "text/plain")
        else:
            p = self._safe(s.dir / OUT_DIR, u.path.lstrip("/"))
            self._file(p) if p else self._send(403, b"forbidden", "text/plain")

    def _camera(self):
        """로봇: --camera 파일(inspect_node의 최신 사진). 노트북(--pull): 지금 연결된 로봇에서 받아 전달."""
        s = self.store
        if s.camera and Path(s.camera).is_file():
            self._send(200, Path(s.camera).read_bytes(), "image/jpeg")
            return
        url = (s.pull_state or {}).get("url")
        if url and s.pull_state.get("ok"):
            try:
                with urllib.request.urlopen(f"{url}/camera.jpg", timeout=2) as r:
                    self._send(200, r.read(), "image/jpeg")
                return
            except (urllib.error.URLError, OSError):
                pass
        self._send(404, b"no camera", "text/plain")

    def _camera_info(self):
        """카메라 상태: 로봇은 surface_seg_node가 사진과 함께 쓰는 <사진>.json (fps, 촬영→사진 지연),
        노트북은 로봇에 물어 그대로 전달한다. age_s = 마지막 사진이 몇 초 전인지 (멈춤 확인)."""
        s = self.store
        if s.camera:
            meta = Path(s.camera).with_suffix(".json")
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
                d["age_s"] = round(time.time() - float(d.get("written") or 0), 2)
                self._json(d)
            except (OSError, ValueError):
                self._json({"error": "no camera info"}, 404)
            return
        url = (s.pull_state or {}).get("url")
        if url and s.pull_state.get("ok"):
            try:
                with urllib.request.urlopen(f"{url}/api/camera", timeout=2) as r:
                    self._send(200, r.read())
                return
            except (urllib.error.URLError, OSError):
                pass
        self._json({"error": "no camera info"}, 404)

    def _camera_stream(self):
        """실시간 카메라 연속 전송 (MJPEG). 로봇: 카메라 사진 파일이 바뀔 때마다 바로 보낸다 (사진마다 요청하는 것보다
        빠르고 지연이 적다). 노트북(--pull): 로봇의 스트림을 그대로 이어 준다."""
        s = self.store
        url = (s.pull_state or {}).get("url")
        local = s.camera and Path(s.camera).is_file()
        if not local and not (url and s.pull_state.get("ok")):
            self._send(404, b"no camera", "text/plain")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            if local:
                cam, last, idle = Path(s.camera), 0.0, 0.0
                while idle < CAMERA_IDLE_S:
                    try:
                        m = cam.stat().st_mtime
                        if m != last:
                            last, idle = m, 0.0
                            data = cam.read_bytes()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                             % len(data) + data + b"\r\n")
                            self.wfile.flush()
                    except FileNotFoundError:
                        pass
                    time.sleep(CAMERA_POLL_S)
                    idle += CAMERA_POLL_S
            else:
                with urllib.request.urlopen(f"{url}/camera.mjpg", timeout=CAMERA_IDLE_S) as r:
                    while True:
                        chunk = r.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
        except (OSError, urllib.error.URLError):     # 브라우저가 닫히거나 로봇과 끊김
            pass

    def do_POST(self):
        s = self.store
        u = urllib.parse.urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if u.path == "/api/ingest":
            try:
                items = json.loads(body.decode("utf-8"))["items"]
            except (ValueError, KeyError) as e:
                self._json({"ok": False, "errors": [f"JSON: {e}"]}, 400)
                return
            seq, errors = s.ingest(items)
            self._json({"ok": not errors, "seq": seq, "errors": errors[:20]}, 200 if not errors else 207)
        elif u.path == "/api/file":
            rel = urllib.parse.parse_qs(u.query).get("path", [""])[0]
            p = self._safe(s.dir, rel)
            if not p or not rel:
                self._send(403, b"forbidden", "text/plain")
                return
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".part")
            tmp.write_bytes(body)
            os.replace(tmp, p)
            s.ingest([{"type": "file", "path": rel}])
            self._json({"ok": True})
        elif u.path == "/api/gps":
            self._gps(body)
        elif u.path == "/api/start":
            self._start(body)
        else:
            self._send(404, b"not found", "text/plain")


    def _forward(self, path: str, body: bytes):
        """복제본(--pull): 받은 요청을 지금 연결된 로봇 서버로 넘기고 답을 그대로 돌려준다 (로봇 stream에 저장 → pull)."""
        s = self.store
        url = (s.pull_state or {}).get("url")
        if not url or not s.pull_state.get("ok"):
            self._json({"ok": False, "error": "로봇 서버와 연결되지 않았습니다"}, 502)
            return
        req = urllib.request.Request(f"{url}{path}", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=4) as r:
                self._send(r.status, r.read())
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())
        except (urllib.error.URLError, OSError) as e:
            self._json({"ok": False, "error": f"로봇 서버: {e}"}, 502)

    def _start(self, body: bytes):
        """지도에서 고른 출발점 {lat, lon, bearing}. 로봇 서버는 저장, 복제본은 로봇 서버로 넘긴다."""
        s = self.store
        try:
            d = json.loads(body.decode("utf-8"))
            lat, lon, bearing = float(d["lat"]), float(d["lon"]), float(d["bearing"]) % 360.0
        except (ValueError, KeyError, TypeError) as e:
            self._json({"ok": False, "error": f"JSON: {e}"}, 400)
            return
        if s.forward_gps:
            self._forward("/api/start", body)
            return
        err = s.add_start(lat, lon, bearing)
        self._json({"ok": not err, "error": err}, 409 if err else 200)

    def _gps(self, body: bytes):
        """기기 위치 {lat, lon, acc}. 로봇 서버는 odom과 짝지어 저장, 복제본은 지금 연결된 로봇 서버로 넘긴다."""
        s = self.store
        try:
            d = json.loads(body.decode("utf-8"))
            lat, lon = float(d["lat"]), float(d["lon"])
            acc = None if d.get("acc") is None else float(d["acc"])
        except (ValueError, KeyError, TypeError) as e:
            self._json({"ok": False, "error": f"JSON: {e}"}, 400)
            return
        if s.forward_gps:
            self._forward("/api/gps", body)
            return
        err = s.add_gps(lat, lon, acc)
        self._json({"ok": not err, "error": err}, 409 if err else 200)


def _https_context(cert_dir: Path):
    """자체 서명 인증서(처음 한 번 openssl로 만든다)로 https를 연다. 휴대폰 브라우저 위치 기능용."""
    import ssl
    import subprocess
    cert, key = cert_dir / "stepmap_cert.pem", cert_dir / "stepmap_key.pem"
    if not cert.is_file() or not key.is_file():
        cert_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650", "-subj", "/CN=stepmap",
                        "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert), str(key))
    return ctx


def _exporter(store: LiveStore, stop: threading.Event):
    while not stop.is_set():
        try:
            if store.export():
                print(f"[live] {time.strftime('%H:%M:%S')} 대시보드 갱신 v{store.version}: {store.summary}", flush=True)
        except Exception as e:  # noqa: BLE001  내보내기 실패는 다음 갱신 때 다시 시도 (스레드가 죽으면 대시보드가 멈춘다)
            print(f"[live] 내보내기 실패: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        stop.wait(EXPORT_MIN_S)


def _pull_once(store: LiveStore, url: str) -> int:
    """url에서 새 항목을 한 번 받아 저장한다. 받은 항목 수를 돌려준다 (실패하면 예외)."""
    last = int(store.meta.get("pull_seq") or 0)
    q = urllib.parse.urlencode({"seq": last, "now": f"{time.time():.3f}"})
    with urllib.request.urlopen(f"{url}/api/since?{q}", timeout=4) as r:
        data = json.loads(r.read().decode("utf-8"))
    items = data["items"]
    off = float(data.get("clock_offset") or 0.0)     # 복제본은 로봇의 시계 보정값을 그대로 쓴다
    if off != store.meta.get("clock_offset"):
        with store.lock:
            store.meta["clock_offset"] = off
            store._save_meta()
            store.dirty = True
    # 사진: 먼저 파일을 받아 둔다. 같은 이름이 다시 오면 로봇이 더 가까이에서 찍은 사진으로 바꾼 것이므로 다시 받는다
    # (예전에는 이미 있으면 건너뛰어 첫 사진에 멈춰 있었다). 한 묶음 안에서는 이름마다 마지막 것만 받는다.
    paths = list(dict.fromkeys(it["path"] for it in items if it["type"] == "file"))
    for rel in paths:
        dst = store.dir / rel
        try:
            with urllib.request.urlopen(f"{url}/files/{urllib.parse.quote(rel)}", timeout=10) as r:
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".part")
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                os.replace(tmp, dst)
        except urllib.error.HTTPError as e:      # 로봇에 없는 사진 하나 때문에 이후 전체가 멈추지 않게
            if e.code != 404:
                raise
            print(f"[live] 사진 없음 (건너뜀): {rel}", flush=True)
    if items:
        store.ingest(items, keep_src_seq=True)
        with store.lock:
            store.meta["pull_seq"] = items[-1]["seq"]
            store._save_meta()
    return len(items)


def _puller(store: LiveStore, urls, stop: threading.Event):
    """원격(로봇) 서버에서 새 항목을 끌어온다. 끊기면 마지막으로 받은 seq 다음부터 다시 받는다.

    urls: 같은 로봇에 닿는 주소 여러 개 (쉼표로 구분한 문자열 또는 목록). 예: 유선 LAN, Go2 WiFi 경유,
    휴대폰 핫스팟, Tailscale. 지금 주소가 끊기면 다음 주소를 차례로 시도하고, 닿는 주소에 머문다.
    """
    if isinstance(urls, str):
        urls = [u for u in urls.split(",") if u.strip()]
    urls = [u.strip().rstrip("/") for u in urls]
    i, connected = 0, None
    while not stop.is_set():
        url = urls[i]
        try:
            n = _pull_once(store, url)
            store.pull_state = {"url": url, "ok": True, "t": time.time()}
            if connected != url:
                print(f"[live] {url} 연결됨 (받은 seq {store.meta.get('pull_seq')})", flush=True)
                connected = url
            if n >= 2000:
                continue                             # 밀린 것이 더 있으면 바로 다음 묶음
        except Exception as e:  # noqa: BLE001  끊김(IncompleteRead 등 http.client 오류 포함): 스레드가 죽지 않고 이어받는다
            if connected is not False:
                print(f"[live] {url} 연결 끊김 ({e}); 다른 주소/재연결 시 이어받음", flush=True)
            store.pull_state = {"url": url, "ok": False, "t": time.time()}
            connected = False
            if len(urls) > 1:
                i = (i + 1) % len(urls)
                if i != 0:
                    continue                         # 다음 주소를 바로 시도, 한 바퀴 돌면 잠시 쉰다
        stop.wait(PULL_PERIOD_S)


DEFAULT_CAMERA = "/dev/shm/terrain_live.jpg"   # 로봇: inspect_node.py가 초당 여러 번 쓰는 최신 카메라 사진
CAMERA_POLL_S = 0.015            # MJPEG: 카메라 사진 파일이 바뀌었는지 보는 간격
CAMERA_IDLE_S = 10.0             # MJPEG: 이만큼 새 사진이 없으면 스트림을 닫는다 (브라우저가 다시 연결)


def serve(folder: str | Path, port: int = 8060, config: str | Path = "config.yaml", pull: str | None = None,
          geo: GeoRef | None = None, host: str = "0.0.0.0", camera: str | None = None,
          https_port: int | None = None, sync_clock: str | None = None) -> None:
    cfg = load_config(config) if Path(config).exists() else {}
    store = LiveStore(folder, cfg, geo)
    store.camera = camera if camera is not None else (None if pull else DEFAULT_CAMERA)
    store.robot_photo = Path(config).resolve().parent / "data" / "robot.png"
    store.forward_gps = bool(pull)
    if store.geo_is_fallback:
        print("[live] config.yaml geo 미설정: 대시보드의 '내 위치로 보정'(기기 GPS)으로 실제 위치를 맞춥니다. "
              "그 전까지 지도 위치는 기준점 대비 상대값", flush=True)
    _Handler.store = store
    stop = threading.Event()
    threading.Thread(target=_exporter, args=(store, stop), daemon=True).start()
    if pull:
        threading.Thread(target=_puller, args=(store, pull, stop), daemon=True).start()
    if sync_clock:                               # LAN으로 연결된 로봇의 시계를 이 컴퓨터 시각으로 (stepmap/clock.py)
        from .clock import watch_clock
        threading.Thread(target=watch_clock, args=(sync_clock, 30.0, lambda m: print("[live] " + m, flush=True), stop),
                         daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"[live] 세션 {store.dir} (항목 {store.seq}개) → http://{host}:{port}/", flush=True)
    if https_port:
        try:
            httpsd = ThreadingHTTPServer((host, https_port), _Handler)
            httpsd.socket = _https_context(Path.home() / ".stepmap").wrap_socket(httpsd.socket, server_side=True)
            threading.Thread(target=httpsd.serve_forever, daemon=True).start()
            store.https_port = https_port
            print(f"[live] 휴대폰 위치용 https://{host}:{https_port}/ (자체 서명 인증서: 처음 한 번 '계속' 눌러 열기)", flush=True)
        except (OSError, ImportError) as e:
            print(f"[live] https 열기 실패 ({e}): http만 씁니다", file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
        store.export(force=True)
