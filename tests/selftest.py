"""本地自测：生成合成 WAV，验证核心定位逻辑与 HTTP 接口的报错行为。

运行：python3 tests/selftest.py
"""

import io
import sys
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from scipy.signal import firwin, lfilter, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio import locate  # noqa: E402
from app.main import app  # noqa: E402

rng = np.random.default_rng(20260925)
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def wav_bytes(samples, rate, channels=1, sampwidth=2):
    """float 波形（-1~1）-> PCM WAV 字节；sampwidth=1 时按 uint8 编码。"""
    samples = np.clip(samples, -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.setcomptype("NONE", "not compressed")
        if sampwidth == 2:
            arr = (samples * 32767).astype("<i2")
            if channels == 2:
                arr = np.repeat(arr[:, None], 2, axis=1).reshape(-1)
            wf.writeframes(arr.tobytes())
        elif sampwidth == 1:
            arr = ((samples * 127) + 128).astype(np.uint8)
            wf.writeframes(arr.tobytes())
        else:  # 24 位仅用于触发格式报错
            arr = (samples * (2 ** 23 - 1)).astype(np.int32)
            b = np.stack(
                [arr & 0xFF, (arr >> 8) & 0xFF, (arr >> 16) & 0xFF], axis=1
            ).astype(np.uint8)
            wf.setnchannels(1)
            wf.writeframes(b.reshape(-1).tobytes())
    return buf.getvalue()


def lp_noise(dur, rate, cutoff, dc=0.0):
    """带限噪声 + 直流偏移，能量集中在 cutoff 以下。"""
    x = rng.standard_normal(int(dur * rate))
    taps = firwin(101, cutoff / (rate / 2))
    x = lfilter(taps, 1.0, x)
    x = x / np.max(np.abs(x)) * 0.7
    return x.astype(np.float64) + dc


def bytes_meta(data, label):
    from app.audio import load_wav

    return load_wav(data, label)


# ---------- 1. 唯一命中：不同采样率、音量与直流偏移 ----------
print("1) 唯一命中")
ref_rate, ref_dur = 44100, 10.0
ref = lp_noise(ref_dur, ref_rate, cutoff=6000, dc=0.05)
insert_s, clip_dur = 3.25, 1.2
i0, i1 = int(insert_s * ref_rate), int((insert_s + clip_dur) * ref_rate)

# 片段：单声道、降到 16 kHz、音量乘 0.3、再带自身直流偏移
clip_hi = ref[i0:i1]
clip = resample_poly(clip_hi, 16000, ref_rate) * 0.3 + 0.1
ref_wav = wav_bytes(ref, ref_rate, channels=2)  # 参考为双声道 44.1k
clip_wav = wav_bytes(clip, 16000, channels=1)   # 片段为单声道 16k

res = locate(bytes_meta(ref_wav, "参考录音"), bytes_meta(clip_wav, "待找片段"))
check("状态为 located", res["status"] == "located", res.get("status"))
hit = res["candidates"][0]
check("起点误差 < 5 ms", abs(hit["start_s"] - insert_s) < 0.005,
      f"start={hit['start_s']} 期望≈{insert_s}")
check("时长误差 < 5 ms", abs((hit["end_s"] - hit["start_s"]) - clip_dur) < 0.005)
check("得分 >= 0.95", hit["score"] >= 0.95, f"score={hit['score']}")
check("分析采样率取较低值", res["sample_rate"] == 16000, res["sample_rate"])

# 同一段内容但极小声（-40 dB 增益），NCC 应不受音量影响
quiet_clip = resample_poly(clip_hi, 22050, ref_rate) * 0.01
res_q = locate(bytes_meta(ref_wav, "参考录音"),
               bytes_meta(wav_bytes(quiet_clip, 22050), "待找片段"))
check("极小音量仍命中", res_q["status"] == "located" and
      abs(res_q["candidates"][0]["start_s"] - insert_s) < 0.005,
      f"status={res_q['status']} score={res_q.get('best_score')}")

# ---------- 2. 无法定位：低分 ----------
print("2) 低分无法定位")
foreign = lp_noise(clip_dur, 44100, cutoff=3000)
res_nf = locate(bytes_meta(ref_wav, "参考录音"),
                bytes_meta(wav_bytes(foreign, 44100), "待找片段"))
check("状态为 not_found / low_score",
      res_nf["status"] == "not_found" and res_nf["reason"] == "low_score",
      f"status={res_nf['status']} score={res_nf.get('best_score')}")

# ---------- 3. 近乎静音 ----------
print("3) 近静音")
silent = np.zeros(int(1.0 * 16000)) + 0.0001
res_s = locate(bytes_meta(ref_wav, "参考录音"),
               bytes_meta(wav_bytes(silent, 16000), "待找片段"))
check("静音片段返回 not_found / silence",
      res_s["status"] == "not_found" and res_s["reason"] == "silence",
      res_s.get("status"))

ref_silent = np.zeros(int(2 * 16000))
res_s2 = locate(bytes_meta(wav_bytes(ref_silent, 16000), "参考录音"),
                bytes_meta(wav_bytes(clip, 16000), "待找片段"))
check("静音参考返回 silence", res_s2["reason"] == "silence")

# ---------- 4. 定位不唯一 ----------
print("4) 定位不唯一")
sr = 16000
ref_rep = lp_noise(10.0, sr, cutoff=3000) * 0.05  # 很低的背景
burst = (rng.standard_normal(int(0.8 * sr)) * 0.8)
burst = lfilter(firwin(101, 0.5), 1.0, burst)
for pos in (2.0, 7.0):
    p = int(pos * sr)
    ref_rep[p:p + burst.size] += burst
rep_clip = ref_rep[int(2.0 * sr):int(2.0 * sr) + burst.size].copy()
res_a = locate(bytes_meta(wav_bytes(ref_rep, sr), "参考录音"),
               bytes_meta(wav_bytes(rep_clip, sr), "待找片段"))
check("状态为 ambiguous", res_a["status"] == "ambiguous", res_a.get("status"))
if res_a["status"] == "ambiguous":
    starts = [c["start_s"] for c in res_a["candidates"]]
    check("识别出 2 个候选", len(starts) == 2, str(starts))
    check("候选位于 ≈2.0s 与 ≈7.0s",
          any(abs(s - 2.0) < 0.01 for s in starts) and
          any(abs(s - 7.0) < 0.01 for s in starts), str(starts))
    check("候选得分差 <= 0.02",
          max(c["score"] for c in res_a["candidates"]) -
          min(c["score"] for c in res_a["candidates"]) <= 0.02,
          str([c["score"] for c in res_a["candidates"]]))

# 候选间隔边界：两处仅相差 0.1s 时不应报不唯一（视作同一峰）
ref_close = lp_noise(10.0, sr, cutoff=3000) * 0.05
for pos in (2.0, 2.1):
    p = int(pos * sr)
    ref_close[p:p + burst.size] += burst
close_clip = ref_close[int(2.0 * sr):int(2.0 * sr) + burst.size].copy()
res_c = locate(bytes_meta(wav_bytes(ref_close, sr), "参考录音"),
               bytes_meta(wav_bytes(close_clip, sr), "待找片段"))
check("相距 < 0.25s 不算不唯一", res_c["status"] == "located",
      res_c.get("status"))

# ---------- 5. HTTP 接口与格式校验 ----------
print("5) HTTP 接口")
client = TestClient(app)


def expect_400(name, ref_data, clip_data):
    r = client.post(
        "/api/locate",
        files={"reference": ("r.wav", ref_data, "audio/wav"),
               "clip": ("c.wav", clip_data, "audio/wav")},
    )
    check(name, r.status_code == 400 and "detail" in r.json(),
          f"{r.status_code}: {r.text[:80]}")


check("健康检查", client.get("/api/health").json()["status"] == "ok")
check("首页可访问", client.get("/").status_code == 200)

ok_clip = wav_bytes(lp_noise(1.0, 16000, 3000), 16000)
ok_ref = wav_bytes(lp_noise(10.0, 16000, 3000), 16000)
r = client.post(
    "/api/locate",
    files={"reference": ("r.wav", ok_ref, "audio/wav"),
           "clip": ("c.wav", ok_clip, "audio/wav")},
)
check("合法请求 200", r.status_code == 200, r.text[:120])

expect_400("非 WAV 内容拒绝", b"RIFFxxxx" + b"\0" * 100, ok_clip)
expect_400("8 位 WAV 拒绝", wav_bytes(lp_noise(2.0, 16000, 3000), 16000, sampwidth=1), ok_clip)
expect_400("24 位 WAV 拒绝", wav_bytes(lp_noise(2.0, 16000, 3000), 16000, sampwidth=3), ok_clip)
expect_400("96 kHz 拒绝", wav_bytes(lp_noise(2.0, 96000, 6000), 96000), ok_clip)
expect_400("6 kHz 拒绝", wav_bytes(lp_noise(2.0, 6000, 2000), 6000), ok_clip)
expect_400("参考 0.5s 拒绝", wav_bytes(lp_noise(0.5, 16000, 3000), 16000), ok_clip)
expect_400("参考 31s 拒绝", wav_bytes(lp_noise(31.0, 16000, 3000), 16000), ok_clip)
expect_400("片段 0.1s 拒绝", ok_ref, wav_bytes(lp_noise(0.1, 16000, 3000), 16000))
expect_400("片段 6s 拒绝", ok_ref, wav_bytes(lp_noise(6.0, 16000, 3000), 16000))
expect_400("片段比参考长拒绝",
           wav_bytes(lp_noise(2.0, 16000, 3000), 16000),
           wav_bytes(lp_noise(3.0, 16000, 3000), 16000))
# 三声道需要单独构造
buf3 = io.BytesIO()
with wave.open(buf3, "wb") as wf:
    wf.setnchannels(3)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes((rng.standard_normal(16000 * 3) * 10000).astype("<i2").tobytes())
expect_400("三声道拒绝", buf3.getvalue(), ok_clip)

# 空数据/字段缺失
r = client.post("/api/locate",
                files={"reference": ("r.wav", b"", "audio/wav"),
                       "clip": ("c.wav", ok_clip, "audio/wav")})
check("空文件 400", r.status_code == 400)

# ---------- 6. 导出命中片段 ----------
print("6) 导出命中片段")
exp_rate = 16000
exp_ref = lp_noise(10.0, exp_rate, cutoff=3000)
exp_insert, exp_clip_dur = 3.25, 1.2
# 片段直接取自参考（同采样率），保证帧级精确裁剪可验证
exp_clip = exp_ref[int(exp_insert * exp_rate):
                   int((exp_insert + exp_clip_dur) * exp_rate)].copy()
exp_ref_wav = wav_bytes(exp_ref, exp_rate)
exp_clip_wav = wav_bytes(exp_clip, exp_rate)


def pcm16(samples, channels=1):
    """与 wav_bytes 编码一致的原始 16 位 PCM（多声道交织）。"""
    arr = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    if channels == 2:
        arr = np.repeat(arr[:, None], 2, axis=1).reshape(-1)
    return arr


def export(pre, post, ref_data=exp_ref_wav, clip_data=exp_clip_wav):
    return client.post(
        "/api/export",
        files={"reference": ("r.wav", ref_data, "audio/wav"),
               "clip": ("c.wav", clip_data, "audio/wav")},
        data={"pre_ms": str(pre), "post_ms": str(post)},
    )


r = export(500, 700)
check("导出请求 200", r.status_code == 200, f"{r.status_code}: {r.text[:100]}")
start_ms = int(r.headers["x-export-start-ms"])
end_ms = int(r.headers["x-export-end-ms"])
exp_start = int(round((exp_insert - 0.5) * exp_rate))       # 2.75 s
exp_end = int(round((exp_insert + exp_clip_dur + 0.7) * exp_rate))  # 5.15 s
check("实际起点毫秒", start_ms == round(exp_start / exp_rate * 1000),
      f"{start_ms}")
check("实际终点毫秒", end_ms == round(exp_end / exp_rate * 1000),
      f"{end_ms}")
check("Content-Disposition 附件",
      r.headers.get("content-disposition", "").startswith("attachment;"),
      r.headers.get("content-disposition", "")[:60])

with wave.open(io.BytesIO(r.content), "rb") as wf:
    check("导出 WAV 采样率与原始一致", wf.getframerate() == exp_rate,
          str(wf.getframerate()))
    check("导出 WAV 为 16 位 PCM", wf.getsampwidth() == 2)
    check("导出 WAV 单声道", wf.getnchannels() == 1)
    out_frames = wf.readframes(wf.getnframes())
check("导出帧数 = 期望起止帧之差",
      len(out_frames) == (exp_end - exp_start) * 2,
      f"{len(out_frames)} vs {(exp_end - exp_start) * 2}")
check("导出 PCM 与原始切片位精确一致",
      out_frames == pcm16(exp_ref)[exp_start:exp_end].tobytes())

# 零上下文：恰为命中区间本身
r0 = export(0, 0)
with wave.open(io.BytesIO(r0.content), "rb") as wf:
    check("零上下文导出时长 ≈ 片段时长",
          abs(wf.getnframes() / exp_rate - exp_clip_dur) < 0.001,
          str(wf.getnframes() / exp_rate))

# 超出开头边界：命中点 0.8s，前上下文 2000ms -> 收紧到 0
head_ref = lp_noise(10.0, exp_rate, 3000)
head_clip = head_ref[int(0.8 * exp_rate):int(2.0 * exp_rate)].copy()
r_h = export(2000, 100, wav_bytes(head_ref, exp_rate), wav_bytes(head_clip, exp_rate))
check("开头越界导出 200", r_h.status_code == 200, r_h.text[:100])
check("开头越界收紧到 0 ms",
      int(r_h.headers["x-export-start-ms"]) == 0,
      r_h.headers.get("x-export-start-ms"))

# 超出结尾边界：命中点 8.5s（结束 9.7s），后上下文 2000ms -> 收紧到 10000ms
tail_ref = lp_noise(10.0, exp_rate, 3000)
tail_clip = tail_ref[int(8.5 * exp_rate):int(9.7 * exp_rate)].copy()
r_t = export(100, 2000, wav_bytes(tail_ref, exp_rate), wav_bytes(tail_clip, exp_rate))
check("结尾越界导出 200", r_t.status_code == 200, r_t.text[:100])
check("结尾越界收紧到 10000 ms",
      int(r_t.headers["x-export-end-ms"]) == 10000,
      r_t.headers.get("x-export-end-ms"))

# 双声道参考：声道保持、PCM 交织位精确
st_insert, st_dur = 2.0, 1.0
st_ref = lp_noise(8.0, exp_rate, 3000)
st_clip = st_ref[int(st_insert * exp_rate):int((st_insert + st_dur) * exp_rate)].copy()
r_st = export(0, 0, wav_bytes(st_ref, exp_rate, channels=2),
              wav_bytes(st_clip, exp_rate, channels=2))
check("双声道导出 200", r_st.status_code == 200, r_st.text[:100])
st_s, st_e = int(st_insert * exp_rate), int((st_insert + st_dur) * exp_rate)
with wave.open(io.BytesIO(r_st.content), "rb") as wf:
    check("导出 WAV 保持双声道", wf.getnchannels() == 2)
    st_frames = wf.readframes(wf.getnframes())
check("双声道 PCM 位精确一致",
      st_frames == pcm16(st_ref, 2)[st_s * 2:st_e * 2].tobytes())

# 定位不唯一 -> 409，不生成文件
r_a = export(0, 0, wav_bytes(ref_rep, sr), wav_bytes(rep_clip, sr))
check("定位不唯一导出 409", r_a.status_code == 409 and "detail" in r_a.json(),
      f"{r_a.status_code}: {r_a.text[:80]}")
# 无法定位（低分）-> 409
r_n = export(0, 0, ref_wav, wav_bytes(foreign, 44100))
check("无法定位导出 409", r_n.status_code == 409, f"{r_n.status_code}")

# 上下文范围无效 -> 400，不生成文件
for bad in (-100, 250, 2100, 1, 150):
    r_bad = export(bad, 0)
    check(f"非法 pre_ms={bad} 返回 400", r_bad.status_code == 400,
          f"{r_bad.status_code}")
r_bad = export(0, 2001)
check("非法 post_ms=2001 返回 400", r_bad.status_code == 400)
# 非 WAV 输入同样走 400
r_bad = export(0, 0, ref_data=b"RIFFxxxx" + b"\0" * 100)
check("导出非 WAV 返回 400", r_bad.status_code == 400)

# ---------- 7. 片段顺序核验（核心逻辑） ----------
print("7) 片段顺序核验")
from app.audio import verify_sequence  # noqa: E402

seq_sr = 16000
seq_ref_base = lp_noise(10.0, seq_sr, cutoff=3000) * 0.03
seq_bursts = [
    lfilter(firwin(101, 0.5), 1.0, rng.standard_normal(int(0.5 * seq_sr)) * 0.8)
    for _ in range(3)
]
seq_positions = [1.0, 4.0, 7.0]
seq_ref = seq_ref_base.copy()
for b, pos in zip(seq_bursts, seq_positions):
    p = int(pos * seq_sr)
    seq_ref[p:p + b.size] += b
seq_ref_m = bytes_meta(wav_bytes(seq_ref, seq_sr), "参考录音")
seq_clip_m = [
    bytes_meta(wav_bytes(b.copy(), seq_sr), f"片段{i + 1}")
    for i, b in enumerate(seq_bursts)
]

res = verify_sequence(seq_ref_m, seq_clip_m)
check("顺序正确状态为 verified", res["status"] == "verified", res.get("status"))
if res["status"] == "verified":
    segs = res["chain"]["segments"]
    check("链含 3 段", len(segs) == 3, str(len(segs)))
    for s, pos in zip(segs, seq_positions):
        check(f"段{s['segment']}起点 ≈{pos}s（10ms 内）",
              abs(s["start_ms"] - int(pos * 1000)) <= 10,
              f"{s['start_ms']}")
    check("首段无空档", segs[0]["gap_before_ms"] is None)
    check("第 2 段空档 ≈2500ms",
          segs[1]["gap_before_ms"] is not None and
          abs(segs[1]["gap_before_ms"] - 2500) <= 5, str(segs[1]["gap_before_ms"]))
    check("第 3 段空档 ≈2500ms",
          segs[2]["gap_before_ms"] is not None and
          abs(segs[2]["gap_before_ms"] - 2500) <= 5, str(segs[2]["gap_before_ms"]))
    check("总分 ≈ 3.0（三段均近满分）",
          abs(res["chain"]["total_score"] - 3.0) < 0.05,
          str(res["chain"]["total_score"]))

# 顺序颠倒：无完整链
res_rev = verify_sequence(seq_ref_m, [seq_clip_m[2], seq_clip_m[1], seq_clip_m[0]])
check("逆序状态为 no_chain", res_rev["status"] == "no_chain", res_rev.get("status"))
check("逆序原因为 order_not_satisfied",
      res_rev.get("reason") == "order_not_satisfied", res_rev.get("reason"))
check("提示包含「顺序不成立」", "顺序不成立" in res_rev["message"])

# 多解：第 1 段只有一处，第 2 段有两处近满分位置 -> 两条不同完整链
amb_ref = lp_noise(10.0, seq_sr, cutoff=3000) * 0.03
ab1 = lfilter(firwin(101, 0.5), 1.0, rng.standard_normal(int(0.5 * seq_sr)) * 0.8)
ab2 = lfilter(firwin(101, 0.5), 1.0, rng.standard_normal(int(0.5 * seq_sr)) * 0.8)
p = int(1.0 * seq_sr)
amb_ref[p:p + ab1.size] += ab1
for pos in (5.0, 7.5):
    p = int(pos * seq_sr)
    amb_ref[p:p + ab2.size] += ab2
res_amb = verify_sequence(
    bytes_meta(wav_bytes(amb_ref, seq_sr), "参考录音"),
    [bytes_meta(wav_bytes(ab1.copy(), seq_sr), "片段1"),
     bytes_meta(wav_bytes(ab2.copy(), seq_sr), "片段2")],
)
check("多解状态为 ambiguous", res_amb["status"] == "ambiguous", res_amb.get("status"))
if res_amb["status"] == "ambiguous":
    best_starts = [s["start_ms"] for s in res_amb["best_chain"]["segments"]]
    alt_starts = [s["start_ms"] for s in res_amb["alternative_chain"]["segments"]]
    # 两条链第一段都在 1000ms，第二段分别取 5000ms / 7500ms（顺序无关排序）
    second_starts = sorted([best_starts[1], alt_starts[1]])
    check("两条链第一段都在 1000ms",
          best_starts[0] == 1000 and alt_starts[0] == 1000,
          f"{best_starts} / {alt_starts}")
    check("两条链第二段为 5000ms 与 7500ms",
          second_starts == [5000, 7500], str(second_starts))
    check("两条链确实不同（第二段位置不同）", best_starts != alt_starts)
    gap = (res_amb["best_chain"]["total_score"]
           - res_amb["alternative_chain"]["total_score"])
    check("两条链总分差 <= 0.02", gap <= 0.02 + 1e-9, f"{gap:.4f}")
    check("提示包含「多」", "多" in res_amb["message"])

# 含低分片段：无完整链（segment_not_found），并指出是哪一段
foreign_seq = lp_noise(0.5, seq_sr, cutoff=2000)
res_nf = verify_sequence(
    seq_ref_m,
    [seq_clip_m[0], bytes_meta(wav_bytes(foreign_seq, seq_sr), "片段2")],
)
check("含低分片段状态为 no_chain", res_nf["status"] == "no_chain")
check("原因为 segment_not_found", res_nf["reason"] == "segment_not_found")
check("指出第 2 段失败",
      [f["segment"] for f in res_nf["failed_segments"]] == [2],
      str(res_nf["failed_segments"]))

# 近静音片段
silent_seg = np.zeros(int(0.5 * seq_sr)) + 0.0001
res_si = verify_sequence(
    seq_ref_m,
    [seq_clip_m[0], bytes_meta(wav_bytes(silent_seg, seq_sr), "片段2")],
)
check("静音片段原因 silence",
      res_si["status"] == "no_chain"
      and res_si["failed_segments"][0]["reason"] == "silence")

# 较低分候选（仍 >=0.8）被保留：仅它能与后段成链时链成立
lr = np.random.default_rng(20260926)
low_ref = lp_noise(10.0, seq_sr, cutoff=3000) * 0.03
lb1 = lfilter(firwin(101, 0.5), 1.0, lr.standard_normal(int(0.5 * seq_sr)) * 0.8)
lb2 = lfilter(firwin(101, 0.5), 1.0, lr.standard_normal(int(0.5 * seq_sr)) * 0.8)
# 片段1：受扰（较低分但仍 >=0.8）副本在 1.0s，干净副本在 6.0s
p = int(1.0 * seq_sr)
low_ref[p:p + lb1.size] += lb1 * 0.8 + lr.standard_normal(lb1.size) * 0.22
p = int(6.0 * seq_sr)
low_ref[p:p + lb1.size] += lb1
# 片段2 在 3.0s：1.0s 低分副本可以接上（3.0 >= 1.5），
# 6.0s 高分副本在其后、无法成链——必须保留低分候选才有完整链
p = int(3.0 * seq_sr)
low_ref[p:p + lb2.size] += lb2
res_low = verify_sequence(
    bytes_meta(wav_bytes(low_ref, seq_sr), "参考录音"),
    [bytes_meta(wav_bytes(lb1.copy(), seq_sr), "片段1"),
     bytes_meta(wav_bytes(lb2.copy(), seq_sr), "片段2")],
)
check("较低分候选场景有解",
      res_low["status"] in ("verified", "ambiguous"), res_low.get("status"))
if res_low["status"] in ("verified", "ambiguous"):
    chain = (res_low["chain"] if res_low["status"] == "verified"
             else res_low["best_chain"])
    check("最高链采用 1.0s 处的较低分候选（低分候选被保留）",
          abs(chain["segments"][0]["start_ms"] - 1000) <= 10
          and chain["segments"][0]["score"] < 0.95,
          f"{chain['segments'][0]['start_ms']}@"
          f"{chain['segments'][0]['score']}")
    check("最高链第 2 段在 3.0s",
          abs(chain["segments"][1]["start_ms"] - 3000) <= 10,
          str(chain["segments"][1]["start_ms"]))

# 相邻片段首尾相接（空档 0）允许
touch_ref = lp_noise(10.0, seq_sr, cutoff=3000) * 0.03
tb1 = lfilter(firwin(101, 0.5), 1.0, rng.standard_normal(int(0.5 * seq_sr)) * 0.8)
tb2 = lfilter(firwin(101, 0.5), 1.0, rng.standard_normal(int(0.5 * seq_sr)) * 0.8)
p = int(2.0 * seq_sr)
touch_ref[p:p + tb1.size] += tb1
touch_ref[p + tb1.size:p + tb1.size + tb2.size] += tb2
res_touch = verify_sequence(
    bytes_meta(wav_bytes(touch_ref, seq_sr), "参考录音"),
    [bytes_meta(wav_bytes(tb1.copy(), seq_sr), "片段1"),
     bytes_meta(wav_bytes(tb2.copy(), seq_sr), "片段2")],
)
check("首尾相接（空档 0）链成立",
      res_touch["status"] == "verified"
      and res_touch["chain"]["segments"][1]["gap_before_ms"] == 0,
      res_touch.get("status"))

# 各段采样率不同：毫秒换算必须按各段自身分析采样率，公共轴仅用于拼链
mr_ref = lp_noise(10.0, seq_sr, cutoff=6000) * 0.03
mb1 = lfilter(firwin(101, 0.5), 1.0, rng.standard_normal(int(0.6 * seq_sr)) * 0.8)
mb2 = lfilter(firwin(101, 0.5), 1.0, rng.standard_normal(int(0.6 * seq_sr)) * 0.8)
p = int(1.0 * seq_sr)
mr_ref[p:p + mb1.size] += mb1
p = int(3.2 * seq_sr)
mr_ref[p:p + mb2.size] += mb2
res_mr = verify_sequence(
    bytes_meta(wav_bytes(mr_ref, seq_sr), "参考录音"),
    [bytes_meta(wav_bytes(resample_poly(mb1, 44100, seq_sr), 44100), "片段1"),
     bytes_meta(wav_bytes(resample_poly(mb2, 8000, seq_sr), 8000), "片段2")],
)
check("异采样率片段链成立", res_mr["status"] == "verified", res_mr.get("status"))
if res_mr["status"] == "verified":
    s1, s2 = res_mr["chain"]["segments"]
    check("异采样率公共轴取最低 8000Hz", res_mr["sample_rate"] == 8000,
          str(res_mr["sample_rate"]))
    check("44.1k 片段起点毫秒正确（1000ms）",
          abs(s1["start_ms"] - 1000) <= 10, str(s1["start_ms"]))
    check("8k 片段起点毫秒正确（3200ms）",
          abs(s2["start_ms"] - 3200) <= 10, str(s2["start_ms"]))
    check("异采样率空档 ≈1600ms",
          abs(s2["gap_before_ms"] - 1600) <= 10, str(s2["gap_before_ms"]))

# ---------- 8. 顺序核验 HTTP 接口 ----------
print("8) 顺序核验 HTTP 接口")


def seq_post(ref_data, clip_datas):
    files = [("reference", ("r.wav", ref_data, "audio/wav"))]
    for i, d in enumerate(clip_datas):
        files.append(("clips", (f"s{i + 1}.wav", d, "audio/wav")))
    # TestClient 接受 (字段名, (文件名, 数据, MIME)) 列表并保留顺序
    return client.post("/api/sequence", files=files)


r = seq_post(wav_bytes(seq_ref, seq_sr),
             [wav_bytes(b.copy(), seq_sr) for b in seq_bursts])
check("顺序核验合法请求 200", r.status_code == 200, f"{r.status_code}: {r.text[:100]}")
check("HTTP 结果 verified", r.status_code == 200 and r.json()["status"] == "verified")

# 颠倒顺序上传 -> 200 + no_chain（业务判定，不是 HTTP 错误）
r = seq_post(wav_bytes(seq_ref, seq_sr),
             [wav_bytes(seq_bursts[2].copy(), seq_sr),
              wav_bytes(seq_bursts[1].copy(), seq_sr),
              wav_bytes(seq_bursts[0].copy(), seq_sr)])
check("逆序 HTTP 200 + no_chain",
      r.status_code == 200 and r.json()["status"] == "no_chain",
      f"{r.status_code}: {r.text[:100]}")

# 只有 1 段 -> 400
r = seq_post(wav_bytes(seq_ref, seq_sr),
             [wav_bytes(seq_bursts[0].copy(), seq_sr)])
check("片段数量不足 400", r.status_code == 400, f"{r.status_code}")

# 6 段 -> 400
r = seq_post(wav_bytes(seq_ref, seq_sr),
             [wav_bytes(seq_bursts[0].copy(), seq_sr)] * 6)
check("片段数量超限 400", r.status_code == 400, f"{r.status_code}")

# 缺 clips 字段 -> 400
r = client.post(
    "/api/sequence",
    files=[("reference", ("r.wav", wav_bytes(seq_ref, seq_sr), "audio/wav"))],
)
check("缺少 clips 字段 400", r.status_code == 400, f"{r.status_code}")

# 非 WAV / 时长越界 -> 400（沿用同一套校验，标签带序号）
r = seq_post(wav_bytes(seq_ref, seq_sr),
             [b"RIFFxxxx" + b"\0" * 100, wav_bytes(seq_bursts[0].copy(), seq_sr)])
check("非 WAV 片段 400", r.status_code == 400 and "片段 1" in r.json()["detail"],
      f"{r.status_code}: {r.text[:100]}")
r = seq_post(wav_bytes(seq_ref, seq_sr),
             [wav_bytes(seq_bursts[0].copy(), seq_sr),
              wav_bytes(lp_noise(0.1, seq_sr, 2000), seq_sr)])
check("片段时长 0.1s 400", r.status_code == 400 and "片段 2" in r.json()["detail"],
      f"{r.status_code}: {r.text[:100]}")

print(f"\n结果：{len(PASS)} 通过，{len(FAIL)} 失败")
if FAIL:
    print("失败项：", FAIL)
    sys.exit(1)
