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

from app.audio import locate, verify_sequence  # noqa: E402
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

# ---------- 7. 片段顺序核验：唯一链 ----------
print("7) 片段顺序核验：唯一链")
seq_rate = 16000
seq_ref = lp_noise(10.0, seq_rate, cutoff=3000) * 0.05
# 三段按预期顺序插入：1.0s 起 0.6s、3.5s 起 0.5s、6.0s 起 0.7s
seq_specs = [(1.0, 0.6), (3.5, 0.5), (6.0, 0.7)]
seq_bursts = []
for p, d in seq_specs:
    b = rng.standard_normal(int(d * seq_rate))
    b = lfilter(firwin(101, 0.5), 1.0, b) * 0.9
    seq_bursts.append(b)
    i0 = int(p * seq_rate)
    seq_ref[i0:i0 + b.size] += b
seq_ref_wav = wav_bytes(seq_ref, seq_rate)
seq_clip_wavs = []
for k, (p, d) in enumerate(seq_specs):
    i0, i1 = int(p * seq_rate), int((p + d) * seq_rate)
    c = seq_ref[i0:i1].copy()
    # 第 2 段用不同采样率 + 小音量 + 直流偏移，验证沿用现有归一化口径
    if k == 1:
        c = resample_poly(c, 22050, seq_rate) * 0.3 + 0.05
        seq_clip_wavs.append(wav_bytes(c, 22050))
    else:
        seq_clip_wavs.append(wav_bytes(c, seq_rate))

res_seq = verify_sequence(
    bytes_meta(seq_ref_wav, "参考录音"),
    [bytes_meta(w, f"第 {i + 1} 段片段") for i, w in enumerate(seq_clip_wavs)],
)
check("核验状态 verified", res_seq["status"] == "verified", res_seq.get("status"))
check("核验唯一（无多解）", res_seq["multiple_solutions"] is False)
starts = [s["start_ms"] for s in res_seq["chain"]["segments"]]
ends = [s["end_ms"] for s in res_seq["chain"]["segments"]]
check("各段起点毫秒为 1000/3500/6000", starts == [1000, 3500, 6000], str(starts))
check("各段终点毫秒为 1600/4000/6700", ends == [1600, 4000, 6700], str(ends))
# 空档：(3.5-1.6)=1.9s、(6.0-4.0)=2.0s
check("相邻空档为 1900/2000 ms",
      res_seq["chain"]["gaps_ms"] == [1900, 2000],
      str(res_seq["chain"]["gaps_ms"]))
check("链总得分 ≈ 3（各段高分）",
      abs(res_seq["chain"]["total_score"] - 3.0) < 0.1,
      str(res_seq["chain"]["total_score"]))
check("每段候选数 >= 1",
      all(len(r["candidates"]) >= 1 for r in res_seq["segments_report"]))

# 两段紧邻（首尾相接，空档 0）也算互不重叠的完整链
touch_ref = lp_noise(6.0, seq_rate, cutoff=3000) * 0.05
t_b1 = lfilter(firwin(101, 0.5), 1.0, rng.standard_normal(int(0.5 * seq_rate))) * 0.9
t_b2 = lfilter(firwin(101, 0.5), 1.0, rng.standard_normal(int(0.6 * seq_rate))) * 0.9
touch_ref[int(2.0 * seq_rate):int(2.0 * seq_rate) + t_b1.size] += t_b1
touch_ref[int(2.5 * seq_rate):int(2.5 * seq_rate) + t_b2.size] += t_b2
t_wav = wav_bytes(touch_ref, seq_rate)
t_c1 = wav_bytes(touch_ref[int(2.0 * seq_rate):int(2.5 * seq_rate)].copy(), seq_rate)
t_c2 = wav_bytes(touch_ref[int(2.5 * seq_rate):int(3.1 * seq_rate)].copy(), seq_rate)
res_t = verify_sequence(bytes_meta(t_wav, "参考录音"),
                        [bytes_meta(t_c1, "第 1 段片段"),
                         bytes_meta(t_c2, "第 2 段片段")])
check("首尾相接空档 0 仍成立",
      res_t["status"] == "verified" and res_t["chain"]["gaps_ms"] == [0],
      str(res_t.get("status")) + " " + str(res_t.get("chain", {}).get("gaps_ms")))

# ---------- 8. 顺序核验：顺序不成立 ----------
print("8) 片段顺序核验：顺序不成立")
# 把第 3 段与第 1 段的上传顺序对调：链需按 6s -> 1s 排列，不存在
res_bad_order = verify_sequence(
    bytes_meta(seq_ref_wav, "参考录音"),
    [bytes_meta(seq_clip_wavs[2], "第 1 段片段"),
     bytes_meta(seq_clip_wavs[0], "第 2 段片段")],
)
check("逆序返回 sequence_not_established / overlap",
      res_bad_order["status"] == "sequence_not_established"
      and res_bad_order["reason"] == "overlap",
      str(res_bad_order.get("status")))

# 某段在参考中不存在（低分）
foreign_seg = wav_bytes(lp_noise(0.6, seq_rate, 3000), seq_rate)
res_foreign = verify_sequence(
    bytes_meta(seq_ref_wav, "参考录音"),
    [bytes_meta(seq_clip_wavs[0], "第 1 段片段"),
     bytes_meta(foreign_seg, "第 2 段片段")],
)
check("含无关片段返回 segment_not_found",
      res_foreign["status"] == "sequence_not_established"
      and res_foreign["reason"] == "segment_not_found",
      str(res_foreign.get("status")))

# 近乎静音片段
silent_seg = wav_bytes(np.zeros(int(0.6 * seq_rate)) + 1e-4, seq_rate)
res_silent = verify_sequence(
    bytes_meta(seq_ref_wav, "参考录音"),
    [bytes_meta(seq_clip_wavs[0], "第 1 段片段"),
     bytes_meta(silent_seg, "第 2 段片段")],
)
check("静音片段返回 silence",
      res_silent["status"] == "sequence_not_established"
      and res_silent["reason"] == "silence",
      str(res_silent.get("status")))

# ---------- 9. 顺序核验：多解 ----------
print("9) 片段顺序核验：多解提示")
multi_ref = lp_noise(10.0, seq_rate, cutoff=3000) * 0.05
m_burst = lfilter(firwin(101, 0.5), 1.0,
                  rng.standard_normal(int(0.5 * seq_rate))) * 0.9
for pos in (1.0, 2.0):  # 第一段内容出现两次（间隔 1s > 0.25s）
    p = int(pos * seq_rate)
    multi_ref[p:p + m_burst.size] += m_burst
m_burst2 = lfilter(firwin(101, 0.5), 1.0,
                   rng.standard_normal(int(0.6 * seq_rate))) * 0.9
p2 = int(4.0 * seq_rate)
multi_ref[p2:p2 + m_burst2.size] += m_burst2
multi_ref_wav = wav_bytes(multi_ref, seq_rate)
m_clip1 = wav_bytes(multi_ref[int(1.0 * seq_rate):int(1.5 * seq_rate)].copy(), seq_rate)
m_clip2 = wav_bytes(multi_ref[int(4.0 * seq_rate):int(4.6 * seq_rate)].copy(), seq_rate)
res_multi = verify_sequence(
    bytes_meta(multi_ref_wav, "参考录音"),
    [bytes_meta(m_clip1, "第 1 段片段"), bytes_meta(m_clip2, "第 2 段片段")],
)
check("多解仍判 verified", res_multi["status"] == "verified",
      res_multi.get("status"))
check("multiple_solutions 为真", res_multi["multiple_solutions"] is True)
check("给出另一条链", "alternative_chain" in res_multi)
if "alternative_chain" in res_multi:
    diff = (res_multi["chain"]["total_score"]
            - res_multi["alternative_chain"]["total_score"])
    check("两链总分差 <= 0.02", 0 <= diff <= 0.02, f"diff={diff:.4f}")
    b1 = res_multi["chain"]["segments"][0]["start_ms"]
    a1 = res_multi["alternative_chain"]["segments"][0]["start_ms"]
    check("两条链确实不同（第 1 段起点不同）", b1 != a1, f"{b1} vs {a1}")

# ---------- 10. 顺序核验 HTTP 接口 ----------
print("10) 片段顺序核验 HTTP 接口")


def seq_post(ref_data, clip_datas):
    files = [("reference", ("r.wav", ref_data, "audio/wav"))]
    files += [("clips", (f"s{i}.wav", d, "audio/wav")) for i, d in enumerate(clip_datas)]
    return client.post("/api/sequence", files=files)


r = seq_post(seq_ref_wav, seq_clip_wavs)
check("核验请求 200", r.status_code == 200, f"{r.status_code}: {r.text[:100]}")
check("首页含顺序核验模块", "片段顺序核验" in client.get("/").text)

r1 = seq_post(seq_ref_wav, seq_clip_wavs[:1])
check("仅 1 段返回 400", r1.status_code == 400, str(r1.status_code))
r6 = seq_post(seq_ref_wav, seq_clip_wavs[:2] * 3)
check("6 段返回 400", r6.status_code == 400, str(r6.status_code))
r_badwav = seq_post(b"RIFFxxxx" + b"\0" * 100, seq_clip_wavs[:2])
check("非 WAV 参考返回 400", r_badwav.status_code == 400)
r_empty = seq_post(seq_ref_wav, [b"", seq_clip_wavs[0]])
check("空片段返回 400", r_empty.status_code == 400)
r_long = seq_post(seq_ref_wav, [seq_clip_wavs[0],
                                wav_bytes(lp_noise(6.0, seq_rate, 3000), seq_rate)])
check("6 秒片段返回 400", r_long.status_code == 400)
r_multi_http = seq_post(multi_ref_wav, [m_clip1, m_clip2])
check("多解场景 HTTP 200", r_multi_http.status_code == 200
      and r_multi_http.json()["multiple_solutions"] is True,
      str(r_multi_http.status_code))

print(f"\n结果：{len(PASS)} 通过，{len(FAIL)} 失败")
if FAIL:
    print("失败项：", FAIL)
    sys.exit(1)
