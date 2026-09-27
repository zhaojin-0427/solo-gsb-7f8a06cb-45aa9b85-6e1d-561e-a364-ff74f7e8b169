"""音频处理核心：WAV 校验与解码、重采样、FFT 归一化互相关定位、命中片段导出。"""

from __future__ import annotations

import io
import wave

import numpy as np
from scipy.signal import fftconvolve, resample_poly

# ---- 业务约束（与需求一一对应） ----
MIN_REF_DURATION_S = 1.0      # 参考录音最短时长（秒）
MAX_REF_DURATION_S = 30.0     # 参考录音最长时长（秒）
MIN_CLIP_DURATION_S = 0.2     # 待找片段最短时长（秒）
MAX_CLIP_DURATION_S = 5.0     # 待找片段最长时长（秒）
MIN_SAMPLE_RATE_HZ = 8000     # 支持的最低采样率
MAX_SAMPLE_RATE_HZ = 48000    # 支持的最高采样率
MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 上传体积上限，防止异常大文件拖垮服务

# ---- 定位判定阈值 ----
SCORE_THRESHOLD = 0.8          # 命中所需的最低归一化相关得分
AMBIGUITY_TOLERANCE = 0.02     # 与最高得分相差不超过该值视为并列候选
MIN_CANDIDATE_SEPARATION_S = 0.25  # 两个候选起点之间的最小间隔（秒）
SILENCE_PEAK = 1e-3            # 峰值低于该值（约 -60 dBFS）视为近乎静音
EPS = 1e-12                    # 数值保护

# ---- 命中片段导出约束 ----
MAX_CONTEXT_MS = 2000          # 命中点前后上下文上限（毫秒）
CONTEXT_STEP_MS = 100          # 上下文步长（毫秒）


class AudioInputError(ValueError):
    """输入文件不符合要求（格式 / 声道 / 位深 / 采样率 / 时长）。"""


def load_wav(data: bytes, label: str) -> dict:
    """解析并校验 WAV 字节流，返回单声道 float32 波形（值域约 [-1, 1]）。"""
    if len(data) > MAX_UPLOAD_BYTES:
        raise AudioInputError(
            f"{label}文件体积超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限"
        )
    try:
        wf = wave.open(io.BytesIO(data), "rb")
    except (wave.Error, EOFError):
        raise AudioInputError(f"{label}文件不是有效的 WAV 文件，或文件已损坏") from None

    with wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        nframes = wf.getnframes()
        comptype = wf.getcomptype()

        if comptype != "NONE":
            raise AudioInputError(f"{label}文件使用了压缩编码（{wf.getcompname()}），仅支持未压缩 PCM WAV")
        if channels not in (1, 2):
            raise AudioInputError(f"{label}文件声道数为 {channels}，仅支持单声道或双声道")
        if sampwidth != 2:
            raise AudioInputError(f"{label}文件位深为 {sampwidth * 8} 位，仅支持 16 位 PCM")
        if not (MIN_SAMPLE_RATE_HZ <= rate <= MAX_SAMPLE_RATE_HZ):
            raise AudioInputError(
                f"{label}文件采样率为 {rate} Hz，仅支持 {MIN_SAMPLE_RATE_HZ}～{MAX_SAMPLE_RATE_HZ} Hz"
            )

        raw = wf.readframes(nframes)

    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size == 0:
        raise AudioInputError(f"{label}文件不包含任何音频数据")
    # 原始交织 16 位 PCM（保留声道）：导出时直接按采样点切片，保证位精确、
    # 不经过浮点转换 / 重采样 / 声道合并。
    pcm = samples
    if channels == 2:
        usable = samples.size - (samples.size % 2)
        samples = samples[:usable].reshape(-1, 2).mean(axis=1)  # 双声道转单声道

    duration = samples.size / rate
    return {
        "signal": (samples / 32768.0).astype(np.float64),
        "rate": rate,
        "channels": channels,
        "duration": duration,
        "pcm": pcm,
    }


def check_duration(meta: dict, min_s: float, max_s: float, label: str) -> None:
    if not (min_s <= meta["duration"] <= max_s):
        raise AudioInputError(
            f"{label}时长为 {meta['duration']:.3f} 秒，要求在 {min_s:g}～{max_s:g} 秒之间"
        )


def is_near_silence(x: np.ndarray) -> bool:
    """近乎静音：满幅峰值低于 -60 dBFS。"""
    return x.size == 0 or float(np.max(np.abs(x))) < SILENCE_PEAK


def prepare(meta: dict, target_rate: int) -> np.ndarray:
    """统一重采样到 target_rate 并去除直流偏移。"""
    x = meta["signal"]
    rate = meta["rate"]
    if rate != target_rate:
        gcd = int(np.gcd(rate, target_rate))
        x = resample_poly(x, target_rate // gcd, rate // gcd)
    return x - np.mean(x)


def normalized_correlation(ref: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """用 FFT 卷积一次性计算所有起点上的归一化互相关系数（NCC）。

    返回长度为 len(ref) - len(clip) + 1 的数组，第 k 项是片段起点为
    第 k 个采样点时的得分，取值范围 [-1, 1]。
    """
    n, m = ref.size, clip.size
    num_valid = n - m + 1

    # 分子：ref 与反转片段的线性卷积，取有效位移区间
    conv = fftconvolve(ref, clip[::-1], mode="full")
    numerator = conv[m - 1 : m - 1 + num_valid]

    # 分母：参考信号每个长度 m 滑动窗口的能量（同样用卷积避免逐点遍历）
    window_energy = fftconvolve(ref * ref, np.ones(m), mode="full")
    window_energy = window_energy[m - 1 : m - 1 + num_valid]

    clip_energy = float(np.dot(clip, clip))
    denominator = np.sqrt(np.maximum(window_energy, 0.0) * clip_energy)

    scores = np.zeros(num_valid)
    np.divide(numerator, denominator, out=scores, where=denominator > EPS)
    return np.clip(scores, -1.0, 1.0)


def pick_candidates(scores: np.ndarray, target_rate: int) -> list[int]:
    """按得分从高到低贪心挑选候选起点，要求彼此间隔 >= 0.25 秒。

    返回所有得分 >= 0.8 且与最高分相差 <= 0.02 的候选下标（按时间升序）。
    """
    best = float(np.max(scores))
    eligible = np.flatnonzero(scores >= max(SCORE_THRESHOLD, best - AMBIGUITY_TOLERANCE))
    if eligible.size == 0:
        return []

    min_gap = int(round(MIN_CANDIDATE_SEPARATION_S * target_rate))
    chosen: list[int] = []
    for idx in eligible[np.argsort(scores[eligible])[::-1]]:
        if all(abs(int(idx) - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
    return sorted(chosen)


def _correlate(ref_meta: dict, clip_meta: dict) -> dict:
    """定位核心：统一采样率后做 NCC 并判定，返回精确（未舍入）的命中信息。

    供 locate（组装 API 响应）与 export_hit（精确裁剪）共用，避免同一段
    信号在「定位 + 导出」时被计算两次。
    """
    # 统一到两者中较低的采样率：片段中不存在的高频无法参与匹配，
    # 向高采样率对齐只会引入失配能量、拉低得分。
    target_rate = min(ref_meta["rate"], clip_meta["rate"])
    ref = prepare(ref_meta, target_rate)
    clip = prepare(clip_meta, target_rate)

    core: dict = {"target_rate": target_rate, "clip_len": int(clip.size)}

    if is_near_silence(ref) or is_near_silence(clip):
        return {**core, "status": "not_found", "reason": "silence"}

    scores = normalized_correlation(ref, clip)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    candidates = pick_candidates(scores, target_rate)

    if best_score < SCORE_THRESHOLD:
        return {**core, "status": "not_found", "reason": "low_score",
                "best_score": best_score}
    if len(candidates) > 1:
        return {**core, "status": "ambiguous", "best_score": best_score,
                "scores": scores, "candidates_idx": candidates}
    return {**core, "status": "located", "best_score": best_score,
            "best_idx": best_idx}


def _status_message(core: dict) -> str:
    """非唯一命中时的人类可读原因（定位响应与导出提示共用同一口径）。"""
    if core["status"] == "not_found":
        if core["reason"] == "silence":
            return "参考录音或待找片段近乎静音，无法定位"
        return (
            f"最高相关得分 {core['best_score']:.4f} 低于阈值 {SCORE_THRESHOLD}，"
            f"无法在参考录音中定位该片段"
        )
    return f"发现 {len(core['candidates_idx'])} 个得分接近的候选位置，定位不唯一"


def locate(ref_meta: dict, clip_meta: dict) -> dict:
    """完整定位流程，返回结构化结果（status: located / ambiguous / not_found）。"""
    core = _correlate(ref_meta, clip_meta)
    target_rate = core["target_rate"]
    clip_s = core["clip_len"] / target_rate

    base = {
        "sample_rate": target_rate,
        "reference": _describe(ref_meta),
        "clip": _describe(clip_meta),
        "clip_duration_s": round(clip_s, 3),
    }

    if core["status"] == "not_found":
        result = {
            **base,
            "status": "not_found",
            "reason": core["reason"],
            "message": _status_message(core),
        }
        if core["reason"] == "low_score":
            result["best_score"] = round(core["best_score"], 4)
        return result

    if core["status"] == "ambiguous":
        return {
            **base,
            "status": "ambiguous",
            "message": _status_message(core),
            "best_score": round(core["best_score"], 4),
            "candidates": [
                {
                    "start_s": round(idx / target_rate, 3),
                    "end_s": round(idx / target_rate + clip_s, 3),
                    "score": round(float(core["scores"][idx]), 4),
                }
                for idx in core["candidates_idx"]
            ],
        }

    start = core["best_idx"] / target_rate
    return {
        **base,
        "status": "located",
        "best_score": round(core["best_score"], 4),
        "candidates": [
            {
                "start_s": round(start, 3),
                "end_s": round(start + clip_s, 3),
                "score": round(core["best_score"], 4),
            }
        ],
    }


def _write_wav(pcm: np.ndarray, channels: int, rate: int) -> bytes:
    """把原始 16 位 PCM（多声道交织）封装为 WAV 字节流，参数与源文件一致。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.setcomptype("NONE", "not compressed")
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def export_hit(ref_meta: dict, clip_meta: dict, pre_ms: int, post_ms: int) -> dict:
    """定位唯一时，按原始参考 WAV 的采样率 / 声道 / 16 位 PCM 精确裁剪命中片段。

    命中区间（起点 best_idx、长度为片段在分析采样率下的长度）前后各附加
    pre_ms / post_ms 毫秒上下文；超出参考录音边界时收紧到边界。
    定位不唯一 / 无法定位 / 范围无效时返回 {"ok": False, ...}，不生成文件。
    """
    core = _correlate(ref_meta, clip_meta)
    if core["status"] != "located":
        return {
            "ok": False,
            "status": core["status"],
            "message": _status_message(core) + "，未生成导出文件",
        }

    rate = ref_meta["rate"]
    channels = ref_meta["channels"]
    pcm = ref_meta["pcm"]
    total_frames = pcm.size // channels

    # 命中区间（秒，精确到分析采样率下的采样点；不使用 locate 已舍入的毫秒值）
    target_rate = core["target_rate"]
    hit_start = core["best_idx"] / target_rate
    hit_end = (core["best_idx"] + core["clip_len"]) / target_rate

    # 附加上下文，并收紧到参考录音边界 [0, total_frames / rate]
    start_s = max(0.0, hit_start - pre_ms / 1000.0)
    end_s = min(total_frames / rate, hit_end + post_ms / 1000.0)

    # 映射到原始参考 WAV 的采样点（按原始采样率直接切 PCM，不重采样）
    start_frame = min(total_frames, max(0, int(round(start_s * rate))))
    end_frame = min(total_frames, max(0, int(round(end_s * rate))))
    if end_frame <= start_frame:
        return {
            "ok": False,
            "status": "invalid_range",
            "message": "导出范围无效（裁剪起点不早于终点），未生成导出文件",
        }

    segment = pcm[start_frame * channels : end_frame * channels]
    return {
        "ok": True,
        "wav": _write_wav(segment, channels, rate),
        "start_ms": int(round(start_frame / rate * 1000)),
        "end_ms": int(round(end_frame / rate * 1000)),
    }


def _describe(meta: dict) -> dict:
    return {
        "duration_s": round(meta["duration"], 3),
        "sample_rate": meta["rate"],
        "channels": meta["channels"],
    }
