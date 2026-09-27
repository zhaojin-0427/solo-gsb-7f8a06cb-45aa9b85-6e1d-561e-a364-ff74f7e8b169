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

# ---- 片段顺序核验约束 ----
MIN_SEQUENCE_SEGMENTS = 2      # 顺序核验最少片段数
MAX_SEQUENCE_SEGMENTS = 5      # 顺序核验最多片段数


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


def _greedy_pick(eligible: np.ndarray, scores: np.ndarray, target_rate: int) -> list[int]:
    """从 eligible 下标中按得分降序贪心挑选，要求彼此间隔 >= 0.25 秒。

    返回下标按时间升序排列。
    """
    min_gap = int(round(MIN_CANDIDATE_SEPARATION_S * target_rate))
    chosen: list[int] = []
    for idx in eligible[np.argsort(scores[eligible])[::-1]]:
        if all(abs(int(idx) - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
    return sorted(chosen)


def pick_candidates(scores: np.ndarray, target_rate: int) -> list[int]:
    """单片段定位所用候选：得分 >= 0.8 且与最高分相差 <= 0.02。

    按得分从高到低贪心挑选，候选起点彼此间隔 >= 0.25 秒，返回时间升序下标。
    """
    best = float(np.max(scores))
    eligible = np.flatnonzero(scores >= max(SCORE_THRESHOLD, best - AMBIGUITY_TOLERANCE))
    if eligible.size == 0:
        return []
    return _greedy_pick(eligible, scores, target_rate)


def pick_all_qualified(scores: np.ndarray, target_rate: int) -> list[int]:
    """顺序核验所用候选：保留全部得分 >= 0.8 的起点（含较低分候选）。

    同样按得分降序贪心、彼此间隔 >= 0.25 秒（间隔内只保留较高峰），
    返回时间升序下标。较低分候选也需保留，供跨片段拼链时做全局判断。
    """
    eligible = np.flatnonzero(scores >= SCORE_THRESHOLD)
    if eligible.size == 0:
        return []
    return _greedy_pick(eligible, scores, target_rate)


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


# ---- 片段顺序核验 ----

def _score_segment(ref_meta: dict, clip_meta: dict, ref_cache: dict) -> dict:
    """计算单段片段在参考录音上的全部合格候选。

    复用定位同一套 WAV 解码、时长校验、重采样与归一化互相关计算；
    与单片段定位的区别仅在于：保留所有得分 >= 0.8 的候选（含较低分），
    因为较低分候选可能正是组成全局顺序链的唯一选择。
    返回 {target_rate, clip_len, status, reason?, best_score?, candidates}，
    candidates 为 [(start_idx, score), ...]，按起点时间升序。
    """
    target_rate = min(ref_meta["rate"], clip_meta["rate"])

    if is_near_silence(ref_meta["signal"]) or is_near_silence(clip_meta["signal"]):
        return {"target_rate": target_rate, "clip_len": 0,
                "status": "not_found", "reason": "silence", "candidates": []}

    ref = ref_cache.get(target_rate)
    if ref is None:
        ref = prepare(ref_meta, target_rate)
        ref_cache[target_rate] = ref
    clip = prepare(clip_meta, target_rate)
    clip_len = int(clip.size)

    scores = normalized_correlation(ref, clip)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < SCORE_THRESHOLD:
        return {"target_rate": target_rate, "clip_len": clip_len,
                "status": "not_found", "reason": "low_score",
                "best_score": best_score, "candidates": []}

    candidates = [
        (idx, float(scores[idx]))
        for idx in pick_all_qualified(scores, target_rate)
    ]
    return {"target_rate": target_rate, "clip_len": clip_len,
            "status": "matched", "best_score": best_score,
            "candidates": candidates}


def _best_two_chains(segments: list[dict]) -> list[tuple[float, tuple]]:
    """动态规划求总分最高（未舍入得分之和）的完整链，并列时保留不同的次优链。

    段内每个候选保留以它结尾的总分最高的两条「不同链」（完整路径不同即视为
    不同链）。每段候选数很少（30 秒参考上至多百余个），直接遍历上一段全部
    候选的至多两条链做合并即可。链要求各段按输入顺序排列：下一段起点不早于
    上一段终点（互不重叠，允许首尾相接，判定容差 CHAIN_EPS_S）。
    状态以 (总分, 各段候选序号组成的 tuple) 表示，最终返回总分降序的至多
    2 条不同完整链。
    """
    # 各段候选下标 / 长度换算到公共采样率时分别取整，量化误差最多 1 个公共
    # 采样点（最低 8 kHz 下约 0.125 ms）；容差取 1 个采样点以正确识别首尾相接。
    tol = 1.0

    # 第一段：每个候选各有一条链
    states: dict[int, list[tuple[float, tuple]]] = {
        cand["u_idx"]: [(cand["score"], (j,))]
        for j, cand in enumerate(segments[0]["candidates"])
    }

    for seg_i, seg in enumerate(segments[1:], start=1):
        prev_items = [(idx, chains) for idx, chains in states.items()]
        new_states: dict[int, list[tuple[float, tuple]]] = {}
        for j, cand in enumerate(seg["candidates"]):
            extended: list[tuple[float, tuple]] = []
            for prev_idx, chains in prev_items:
                prev_end = prev_idx + segments[seg_i - 1]["clip_len"]
                if prev_end <= cand["u_idx"] + tol:
                    for total, path in chains:
                        extended.append((total + cand["score"], path + (j,)))
            if extended:
                extended.sort(key=lambda t: -t[0])
                # 来源链路径互不相同，无需再按路径去重
                new_states[cand["u_idx"]] = extended[:2]
        states = new_states
        if not states:
            return []

    finals = [item for chains in states.values() for item in chains]
    finals.sort(key=lambda t: -t[0])
    return finals[:2]


def _chain_payload(segments: list[dict], path: tuple[int, ...], total: float,
                   analysis_rate: int) -> dict:
    """把一条链（各段候选在本段内的序号、未舍入总分）组装为对外结构（毫秒精度）。

    各段 raw_idx / clip_len_raw 是该段自身分析采样率（raw_rate）下的采样点，
    毫秒换算必须除以 raw_rate，而不是公共采样率。
    """
    items = []
    for seg_i, cand_j in enumerate(path):
        cand = segments[seg_i]["candidates"][cand_j]
        rate = segments[seg_i]["raw_rate"]
        start_s = cand["raw_idx"] / rate
        end_s = (cand["raw_idx"] + segments[seg_i]["clip_len_raw"]) / rate
        start_ms = int(round(start_s * 1000))
        end_ms = int(round(end_s * 1000))
        # 相邻空档以未舍入时间计算，避免各自取整后出现 -1 ms
        gap_value = None
        if items:
            gap_value = max(0.0, start_s - items[-1]["_end_s_raw"])
        items.append({
            "segment": seg_i + 1,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "score": round(float(cand["score"]), 4),
            "gap_before_ms": (None if gap_value is None
                              else int(round(gap_value * 1000))),
            "_end_s_raw": end_s,
        })
    for item in items:
        item.pop("_end_s_raw", None)
    return {
        "total_score": round(total, 4),
        "analysis_sample_rate": analysis_rate,
        "segments": items,
    }


def verify_sequence(ref_meta: dict, clip_metas: list[dict]) -> dict:
    """片段顺序核验：2～5 段片段按输入顺序在参考录音中拼出互不重叠的完整链。

    - 每段保留全部得分 >= 0.8 的候选（同段候选相隔至少 0.25 秒，按得分降序
      贪心选取），含较低分候选供全局判断；
    - 以「各取一处、按输入顺序、互不重叠」为约束，未舍入得分之和最高的完整链
      为结果；另一条不同链总分相差不超过 0.02 时提示多解；
    - 无完整链时说明顺序不成立。
    """
    # 顺序核验在「参考录音与所有片段的最低公共采样率」下统一比较位置，
    # 保证不同采样率片段的候选下标处于同一时间轴。
    analysis_rate = min([ref_meta["rate"]] + [m["rate"] for m in clip_metas])
    ref_cache: dict = {}

    raw_segments: list[dict] = []
    segment_info: list[dict] = []
    for clip_meta in clip_metas:
        seg = _score_segment(ref_meta, clip_meta, ref_cache)
        raw_segments.append(seg)
        segment_info.append(_describe(clip_meta))

    base = {
        "sample_rate": analysis_rate,
        "reference": _describe(ref_meta),
        "segments_info": segment_info,
        "segment_count": len(clip_metas),
    }

    failed = [
        {"segment": i + 1, "reason": seg["reason"],
         "best_score": round(seg["best_score"], 4)
         if seg.get("best_score") is not None else None}
        for i, seg in enumerate(raw_segments)
        if seg["status"] != "matched"
    ]
    if failed:
        return {
            **base,
            "status": "no_chain",
            "reason": "segment_not_found",
            "message": "存在得分低于 0.8 或近乎静音的片段，无法核验顺序",
            "failed_segments": failed,
        }

    # 拼链用统一时间轴（analysis_rate）：u_idx 为公共轴候选起点、
    # clip_len 为公共轴片段长度；raw_idx / clip_len_raw 保留本段采样率下
    # 的原值，供结果精确换算毫秒与回查得分。
    unified = []
    for seg in raw_segments:
        rate = seg["target_rate"]
        scale = analysis_rate / rate
        unified.append({
            "target_rate": analysis_rate,
            "clip_len": int(round(seg["clip_len"] * scale)),
            "clip_len_raw": seg["clip_len"],
            "raw_rate": rate,
            "candidates": [
                {"u_idx": int(round(idx * scale)),
                 "raw_idx": idx,
                 "score": score}
                for idx, score in seg["candidates"]
            ],
        })

    chains = _best_two_chains(unified)
    if not chains:
        return {
            **base,
            "status": "no_chain",
            "reason": "order_not_satisfied",
            "message": (
                "各段虽有命中位置，但无法按输入顺序选出互不重叠的完整链，"
                "顺序不成立"
            ),
        }

    best_total, best_path = chains[0]
    best = _chain_payload(unified, best_path, best_total, analysis_rate)
    if len(chains) == 2:
        alt_total, alt_path = chains[1]
        if best_total - alt_total <= AMBIGUITY_TOLERANCE + 1e-12:
            return {
                **base,
                "status": "ambiguous",
                "message": (
                    f"存在多条不同的完整链，且总分相差不超过 "
                    f"{AMBIGUITY_TOLERANCE:g}（最高 {best_total:.4f}、"
                    f"另一条 {alt_total:.4f}），顺序链不唯一"
                ),
                "best_chain": best,
                "alternative_chain": _chain_payload(
                    unified, alt_path, alt_total, analysis_rate),
            }

    return {**base, "status": "verified",
            "message": "各段可按输入顺序组成唯一的完整链，顺序成立",
            "chain": best}
