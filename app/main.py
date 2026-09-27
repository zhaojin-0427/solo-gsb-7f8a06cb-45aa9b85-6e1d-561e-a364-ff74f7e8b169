"""FastAPI 入口：提供片段定位、命中片段导出接口并托管原生前端静态页面。"""

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .audio import (
    CONTEXT_STEP_MS,
    MAX_CLIP_DURATION_S,
    MAX_CONTEXT_MS,
    MAX_REF_DURATION_S,
    MIN_CLIP_DURATION_S,
    MIN_REF_DURATION_S,
    AudioInputError,
    check_duration,
    export_hit,
    load_wav,
    locate,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="录音片段定位", version="1.1.0")


async def _load_and_check(reference: UploadFile, clip: UploadFile) -> tuple[dict, dict]:
    """读取并校验两个上传文件，返回 (参考录音, 待找片段) 元数据。

    格式 / 采样率 / 声道 / 位深 / 时长不合要求时返回 400。
    """
    try:
        ref_meta = load_wav(await reference.read(), "参考录音")
        clip_meta = load_wav(await clip.read(), "待找片段")
        check_duration(ref_meta, MIN_REF_DURATION_S, MAX_REF_DURATION_S, "参考录音")
        check_duration(clip_meta, MIN_CLIP_DURATION_S, MAX_CLIP_DURATION_S, "待找片段")
        if clip_meta["duration"] >= ref_meta["duration"]:
            raise AudioInputError(
                f"待找片段时长（{clip_meta['duration']:.3f} 秒）必须小于参考录音时长"
                f"（{ref_meta['duration']:.3f} 秒）"
            )
    except AudioInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ref_meta, clip_meta


def _check_context(label: str, value: int) -> None:
    """命中点前后上下文：0～2000 ms，步长 100 ms。"""
    if value < 0 or value > MAX_CONTEXT_MS or value % CONTEXT_STEP_MS != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label}为 {value} 毫秒，需在 0～{MAX_CONTEXT_MS} 毫秒之间"
                f"且为 {CONTEXT_STEP_MS} 的整数倍"
            ),
        )


@app.post("/api/locate")
async def locate_clip(
    reference: UploadFile = File(..., description="1～30 秒参考录音（WAV）"),
    clip: UploadFile = File(..., description="0.2～5 秒待找片段（WAV）"),
):
    ref_meta, clip_meta = await _load_and_check(reference, clip)
    return locate(ref_meta, clip_meta)


@app.post("/api/export")
async def export_clip(
    reference: UploadFile = File(..., description="1～30 秒参考录音（WAV）"),
    clip: UploadFile = File(..., description="0.2～5 秒待找片段（WAV）"),
    pre_ms: int = Form(0, description="命中点前上下文（毫秒，0～2000，步长 100）"),
    post_ms: int = Form(0, description="命中点后上下文（毫秒，0～2000，步长 100）"),
):
    _check_context("命中点前上下文", pre_ms)
    _check_context("命中点后上下文", post_ms)

    ref_meta, clip_meta = await _load_and_check(reference, clip)
    result = export_hit(ref_meta, clip_meta, pre_ms, post_ms)
    if not result["ok"]:
        # 定位不唯一 / 无法定位 / 范围无效：不生成文件
        raise HTTPException(status_code=409, detail=result["message"])

    filename = f"hit_{result['start_ms']}-{result['end_ms']}ms.wav"
    return Response(
        content=result["wav"],
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Export-Start-Ms": str(result["start_ms"]),
            "X-Export-End-Ms": str(result["end_ms"]),
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
