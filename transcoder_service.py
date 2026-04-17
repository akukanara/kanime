import argparse
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, closing
from pathlib import Path

import boto3
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from botocore.exceptions import BotoCoreError, ClientError


APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "transcoder_videos.db"
WORKDIR_ROOT = APP_DIR / "transcoder_workdir"
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip()
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "").strip()
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "").strip()
S3_REGION_NAME = os.getenv("S3_REGION_NAME", "").strip() or None
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL", "").strip()
TARGET_RESOLUTIONS = {"480p": 480, "720p": 720, "1080p": 1080}
TARGET_AUDIO_BITRATES_KBPS = {"480p": 64, "720p": 96, "1080p": 128}
TARGET_SIZE_LIMITS_BYTES = {"480p": 100 * 1024 * 1024, "720p": 150 * 1024 * 1024, "1080p": 199 * 1024 * 1024}
TARGET_VIDEO_BITRATE_CAPS_KBPS = {"480p": 700, "720p": 1600, "1080p": 2800}
MIN_VIDEO_BITRATE_KBPS = 150
NVENC_PRESET = os.getenv("NVENC_PRESET", "p1")
NVENC_CQ = os.getenv("NVENC_CQ", "25")
CUDA_SCALE_FILTER = os.getenv("CUDA_SCALE_FILTER", "scale_cuda")
STORAGE_UPLOAD_MAX_ATTEMPTS = 3
STORAGE_UPLOAD_RETRY_DELAY_SECONDS = 3
DEV_MODE = False
LOGGER = logging.getLogger("transcoder_service")
S3_CLIENT = None


class VideoVariantResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    job_id: str
    title: str | None
    episode: str
    resolution: str
    link: str


class TranscodeResponse(BaseModel):
    job_id: str
    title: str | None
    episode: str
    variants: list[VideoVariantResponse]


def configure_logging(dev_mode: bool) -> None:
    level = logging.DEBUG if dev_mode else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    LOGGER.setLevel(level)


def log_info(message: str, **context: str) -> None:
    if context:
        LOGGER.info("%s | %s", message, " ".join(f"{key}={value}" for key, value in context.items()))
    else:
        LOGGER.info(message)


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    WORKDIR_ROOT.mkdir(parents=True, exist_ok=True)
    with closing(get_connection()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS video_uploads (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                episode TEXT NOT NULL DEFAULT '',
                resolution TEXT NOT NULL,
                link TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.commit()


def ensure_system_dependencies() -> None:
    missing = [command for command in ["ffmpeg", "ffprobe"] if shutil.which(command) is None]
    if missing:
        raise RuntimeError(f"Missing required system dependencies: {', '.join(missing)}")
    if not all([S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
        raise RuntimeError("Missing S3/MinIO configuration. Set S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, and S3_BUCKET_NAME.")
    encoder_list = run_command_with_output(["ffmpeg", "-hide_banner", "-encoders"], cwd=APP_DIR)
    if "hevc_nvenc" not in encoder_list:
        raise RuntimeError("ffmpeg does not expose hevc_nvenc")
    filter_list = run_command_with_output(["ffmpeg", "-hide_banner", "-filters"], cwd=APP_DIR)
    if "scale_cuda" not in filter_list and "scale_npp" not in filter_list:
        raise RuntimeError("ffmpeg does not expose scale_cuda or scale_npp")
    active_filter = "scale_cuda" if "scale_cuda" in filter_list else "scale_npp"
    log_info("nvidia path ready", encoder="hevc_nvenc", scale_filter=active_filter, preset=NVENC_PRESET, cq=NVENC_CQ)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    global S3_CLIENT
    ensure_system_dependencies()
    init_db()
    S3_CLIENT = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        region_name=S3_REGION_NAME,
    )
    log_info("storage ready", endpoint=S3_ENDPOINT_URL, bucket=S3_BUCKET_NAME, public_base_url=S3_PUBLIC_BASE_URL or S3_ENDPOINT_URL)
    yield


app = FastAPI(title="Transcoder API", lifespan=lifespan)


def run_command(command: list[str], cwd: Path) -> None:
    if DEV_MODE:
        log_info("running command", cwd=str(cwd), command=" ".join(command))
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        output_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            output_lines.append(line)
        if process.wait() != 0:
            combined = "".join(output_lines)
            raise HTTPException(status_code=500, detail={"message": f"Command failed: {' '.join(command)}", "stdout": combined[-4000:]})
        return

    try:
        subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail={"message": f"Command failed: {' '.join(command)}", "stdout": exc.stdout[-4000:], "stderr": exc.stderr[-4000:]}) from exc


def run_command_with_output(command: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail={"message": f"Command failed: {' '.join(command)}", "stdout": exc.stdout[-4000:], "stderr": exc.stderr[-4000:]}) from exc
    return result.stdout


def get_media_duration_seconds(input_path: Path, workspace: Path) -> float:
    log_info("probing duration", file=input_path.name)
    stdout = run_command_with_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)], cwd=workspace)
    duration = float(stdout.strip())
    log_info("duration probed", file=input_path.name, duration_seconds=str(duration))
    return duration


def get_media_height(input_path: Path, workspace: Path) -> int:
    log_info("probing height", file=input_path.name)
    stdout = run_command_with_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
        cwd=workspace,
    )
    height = int(stdout.strip())
    log_info("height probed", file=input_path.name, height=str(height))
    return height


def calculate_video_bitrate_kbps(duration_seconds: float, resolution: str) -> int:
    audio_bitrate_kbps = TARGET_AUDIO_BITRATES_KBPS[resolution]
    size_budget_kbps = int((TARGET_SIZE_LIMITS_BYTES[resolution] * 8) / duration_seconds / 1000) - audio_bitrate_kbps - 48
    video_bitrate_kbps = min(size_budget_kbps, TARGET_VIDEO_BITRATE_CAPS_KBPS[resolution])
    if video_bitrate_kbps < MIN_VIDEO_BITRATE_KBPS:
        raise HTTPException(status_code=400, detail=f"Video terlalu panjang untuk target ukuran {resolution}")
    return video_bitrate_kbps


def build_ffmpeg_command(input_path: Path, output_path: Path, height: int, video_bitrate_kbps: int, audio_bitrate_kbps: int) -> list[str]:
    scale_filter = "scale_cuda" if CUDA_SCALE_FILTER == "scale_cuda" else "scale_npp"
    return [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(input_path),
        "-vf",
        f"{scale_filter}=-2:{height}",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        NVENC_PRESET,
        "-tune",
        "hq",
        "-cq",
        NVENC_CQ,
        "-b:v",
        f"{video_bitrate_kbps}k",
        "-maxrate",
        f"{video_bitrate_kbps}k",
        "-bufsize",
        f"{video_bitrate_kbps * 2}k",
        "-c:a",
        "aac",
        "-b:a",
        f"{audio_bitrate_kbps}k",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def compress_video(input_path: Path, resolution: str, height: int, workspace: Path) -> Path:
    output_path = workspace / f"{input_path.stem}_{resolution}.mp4"
    log_info("compression started", file=input_path.name, resolution=resolution, height=str(height))
    duration = get_media_duration_seconds(input_path, workspace)
    video_bitrate_kbps = calculate_video_bitrate_kbps(duration, resolution)
    audio_bitrate_kbps = TARGET_AUDIO_BITRATES_KBPS[resolution]
    run_command(build_ffmpeg_command(input_path, output_path, height, video_bitrate_kbps, audio_bitrate_kbps), cwd=workspace)
    log_info("compression pass finished", output=output_path.name, resolution=resolution, size_bytes=str(output_path.stat().st_size))
    if output_path.stat().st_size > TARGET_SIZE_LIMITS_BYTES[resolution]:
        log_info("compression retry needed", output=output_path.name, resolution=resolution, size_bytes=str(output_path.stat().st_size))
        output_path.unlink(missing_ok=True)
        reduced = max(int(video_bitrate_kbps * 0.7), MIN_VIDEO_BITRATE_KBPS)
        run_command(build_ffmpeg_command(input_path, output_path, height, reduced, audio_bitrate_kbps), cwd=workspace)
        log_info("compression retry finished", output=output_path.name, resolution=resolution, size_bytes=str(output_path.stat().st_size))
    if output_path.stat().st_size > TARGET_SIZE_LIMITS_BYTES[resolution]:
        raise HTTPException(status_code=400, detail=f"Hasil kompresi untuk {resolution} melebihi target ukuran")
    log_info("compression finished", output=output_path.name, resolution=resolution, size_bytes=str(output_path.stat().st_size))
    return output_path


def save_variant(job_id: str, title: str | None, episode: str, resolution: str, link: str) -> dict:
    record_id = str(uuid.uuid4())
    log_info("saving variant", job_id=job_id, episode=episode, resolution=resolution, link=link)
    with closing(get_connection()) as connection:
        connection.execute(
            "INSERT INTO video_uploads (id, job_id, title, episode, resolution, link) VALUES (?, ?, ?, ?, ?, ?)",
            (record_id, job_id, title or "", episode, resolution, link),
        )
        connection.commit()
    return {"_id": record_id, "job_id": job_id, "title": title, "episode": episode, "resolution": resolution, "link": link}


def build_storage_link(object_key: str) -> str:
    base = S3_PUBLIC_BASE_URL or S3_ENDPOINT_URL
    return f"{base.rstrip('/')}/{S3_BUCKET_NAME}/{object_key}"


def upload_to_storage(file_path: Path, job_id: str, episode: str, resolution: str) -> str:
    log_info("storage upload started", file=file_path.name, size_bytes=str(file_path.stat().st_size), job_id=job_id, resolution=resolution)
    assert S3_CLIENT is not None
    object_key = f"{job_id}/{episode}/{resolution}/{file_path.name}"
    last_error: Exception | None = None
    for attempt in range(1, STORAGE_UPLOAD_MAX_ATTEMPTS + 1):
        log_info("storage upload attempt", file=file_path.name, attempt=str(attempt), max_attempts=str(STORAGE_UPLOAD_MAX_ATTEMPTS))
        try:
            S3_CLIENT.upload_file(
                str(file_path),
                S3_BUCKET_NAME,
                object_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
            link = build_storage_link(object_key)
            log_info("storage upload finished", file=file_path.name, link=link, attempt=str(attempt))
            return link
        except (BotoCoreError, ClientError) as exc:
            last_error = exc
            log_info("storage upload attempt failed", file=file_path.name, attempt=str(attempt), detail=str(exc))
            if attempt < STORAGE_UPLOAD_MAX_ATTEMPTS:
                time.sleep(STORAGE_UPLOAD_RETRY_DELAY_SECONDS)
    raise HTTPException(status_code=502, detail={"message": "Storage upload failed", "error": str(last_error) if last_error else "unknown"})


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcode/upload", response_model=TranscodeResponse)
async def transcode_upload(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    title: str = Form(default=""),
    episode: str = Form(...),
) -> TranscodeResponse:
    workspace = WORKDIR_ROOT / f"{job_id}_{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=True)
    source_path = workspace / (file.filename or f"{uuid.uuid4()}.mp4")
    log_info("transcode request received", job_id=job_id, episode=episode, filename=file.filename or "", workspace=str(workspace))
    try:
        with source_path.open("wb") as handle:
            total_bytes = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                total_bytes += len(chunk)
            log_info("source file stored", job_id=job_id, file=source_path.name, size_bytes=str(total_bytes))

        source_height = get_media_height(source_path, workspace)
        variants: list[VideoVariantResponse] = []
        for resolution, height in TARGET_RESOLUTIONS.items():
            if height > source_height:
                log_info("resolution skipped", job_id=job_id, file=source_path.name, resolution=resolution, source_height=str(source_height))
                continue
            compressed = compress_video(source_path, resolution, height, workspace)
            try:
                link = upload_to_storage(compressed, job_id, episode, resolution)
                variants.append(VideoVariantResponse(**save_variant(job_id, title or None, episode, resolution, link)))
            finally:
                compressed.unlink(missing_ok=True)
                log_info("compressed file removed", file=compressed.name)

        if not variants:
            raise HTTPException(status_code=400, detail="Tidak ada resolusi target yang cocok dengan tinggi video sumber")
        log_info("transcode request finished", job_id=job_id, episode=episode, variant_count=str(len(variants)))
        return TranscodeResponse(job_id=job_id, title=title or None, episode=episode, variants=variants)
    finally:
        await file.close()
        source_path.unlink(missing_ok=True)
        shutil.rmtree(workspace, ignore_errors=True)
        log_info("transcode workspace removed", job_id=job_id, workspace=str(workspace))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    DEV_MODE = args.dev
    configure_logging(DEV_MODE)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "9000")), reload=False, log_level="debug" if DEV_MODE else "info")
