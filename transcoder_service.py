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

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field


APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "transcoder_videos.db"
WORKDIR_ROOT = APP_DIR / "transcoder_workdir"
CATBOX_UPLOAD_URL = "https://catbox.moe/user/api.php"
CATBOX_ORIGIN = "https://catbox.moe"
CATBOX_REFERER = "https://catbox.moe/"
DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
CATBOX_USERHASH = os.getenv("CATBOX_USERHASH", "").strip()
TARGET_RESOLUTIONS = {"480p": 480, "720p": 720, "1080p": 1080}
TARGET_AUDIO_BITRATES_KBPS = {"480p": 64, "720p": 96, "1080p": 128}
TARGET_SIZE_LIMITS_BYTES = {"480p": 100 * 1024 * 1024, "720p": 150 * 1024 * 1024, "1080p": 199 * 1024 * 1024}
TARGET_VIDEO_BITRATE_CAPS_KBPS = {"480p": 700, "720p": 1600, "1080p": 2800}
MIN_VIDEO_BITRATE_KBPS = 150
NVENC_PRESET = os.getenv("NVENC_PRESET", "p4")
NVENC_CQ = os.getenv("NVENC_CQ", "25")
CATBOX_UPLOAD_MAX_ATTEMPTS = 3
CATBOX_UPLOAD_RETRY_DELAY_SECONDS = 3
DEV_MODE = False
LOGGER = logging.getLogger("transcoder_service")


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
    missing = [command for command in ["ffmpeg", "ffprobe", "curl"] if shutil.which(command) is None]
    if missing:
        raise RuntimeError(f"Missing required system dependencies: {', '.join(missing)}")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    ensure_system_dependencies()
    init_db()
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
    stdout = run_command_with_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)], cwd=workspace)
    return float(stdout.strip())


def get_media_height(input_path: Path, workspace: Path) -> int:
    stdout = run_command_with_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
        cwd=workspace,
    )
    return int(stdout.strip())


def calculate_video_bitrate_kbps(duration_seconds: float, resolution: str) -> int:
    audio_bitrate_kbps = TARGET_AUDIO_BITRATES_KBPS[resolution]
    size_budget_kbps = int((TARGET_SIZE_LIMITS_BYTES[resolution] * 8) / duration_seconds / 1000) - audio_bitrate_kbps - 48
    video_bitrate_kbps = min(size_budget_kbps, TARGET_VIDEO_BITRATE_CAPS_KBPS[resolution])
    if video_bitrate_kbps < MIN_VIDEO_BITRATE_KBPS:
        raise HTTPException(status_code=400, detail=f"Video terlalu panjang untuk target ukuran {resolution}")
    return video_bitrate_kbps


def build_ffmpeg_command(input_path: Path, output_path: Path, height: int, video_bitrate_kbps: int, audio_bitrate_kbps: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "cuda",
        "-i",
        str(input_path),
        "-vf",
        f"scale=-2:{height}",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        NVENC_PRESET,
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
    duration = get_media_duration_seconds(input_path, workspace)
    video_bitrate_kbps = calculate_video_bitrate_kbps(duration, resolution)
    audio_bitrate_kbps = TARGET_AUDIO_BITRATES_KBPS[resolution]
    run_command(build_ffmpeg_command(input_path, output_path, height, video_bitrate_kbps, audio_bitrate_kbps), cwd=workspace)
    if output_path.stat().st_size > TARGET_SIZE_LIMITS_BYTES[resolution]:
        output_path.unlink(missing_ok=True)
        reduced = max(int(video_bitrate_kbps * 0.7), MIN_VIDEO_BITRATE_KBPS)
        run_command(build_ffmpeg_command(input_path, output_path, height, reduced, audio_bitrate_kbps), cwd=workspace)
    if output_path.stat().st_size > TARGET_SIZE_LIMITS_BYTES[resolution]:
        raise HTTPException(status_code=400, detail=f"Hasil kompresi untuk {resolution} melebihi target ukuran")
    return output_path


def save_variant(job_id: str, title: str | None, episode: str, resolution: str, link: str) -> dict:
    record_id = str(uuid.uuid4())
    with closing(get_connection()) as connection:
        connection.execute(
            "INSERT INTO video_uploads (id, job_id, title, episode, resolution, link) VALUES (?, ?, ?, ?, ?, ?)",
            (record_id, job_id, title or "", episode, resolution, link),
        )
        connection.commit()
    return {"_id": record_id, "job_id": job_id, "title": title, "episode": episode, "resolution": resolution, "link": link}


def upload_to_catbox(file_path: Path, workspace: Path) -> str:
    last_error: HTTPException | None = None
    for attempt in range(1, CATBOX_UPLOAD_MAX_ATTEMPTS + 1):
        command = [
            "curl",
            "--location",
            "--show-error",
            "--fail-with-body",
            "--user-agent",
            DEFAULT_BROWSER_USER_AGENT,
            "--header",
            "Accept: text/plain, */*;q=0.8",
            "--header",
            "Accept-Language: en-US,en;q=0.9",
            "--header",
            f"Origin: {CATBOX_ORIGIN}",
            "--header",
            f"Referer: {CATBOX_REFERER}",
            "--form",
            "reqtype=fileupload",
            "--form",
            f"fileToUpload=@{file_path.name}",
            CATBOX_UPLOAD_URL,
        ]
        if CATBOX_USERHASH:
            command[16:16] = ["--form", f"userhash={CATBOX_USERHASH}"]
        if not DEV_MODE:
            command.insert(1, "--silent")
        try:
            stdout = run_command_with_output(command, cwd=workspace).strip()
            if stdout.startswith("http"):
                return stdout
            raise HTTPException(status_code=502, detail={"message": "Catbox upload failed", "response": stdout[:1000]})
        except HTTPException as exc:
            last_error = exc if exc.status_code == 502 else HTTPException(status_code=502, detail=exc.detail)
            if attempt < CATBOX_UPLOAD_MAX_ATTEMPTS:
                time.sleep(CATBOX_UPLOAD_RETRY_DELAY_SECONDS)
    assert last_error is not None
    raise last_error


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
    try:
        with source_path.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

        source_height = get_media_height(source_path, workspace)
        variants: list[VideoVariantResponse] = []
        for resolution, height in TARGET_RESOLUTIONS.items():
            if height > source_height:
                continue
            compressed = compress_video(source_path, resolution, height, workspace)
            try:
                link = upload_to_catbox(compressed, workspace)
                variants.append(VideoVariantResponse(**save_variant(job_id, title or None, episode, resolution, link)))
            finally:
                compressed.unlink(missing_ok=True)

        if not variants:
            raise HTTPException(status_code=400, detail="Tidak ada resolusi target yang cocok dengan tinggi video sumber")
        return TranscodeResponse(job_id=job_id, title=title or None, episode=episode, variants=variants)
    finally:
        await file.close()
        source_path.unlink(missing_ok=True)
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    DEV_MODE = args.dev
    configure_logging(DEV_MODE)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "9000")), reload=False, log_level="debug" if DEV_MODE else "info")
