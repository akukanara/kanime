import argparse
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
import uvicorn
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextlib import closing
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "videos.db"
WORKDIR_ROOT = APP_DIR / "workdir"
CATBOX_UPLOAD_URL = "https://catbox.moe/user/api.php"
CATBOX_ORIGIN = "https://catbox.moe"
CATBOX_REFERER = "https://catbox.moe/"
DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
CATBOX_USERHASH = os.getenv("CATBOX_USERHASH", "").strip()
DEFAULT_SOURCE_URL = "https://www.bilibili.tv/id/play/2343020"
MAX_UPLOAD_SIZE_BYTES = 199 * 1024 * 1024
TARGET_RESOLUTIONS = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}
TARGET_AUDIO_BITRATES_KBPS = {
    "480p": 64,
    "720p": 96,
    "1080p": 128,
}
TARGET_SIZE_LIMITS_BYTES = {
    "480p": 100 * 1024 * 1024,
    "720p": 150 * 1024 * 1024,
    "1080p": 199 * 1024 * 1024,
}
TARGET_VIDEO_BITRATE_CAPS_KBPS = {
    "480p": 700,
    "720p": 1600,
    "1080p": 2800,
}
MIN_VIDEO_BITRATE_KBPS = 150
VAAPI_DEVICE = os.getenv("VAAPI_DEVICE", "/dev/dri/renderD128")
USE_VAAPI = os.getenv("FFMPEG_USE_VAAPI", "1") != "0"
VAAPI_EXTRA_HW_FRAMES = os.getenv("VAAPI_EXTRA_HW_FRAMES", "64")
VIDEO_ENCODER_PRESET = "medium"
CATBOX_UPLOAD_MAX_ATTEMPTS = 3
CATBOX_UPLOAD_RETRY_DELAY_SECONDS = 3
DEV_MODE = False
LOGGER = logging.getLogger("video_pipeline")


class ProcessVideoRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    source_url: str = Field(default=DEFAULT_SOURCE_URL)


class VideoVariantResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    job_id: str
    title: str | None
    episode: str
    resolution: str
    link: str


class VideoJobResponse(BaseModel):
    job_id: str
    status: str
    error: str | None = None
    title: str | None
    episode: str | None
    variants: list[VideoVariantResponse]


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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS video_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                episode TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT ''
            )
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(video_uploads)").fetchall()
        }
        if "episode" not in columns:
            connection.execute(
                "ALTER TABLE video_uploads ADD COLUMN episode TEXT NOT NULL DEFAULT ''"
            )
        if "link" not in columns:
            connection.execute(
                "ALTER TABLE video_uploads ADD COLUMN link TEXT NOT NULL DEFAULT ''"
            )
        job_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(video_jobs)").fetchall()
        }
        if "title" not in job_columns:
            connection.execute(
                "ALTER TABLE video_jobs ADD COLUMN title TEXT NOT NULL DEFAULT ''"
            )
        if "episode" not in job_columns:
            connection.execute(
                "ALTER TABLE video_jobs ADD COLUMN episode TEXT NOT NULL DEFAULT ''"
            )
        if "error" not in job_columns:
            connection.execute(
                "ALTER TABLE video_jobs ADD COLUMN error TEXT NOT NULL DEFAULT ''"
            )
        connection.commit()


def ensure_system_dependencies() -> None:
    required_commands = ["yt-dlp", "ffmpeg", "ffprobe", "curl"]
    missing_commands = [command for command in required_commands if shutil.which(command) is None]
    if missing_commands:
        missing = ", ".join(missing_commands)
        raise RuntimeError(f"Missing required system dependencies: {missing}")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    ensure_system_dependencies()
    init_db()
    yield


app = FastAPI(title="Video Pipeline API", lifespan=lifespan)


def configure_logging(dev_mode: bool) -> None:
    level = logging.DEBUG if dev_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    LOGGER.setLevel(level)


def log_info(message: str, **context: Any) -> None:
    if context:
        LOGGER.info("%s | %s", message, " ".join(f"{key}={value}" for key, value in context.items()))
    else:
        LOGGER.info(message)


def sanitize_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe[:80] or "video"


def run_command(command: list[str], cwd: Path) -> None:
    if DEV_MODE:
        log_info("running command", cwd=str(cwd), command=" ".join(command))
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=f"Command not found: {command[0]}") from exc

        output_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            output_lines.append(line)
        return_code = process.wait()
        if return_code != 0:
            combined_output = "".join(output_lines)
            raise HTTPException(
                status_code=500,
                detail={
                    "message": f"Command failed: {' '.join(command)}",
                    "stdout": combined_output[-4000:],
                    "stderr": combined_output[-4000:],
                },
            )
        return

    try:
        subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Command failed: {' '.join(command)}",
                "stdout": exc.stdout[-4000:],
                "stderr": exc.stderr[-4000:],
            },
        ) from exc


def run_command_with_output(command: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Command failed: {' '.join(command)}",
                "stdout": exc.stdout[-4000:],
                "stderr": exc.stderr[-4000:],
            },
        ) from exc
    return result.stdout


def strip_trailing_media_id(value: str) -> str:
    return re.sub(r"\s*\[\d+\]$", "", value).strip()


def is_video_file(file_path: Path) -> bool:
    return file_path.is_file() and file_path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}


def resolve_video_labels(file_path: Path, requested_title: str | None) -> tuple[str | None, str]:
    episode = strip_trailing_media_id(file_path.stem.strip())
    if not episode:
        raise HTTPException(status_code=500, detail="Unable to determine title from downloaded filename")
    return requested_title, episode


def download_videos(source_url: str, workspace: Path) -> list[Path]:
    log_info("download started", source_url=source_url, workspace=str(workspace))
    command = [
        "yt-dlp",
        "--cookies-from-browser",
        "chrome",
        source_url,
    ]
    run_command(command, cwd=workspace)

    files = sorted((path for path in workspace.iterdir() if is_video_file(path)), key=lambda item: item.name)
    if not files:
        raise HTTPException(status_code=500, detail="Downloaded video file was not found")

    log_info("download finished", file_count=len(files))
    return files


def get_media_duration_seconds(input_path: Path, workspace: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        result = subprocess.run(command, cwd=workspace, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Command not found: ffprobe") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to read media duration with ffprobe",
                "stdout": exc.stdout[-4000:],
                "stderr": exc.stderr[-4000:],
            },
        ) from exc

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Unable to parse media duration") from exc

    if duration <= 0:
        raise HTTPException(status_code=500, detail="Media duration must be greater than zero")

    return duration


def get_media_height(input_path: Path, workspace: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    stdout = run_command_with_output(command, cwd=workspace)
    try:
        height = int(stdout.strip())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Unable to parse media height") from exc

    if height <= 0:
        raise HTTPException(status_code=500, detail="Media height must be greater than zero")

    return height


def calculate_video_bitrate_kbps(duration_seconds: float, resolution: str) -> int:
    audio_bitrate_kbps = TARGET_AUDIO_BITRATES_KBPS[resolution]
    size_budget_kbps = int((TARGET_SIZE_LIMITS_BYTES[resolution] * 8) / duration_seconds / 1000) - audio_bitrate_kbps - 48
    profile_cap_kbps = TARGET_VIDEO_BITRATE_CAPS_KBPS[resolution]
    video_bitrate_kbps = min(size_budget_kbps, profile_cap_kbps)
    if video_bitrate_kbps < MIN_VIDEO_BITRATE_KBPS:
        raise HTTPException(
            status_code=400,
            detail=f"Video terlalu panjang untuk target ukuran {resolution} dengan bitrate yang masih layak",
        )
    return video_bitrate_kbps


def should_use_vaapi() -> bool:
    return USE_VAAPI and Path(VAAPI_DEVICE).exists()


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    resolution: str,
    height: int,
    video_bitrate_kbps: int,
    audio_bitrate_kbps: int,
    use_vaapi: bool,
    vaapi_low_power: bool = True,
) -> tuple[list[str], str, str]:
    if use_vaapi:
        command = [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "vaapi",
            "-hwaccel_device",
            VAAPI_DEVICE,
            "-hwaccel_output_format",
            "vaapi",
            "-extra_hw_frames",
            VAAPI_EXTRA_HW_FRAMES,
            "-i",
            str(input_path),
            "-vf",
            f"scale_vaapi=w=-2:h={height}:format=nv12",
            "-c:v",
            "hevc_vaapi",
            "-rc_mode",
            "CQP",
            "-qp",
            "25",
            "-compression_level",
            "4",
            "-low_power",
            "1" if vaapi_low_power else "0",
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
        preset_name = "vaapi-lowpower" if vaapi_low_power else "vaapi"
        return command, "hevc_vaapi", preset_name

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"scale=-2:{height}",
        "-c:v",
        "libx265",
        "-preset",
        VIDEO_ENCODER_PRESET,
        "-b:v",
        f"{video_bitrate_kbps}k",
        "-maxrate",
        f"{video_bitrate_kbps}k",
        "-bufsize",
        f"{video_bitrate_kbps * 2}k",
        "-tag:v",
        "hvc1",
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
    return command, "libx265", VIDEO_ENCODER_PRESET


def compress_video(input_path: Path, resolution: str, height: int, workspace: Path) -> Path:
    output_path = workspace / f"{input_path.stem}_{resolution}.mp4"
    log_info("compression started", input=str(input_path.name), resolution=resolution)
    duration_seconds = get_media_duration_seconds(input_path, workspace)
    video_bitrate_kbps = calculate_video_bitrate_kbps(duration_seconds, resolution)
    audio_bitrate_kbps = TARGET_AUDIO_BITRATES_KBPS[resolution]
    use_vaapi = should_use_vaapi()
    command, codec_name, preset_name = build_ffmpeg_command(
        input_path=input_path,
        output_path=output_path,
        resolution=resolution,
        height=height,
        video_bitrate_kbps=video_bitrate_kbps,
        audio_bitrate_kbps=audio_bitrate_kbps,
        use_vaapi=use_vaapi,
        vaapi_low_power=True,
    )
    try:
        run_command(command, cwd=workspace)
    except HTTPException:
        if not use_vaapi:
            raise
        log_info("vaapi low-power encode failed, retrying normal vaapi", input=str(input_path.name), resolution=resolution)
        output_path.unlink(missing_ok=True)
        command, codec_name, preset_name = build_ffmpeg_command(
            input_path=input_path,
            output_path=output_path,
            resolution=resolution,
            height=height,
            video_bitrate_kbps=video_bitrate_kbps,
            audio_bitrate_kbps=audio_bitrate_kbps,
            use_vaapi=True,
            vaapi_low_power=False,
        )
        try:
            run_command(command, cwd=workspace)
        except HTTPException:
            log_info("vaapi encode failed, falling back to software", input=str(input_path.name), resolution=resolution)
            output_path.unlink(missing_ok=True)
            command, codec_name, preset_name = build_ffmpeg_command(
                input_path=input_path,
                output_path=output_path,
                resolution=resolution,
                height=height,
                video_bitrate_kbps=video_bitrate_kbps,
                audio_bitrate_kbps=audio_bitrate_kbps,
                use_vaapi=False,
            )
            run_command(command, cwd=workspace)

    target_size_limit_bytes = TARGET_SIZE_LIMITS_BYTES[resolution]
    if output_path.stat().st_size > target_size_limit_bytes:
        output_path.unlink(missing_ok=True)
        reduced_video_bitrate_kbps = max(int(video_bitrate_kbps * 0.7), MIN_VIDEO_BITRATE_KBPS)
        retry_command, codec_name, preset_name = build_ffmpeg_command(
            input_path=input_path,
            output_path=output_path,
            resolution=resolution,
            height=height,
            video_bitrate_kbps=reduced_video_bitrate_kbps,
            audio_bitrate_kbps=audio_bitrate_kbps,
            use_vaapi=should_use_vaapi(),
            vaapi_low_power=True,
        )
        try:
            run_command(retry_command, cwd=workspace)
        except HTTPException:
            if not should_use_vaapi():
                raise
            log_info("vaapi low-power retry encode failed, retrying normal vaapi", input=str(input_path.name), resolution=resolution)
            output_path.unlink(missing_ok=True)
            retry_command, codec_name, preset_name = build_ffmpeg_command(
                input_path=input_path,
                output_path=output_path,
                resolution=resolution,
                height=height,
                video_bitrate_kbps=reduced_video_bitrate_kbps,
                audio_bitrate_kbps=audio_bitrate_kbps,
                use_vaapi=True,
                vaapi_low_power=False,
            )
            try:
                run_command(retry_command, cwd=workspace)
            except HTTPException:
                log_info("vaapi retry encode failed, falling back to software", input=str(input_path.name), resolution=resolution)
                output_path.unlink(missing_ok=True)
                retry_command, codec_name, preset_name = build_ffmpeg_command(
                    input_path=input_path,
                    output_path=output_path,
                    resolution=resolution,
                    height=height,
                    video_bitrate_kbps=reduced_video_bitrate_kbps,
                    audio_bitrate_kbps=audio_bitrate_kbps,
                    use_vaapi=False,
                )
                run_command(retry_command, cwd=workspace)

    if output_path.stat().st_size > target_size_limit_bytes:
        size_mb = round(output_path.stat().st_size / 1024 / 1024, 2)
        limit_mb = round(target_size_limit_bytes / 1024 / 1024, 2)
        raise HTTPException(
            status_code=400,
            detail=f"Hasil kompresi untuk {resolution} masih {size_mb}MB, melebihi target {limit_mb}MB",
        )

    log_info(
        "compression finished",
        output=str(output_path.name),
        resolution=resolution,
        codec=codec_name,
        preset=preset_name,
        video_bitrate_kbps=video_bitrate_kbps,
        audio_bitrate_kbps=audio_bitrate_kbps,
        size_bytes=output_path.stat().st_size,
    )
    return output_path


def upload_to_catbox(file_path: Path, workspace: Path) -> str:
    log_info("upload started", file=str(file_path.name))
    last_error: HTTPException | None = None
    effective_userhash = CATBOX_USERHASH
    for attempt in range(1, CATBOX_UPLOAD_MAX_ATTEMPTS + 1):
        log_info("upload attempt", file=str(file_path.name), attempt=attempt, max_attempts=CATBOX_UPLOAD_MAX_ATTEMPTS)
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
        if effective_userhash:
            command[6:6] = ["--form", f"userhash={effective_userhash}"]
        if not DEV_MODE:
            command.insert(1, "--silent")

        try:
            stdout = run_command_with_output(command, cwd=workspace).strip()
            if not stdout.startswith("http"):
                detail: dict[str, str] = {
                    "message": "Catbox upload failed: invalid response body",
                    "response": stdout[:1000],
                }
                if stdout == "Invalid uploader":
                    detail["message"] = "Catbox rejected uploader credentials"
                    detail["hint"] = (
                        "Set CATBOX_USERHASH with a valid Catbox userhash or use a different storage provider. "
                        "Anonymous uploads may be blocked."
                    )
                raise HTTPException(
                    status_code=502,
                    detail=detail,
                )

            log_info("upload finished", file=str(file_path.name), link=stdout, attempt=attempt)
            return stdout
        except HTTPException as exc:
            last_error = exc if exc.status_code == 502 else HTTPException(status_code=502, detail=exc.detail)
            log_info("upload attempt failed", file=str(file_path.name), attempt=attempt, detail=str(exc.detail))
            if attempt < CATBOX_UPLOAD_MAX_ATTEMPTS:
                time.sleep(CATBOX_UPLOAD_RETRY_DELAY_SECONDS)

    assert last_error is not None
    raise last_error


def save_variant(
    job_id: str,
    title: str | None,
    episode: str,
    resolution: str,
    link: str,
) -> dict[str, Any]:
    record_id = str(uuid.uuid4())
    with closing(get_connection()) as connection:
        connection.execute(
            """
            INSERT INTO video_uploads (id, job_id, title, episode, resolution, link)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (record_id, job_id, title or "", episode, resolution, link),
        )
        connection.commit()

    return {
        "_id": record_id,
        "job_id": job_id,
        "title": title,
        "episode": episode,
        "resolution": resolution,
        "link": link,
    }


def create_job(job_id: str) -> None:
    log_info("job created", job_id=job_id)
    with closing(get_connection()) as connection:
        connection.execute(
            """
            INSERT INTO video_jobs (job_id, status, title, episode, error)
            VALUES (?, 'processing', '', '', '')
            """,
            (job_id,),
        )
        connection.commit()


def update_job(
    job_id: str,
    status: str,
    title: str | None = None,
    episode: str | None = None,
    error: str | None = None,
) -> None:
    log_info("job updated", job_id=job_id, status=status, title=title or "", episode=episode or "")
    with closing(get_connection()) as connection:
        connection.execute(
            """
            UPDATE video_jobs
            SET status = ?, title = COALESCE(?, title), episode = COALESCE(?, episode), error = COALESCE(?, error)
            WHERE job_id = ?
            """,
            (status, title, episode, error, job_id),
        )
        connection.commit()


def delete_variants(job_id: str) -> None:
    log_info("variants deleted", job_id=job_id)
    with closing(get_connection()) as connection:
        connection.execute("DELETE FROM video_uploads WHERE job_id = ?", (job_id,))
        connection.commit()


def get_auto_resolutions(input_path: Path, workspace: Path) -> dict[str, int]:
    source_height = get_media_height(input_path, workspace)
    resolutions = {
        label: height for label, height in TARGET_RESOLUTIONS.items() if height <= source_height
    }
    if resolutions:
        return resolutions
    raise HTTPException(
        status_code=400,
        detail="Tinggi video sumber di bawah 480p, jadi tidak bisa diproses ke profil standar 480p/720p/1080p",
    )


def build_job_response(job_id: str) -> VideoJobResponse:
    with closing(get_connection()) as connection:
        job = connection.execute(
            """
            SELECT job_id, status, title, episode, error
            FROM video_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        rows = connection.execute(
            """
            SELECT id AS _id, job_id, title, episode, resolution, link
            FROM video_uploads
            WHERE job_id = ?
            ORDER BY resolution
            """,
            (job_id,),
        ).fetchall()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    variants = []
    for row in rows:
        payload = dict(row)
        payload["title"] = payload["title"] or None
        variants.append(VideoVariantResponse(**payload))
    return VideoJobResponse(
        job_id=job["job_id"],
        status=job["status"],
        error=job["error"] or None,
        title=job["title"] or None,
        episode=job["episode"] or None,
        variants=variants,
    )


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/videos/process", response_model=VideoJobResponse)
def process_video(payload: ProcessVideoRequest) -> VideoJobResponse:
    job_id = str(uuid.uuid4())
    create_job(job_id)
    workspace = WORKDIR_ROOT / job_id
    workspace.mkdir(parents=True, exist_ok=True)
    original_files: list[Path] = []

    try:
        log_info("pipeline started", job_id=job_id, source_url=payload.source_url, title=payload.title or "")
        original_files = download_videos(payload.source_url, workspace)
        update_job(job_id, "processing", title=payload.title or "", episode="", error="")

        for original_file in original_files:
            resolved_title, resolved_episode = resolve_video_labels(original_file, payload.title)
            log_info("episode processing", job_id=job_id, episode=resolved_episode, file=str(original_file.name))
            selected_resolutions = get_auto_resolutions(original_file, workspace)

            for resolution, height in selected_resolutions.items():
                compressed_file = compress_video(original_file, resolution, height, workspace)
                try:
                    uploaded_link = upload_to_catbox(compressed_file, workspace)
                    save_variant(job_id, resolved_title, resolved_episode, resolution, uploaded_link)
                finally:
                    compressed_file.unlink(missing_ok=True)

            original_file.unlink(missing_ok=True)

        final_episode = ""
        if len(original_files) == 1:
            _, final_episode = resolve_video_labels(original_files[0], payload.title)
        update_job(job_id, "success", title=payload.title or "", episode=final_episode, error="")
        log_info("pipeline finished", job_id=job_id, status="success")
    except Exception as exc:
        delete_variants(job_id)
        update_job(job_id, "failed", error=str(exc))
        LOGGER.exception("pipeline failed | job_id=%s", job_id)
        raise
    finally:
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
            log_info("workspace removed", job_id=job_id, workspace=str(workspace))

    return build_job_response(job_id)


@app.get("/videos/{job_id}", response_model=VideoJobResponse)
def get_video_job(job_id: str) -> VideoJobResponse:
    return build_job_response(job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    DEV_MODE = args.dev
    configure_logging(DEV_MODE)
    if DEV_MODE:
        log_info("dev mode enabled")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False, log_level="debug" if DEV_MODE else "info")
