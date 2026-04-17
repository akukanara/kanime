import argparse
import asyncio
import boto3
import html
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, closing
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from botocore.exceptions import BotoCoreError, ClientError


APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "videos.db"
WORKDIR_ROOT = APP_DIR / "workdir"
DEFAULT_SOURCE_URL = "https://www.bilibili.tv/id/play/2343020"
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
LOGGER = logging.getLogger("video_downloader")
JOB_LOGS: dict[str, list[str]] = {}
JOB_LOGS_LOCK = threading.Lock()
S3_CLIENT = None


class ProcessVideoRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    source_url: str = Field(default=DEFAULT_SOURCE_URL)
    cookies_text: str | None = Field(default=None)


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


def configure_logging(dev_mode: bool) -> None:
    level = logging.DEBUG if dev_mode else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    LOGGER.setLevel(level)


def log_info(message: str, **context: str) -> None:
    rendered = message
    if context:
        rendered = f"{message} | " + " ".join(f"{key}={value}" for key, value in context.items())
    if context:
        LOGGER.info("%s | %s", message, " ".join(f"{key}={value}" for key, value in context.items()))
    else:
        LOGGER.info(message)
    job_id = context.get("job_id")
    if job_id:
        with JOB_LOGS_LOCK:
            JOB_LOGS.setdefault(job_id, []).append(rendered)


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
        columns = [row["name"] for row in connection.execute("PRAGMA table_info(video_uploads)").fetchall()]
        expected_columns = ["id", "job_id", "title", "episode", "resolution", "link"]
        if columns != expected_columns:
            log_info("migrating video_uploads schema", old_columns=",".join(columns))
            connection.execute("ALTER TABLE video_uploads RENAME TO video_uploads_legacy")
            connection.execute(
                """
                CREATE TABLE video_uploads (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    episode TEXT NOT NULL DEFAULT '',
                    resolution TEXT NOT NULL,
                    link TEXT NOT NULL DEFAULT ''
                )
                """
            )
            legacy_columns = set(columns)
            if {"id", "job_id", "title", "episode", "resolution", "link"}.issubset(legacy_columns):
                connection.execute(
                    """
                    INSERT INTO video_uploads (id, job_id, title, episode, resolution, link)
                    SELECT id, job_id, title, episode, resolution, link
                    FROM video_uploads_legacy
                    """
                )
            connection.execute("DROP TABLE video_uploads_legacy")
        connection.commit()


def ensure_system_dependencies() -> None:
    missing_commands = [command for command in ["yt-dlp", "ffmpeg", "ffprobe"] if shutil.which(command) is None]
    if missing_commands:
        raise RuntimeError(f"Missing required system dependencies: {', '.join(missing_commands)}")
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


app = FastAPI(title="Downloader API", lifespan=lifespan)


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
            combined_output = "".join(output_lines)
            raise HTTPException(status_code=500, detail={"message": f"Command failed: {' '.join(command)}", "stdout": combined_output[-4000:]})
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


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}


def strip_trailing_media_id(value: str) -> str:
    return re.sub(r"\s*\[\d+\]$", "", value).strip()


def resolve_episode(file_path: Path) -> str:
    episode = strip_trailing_media_id(file_path.stem.strip())
    if not episode:
        raise HTTPException(status_code=500, detail="Unable to determine episode from downloaded filename")
    return episode


def download_videos(source_url: str, workspace: Path, cookies_text: str | None = None) -> list[Path]:
    log_info("download started", source_url=source_url, workspace=str(workspace))
    if cookies_text and cookies_text.strip():
        cookies_path = workspace / "cookies.txt"
        cookies_path.write_text(cookies_text, encoding="utf-8")
        command = ["yt-dlp", "--cookies", str(cookies_path), source_url]
    else:
        command = ["yt-dlp", "--cookies-from-browser", "chrome", source_url]
    run_command(command, cwd=workspace)
    files = sorted((path for path in workspace.iterdir() if is_video_file(path)), key=lambda item: item.name)
    if not files:
        raise HTTPException(status_code=500, detail="Downloaded video file was not found")
    log_info("download finished", file_count=str(len(files)))
    return files


def create_job(job_id: str) -> None:
    log_info("job created", job_id=job_id)
    with closing(get_connection()) as connection:
        connection.execute("INSERT INTO video_jobs (job_id, status, title, episode, error) VALUES (?, 'processing', '', '', '')", (job_id,))
        connection.commit()


def update_job(job_id: str, status: str, title: str | None = None, episode: str | None = None, error: str | None = None) -> None:
    log_info("job updated", job_id=job_id, status=status, title=title or "", episode=episode or "", error=error or "")
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


def save_variant(job_id: str, title: str | None, episode: str, resolution: str, link: str) -> None:
    log_info("saving variant", job_id=job_id, episode=episode, resolution=resolution, link=link)
    with closing(get_connection()) as connection:
        connection.execute(
            "INSERT INTO video_uploads (id, job_id, title, episode, resolution, link) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), job_id, title or "", episode, resolution, link),
        )
        connection.commit()


def delete_variants(job_id: str) -> None:
    log_info("deleting variants", job_id=job_id)
    with closing(get_connection()) as connection:
        connection.execute("DELETE FROM video_uploads WHERE job_id = ?", (job_id,))
        connection.commit()


def build_job_response(job_id: str) -> VideoJobResponse:
    log_info("building job response", job_id=job_id)
    with closing(get_connection()) as connection:
        job = connection.execute("SELECT job_id, status, title, episode, error FROM video_jobs WHERE job_id = ?", (job_id,)).fetchone()
        rows = connection.execute(
            "SELECT id AS _id, job_id, title, episode, resolution, link FROM video_uploads WHERE job_id = ? ORDER BY episode, resolution",
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


def list_jobs() -> list[dict]:
    with closing(get_connection()) as connection:
        rows = connection.execute(
            "SELECT job_id, status, title, episode, error, rowid FROM video_jobs ORDER BY rowid DESC LIMIT 100"
        ).fetchall()
    return [dict(row) for row in rows]


def get_media_duration_seconds(input_path: Path, workspace: Path, job_id: str) -> float:
    log_info("probing duration", job_id=job_id, file=input_path.name)
    stdout = run_command_with_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)], cwd=workspace)
    duration = float(stdout.strip())
    log_info("duration probed", job_id=job_id, file=input_path.name, duration_seconds=str(duration))
    return duration


def get_media_height(input_path: Path, workspace: Path, job_id: str) -> int:
    log_info("probing height", job_id=job_id, file=input_path.name)
    stdout = run_command_with_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
        cwd=workspace,
    )
    height = int(stdout.strip())
    log_info("height probed", job_id=job_id, file=input_path.name, height=str(height))
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


def compress_video(input_path: Path, resolution: str, height: int, workspace: Path, job_id: str) -> Path:
    output_path = workspace / f"{input_path.stem}_{resolution}.mp4"
    log_info("compression started", job_id=job_id, file=input_path.name, resolution=resolution, height=str(height))
    duration = get_media_duration_seconds(input_path, workspace, job_id)
    video_bitrate_kbps = calculate_video_bitrate_kbps(duration, resolution)
    audio_bitrate_kbps = TARGET_AUDIO_BITRATES_KBPS[resolution]
    run_command(build_ffmpeg_command(input_path, output_path, height, video_bitrate_kbps, audio_bitrate_kbps), cwd=workspace)
    log_info("compression pass finished", job_id=job_id, output=output_path.name, resolution=resolution, size_bytes=str(output_path.stat().st_size))
    if output_path.stat().st_size > TARGET_SIZE_LIMITS_BYTES[resolution]:
        log_info("compression retry needed", job_id=job_id, output=output_path.name, resolution=resolution, size_bytes=str(output_path.stat().st_size))
        output_path.unlink(missing_ok=True)
        reduced = max(int(video_bitrate_kbps * 0.7), MIN_VIDEO_BITRATE_KBPS)
        run_command(build_ffmpeg_command(input_path, output_path, height, reduced, audio_bitrate_kbps), cwd=workspace)
        log_info("compression retry finished", job_id=job_id, output=output_path.name, resolution=resolution, size_bytes=str(output_path.stat().st_size))
    if output_path.stat().st_size > TARGET_SIZE_LIMITS_BYTES[resolution]:
        raise HTTPException(status_code=400, detail=f"Hasil kompresi untuk {resolution} melebihi target ukuran")
    log_info("compression finished", job_id=job_id, output=output_path.name, resolution=resolution, size_bytes=str(output_path.stat().st_size))
    return output_path


def build_storage_link(object_key: str) -> str:
    base = S3_PUBLIC_BASE_URL or S3_ENDPOINT_URL
    return f"{base.rstrip('/')}/{S3_BUCKET_NAME}/{object_key}"


def upload_to_storage(file_path: Path, job_id: str, episode: str, resolution: str) -> str:
    assert S3_CLIENT is not None
    log_info("storage upload started", job_id=job_id, file=file_path.name, resolution=resolution, size_bytes=str(file_path.stat().st_size))
    object_key = f"{job_id}/{episode}/{resolution}/{file_path.name}"
    last_error: Exception | None = None
    for attempt in range(1, STORAGE_UPLOAD_MAX_ATTEMPTS + 1):
        log_info("storage upload attempt", job_id=job_id, file=file_path.name, attempt=str(attempt), max_attempts=str(STORAGE_UPLOAD_MAX_ATTEMPTS))
        try:
            S3_CLIENT.upload_file(str(file_path), S3_BUCKET_NAME, object_key, ExtraArgs={"ContentType": "video/mp4"})
            link = build_storage_link(object_key)
            log_info("storage upload finished", job_id=job_id, file=file_path.name, link=link, attempt=str(attempt))
            return link
        except (BotoCoreError, ClientError) as exc:
            last_error = exc
            log_info("storage upload attempt failed", job_id=job_id, file=file_path.name, attempt=str(attempt), detail=str(exc))
            if attempt < STORAGE_UPLOAD_MAX_ATTEMPTS:
                time.sleep(STORAGE_UPLOAD_RETRY_DELAY_SECONDS)
    raise HTTPException(status_code=502, detail={"message": "Storage upload failed", "error": str(last_error) if last_error else "unknown"})


def transcode_and_store(job_id: str, file_path: Path, title: str | None, episode: str, workspace: Path) -> list[dict]:
    source_height = get_media_height(file_path, workspace, job_id)
    variants: list[dict] = []
    for resolution, height in TARGET_RESOLUTIONS.items():
        if height > source_height:
            log_info("resolution skipped", job_id=job_id, file=file_path.name, resolution=resolution, source_height=str(source_height))
            continue
        compressed = compress_video(file_path, resolution, height, workspace, job_id)
        try:
            link = upload_to_storage(compressed, job_id, episode, resolution)
            variants.append({"title": title, "episode": episode, "resolution": resolution, "link": link})
        finally:
            compressed.unlink(missing_ok=True)
            log_info("compressed file removed", job_id=job_id, file=compressed.name)
    if not variants:
        raise HTTPException(status_code=400, detail="Tidak ada resolusi target yang cocok dengan tinggi video sumber")
    return variants


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Video Pipeline</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f4f1ea; color: #1f1c18; }}
    .grid {{ display: grid; grid-template-columns: 420px 1fr; gap: 24px; }}
    textarea, input {{ width: 100%; box-sizing: border-box; padding: 10px; margin-top: 6px; margin-bottom: 12px; }}
    textarea {{ min-height: 180px; }}
    button {{ padding: 10px 14px; border: 0; background: #0f766e; color: white; cursor: pointer; }}
    pre {{ background: #111827; color: #e5e7eb; padding: 12px; min-height: 360px; overflow: auto; white-space: pre-wrap; }}
    .card {{ background: white; padding: 16px; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }}
    ul {{ padding-left: 18px; }}
    li {{ cursor: pointer; margin-bottom: 8px; }}
  </style>
</head>
<body>
  <h1>Video Pipeline</h1>
  <div class="grid">
    <div class="card">
      <label>Source URL</label>
      <input id="source_url" value="{html.escape(DEFAULT_SOURCE_URL)}" />
      <label>Title</label>
      <input id="title" placeholder="Optional title override" />
      <label>Cookies Text</label>
      <textarea id="cookies_text" placeholder="Paste Netscape cookies text here. Leave empty to use --cookies-from-browser chrome."></textarea>
      <button onclick="startJob()">Start Job</button>
      <h3>Jobs</h3>
      <ul id="jobs"></ul>
    </div>
    <div class="card">
      <h3 id="job_title">No job selected</h3>
      <pre id="logs"></pre>
      <pre id="details"></pre>
    </div>
  </div>
  <script>
    let currentJobId = null;
    let eventSource = null;
    async function refreshJobs() {{
      const res = await fetch('/videos');
      const jobs = await res.json();
      const ul = document.getElementById('jobs');
      ul.innerHTML = '';
      for (const job of jobs) {{
        const li = document.createElement('li');
        li.textContent = `${{job.job_id}} | ${{job.status}} | ${{job.title || ''}}`;
        li.onclick = () => selectJob(job.job_id);
        ul.appendChild(li);
      }}
    }}
    async function startJob() {{
      const payload = {{
        source_url: document.getElementById('source_url').value,
        title: document.getElementById('title').value || null,
        cookies_text: document.getElementById('cookies_text').value || null
      }};
      const res = await fetch('/videos/process', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      const data = await res.json();
      if (!res.ok) {{
        alert(JSON.stringify(data, null, 2));
        return;
      }}
      await refreshJobs();
      selectJob(data.job_id);
    }}
    async function selectJob(jobId) {{
      currentJobId = jobId;
      document.getElementById('job_title').textContent = jobId;
      document.getElementById('logs').textContent = '';
      if (eventSource) eventSource.close();
      eventSource = new EventSource(`/videos/${{jobId}}/logs/stream`);
      eventSource.onmessage = (event) => {{
        document.getElementById('logs').textContent += event.data + "\\n";
      }};
      await refreshJobDetails();
    }}
    async function refreshJobDetails() {{
      if (!currentJobId) return;
      const res = await fetch(`/videos/${{currentJobId}}`);
      const data = await res.json();
      document.getElementById('details').textContent = JSON.stringify(data, null, 2);
      if (data.status === 'processing') setTimeout(refreshJobDetails, 2000);
      refreshJobs();
    }}
    refreshJobs();
  </script>
</body>
</html>"""


@app.get("/videos")
def videos_index() -> list[dict]:
    return list_jobs()


@app.get("/videos/{job_id}/logs")
def get_job_logs(job_id: str) -> dict[str, list[str]]:
    with JOB_LOGS_LOCK:
        return {"job_id": job_id, "logs": list(JOB_LOGS.get(job_id, []))}


@app.get("/videos/{job_id}/logs/stream")
async def stream_job_logs(job_id: str) -> StreamingResponse:
    async def event_generator():
        position = 0
        while True:
            with JOB_LOGS_LOCK:
                entries = JOB_LOGS.get(job_id, [])
                new_entries = entries[position:]
                position = len(entries)
            for entry in new_entries:
                yield f"data: {entry}\\n\\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/videos/process", response_model=VideoJobResponse)
def process_video(payload: ProcessVideoRequest) -> VideoJobResponse:
    job_id = str(uuid.uuid4())
    workspace = WORKDIR_ROOT / job_id
    workspace.mkdir(parents=True, exist_ok=True)
    create_job(job_id)
    single_episode: str | None = None

    try:
        log_info("pipeline started", job_id=job_id, source_url=payload.source_url, title=payload.title or "")
        files = download_videos(payload.source_url, workspace, payload.cookies_text)
        update_job(job_id, "processing", title=payload.title or "", episode="")

        for index, file_path in enumerate(files):
            episode = resolve_episode(file_path)
            log_info("episode processing", job_id=job_id, file=file_path.name, episode=episode)
            if len(files) == 1 and index == 0:
                single_episode = episode
            variants = transcode_and_store(job_id, file_path, payload.title, episode, workspace)
            for variant in variants:
                save_variant(job_id, variant.get("title"), variant["episode"], variant["resolution"], variant["link"])
            file_path.unlink(missing_ok=True)
            log_info("source file removed", job_id=job_id, file=file_path.name)

        update_job(job_id, "success", title=payload.title or "", episode=single_episode or "", error="")
        log_info("pipeline finished", job_id=job_id, status="success")
    except Exception as exc:
        delete_variants(job_id)
        update_job(job_id, "failed", error=str(exc))
        LOGGER.exception("pipeline failed | job_id=%s", job_id)
        raise
    finally:
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False, log_level="debug" if DEV_MODE else "info")
