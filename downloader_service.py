import argparse
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, closing
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from requests.exceptions import RequestException, SSLError


APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "videos.db"
WORKDIR_ROOT = APP_DIR / "workdir"
DEFAULT_SOURCE_URL = "https://www.bilibili.tv/id/play/2343020"
TRANSCODER_API_URL = os.getenv("TRANSCODER_API_URL", "http://10.253.128.163:9080/transcode/upload").strip()
TRANSCODER_API_VERIFY_SSL = os.getenv("TRANSCODER_API_VERIFY_SSL", "1") != "0"
DEV_MODE = False
LOGGER = logging.getLogger("video_downloader")


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
        connection.commit()


def ensure_system_dependencies() -> None:
    missing_commands = [command for command in ["yt-dlp"] if shutil.which(command) is None]
    if missing_commands:
        raise RuntimeError(f"Missing required system dependencies: {', '.join(missing_commands)}")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    ensure_system_dependencies()
    init_db()
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


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}


def strip_trailing_media_id(value: str) -> str:
    return re.sub(r"\s*\[\d+\]$", "", value).strip()


def resolve_episode(file_path: Path) -> str:
    episode = strip_trailing_media_id(file_path.stem.strip())
    if not episode:
        raise HTTPException(status_code=500, detail="Unable to determine episode from downloaded filename")
    return episode


def download_videos(source_url: str, workspace: Path) -> list[Path]:
    log_info("download started", source_url=source_url, workspace=str(workspace))
    run_command(["yt-dlp", "--cookies-from-browser", "chrome", source_url], cwd=workspace)
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


def submit_to_transcoder(job_id: str, file_path: Path, title: str | None, episode: str) -> list[dict]:
    file_size = str(file_path.stat().st_size)
    log_info("remote transcode started", job_id=job_id, file=file_path.name, episode=episode, file_size=file_size, url=TRANSCODER_API_URL)
    with file_path.open("rb") as handle:
        try:
            response = requests.post(
                TRANSCODER_API_URL,
                data={"job_id": job_id, "title": title or "", "episode": episode},
                files={"file": (file_path.name, handle, "application/octet-stream")},
                timeout=7200,
                verify=TRANSCODER_API_VERIFY_SSL,
            )
        except SSLError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "SSL handshake to transcoder API failed",
                    "url": TRANSCODER_API_URL,
                    "hint": "Check the HTTPS reverse proxy/certificate on the transcoder host, or temporarily set TRANSCODER_API_VERIFY_SSL=0.",
                    "error": str(exc),
                },
            ) from exc
        except RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Transcoder API request failed",
                    "url": TRANSCODER_API_URL,
                    "error": str(exc),
                },
            ) from exc
    log_info("remote transcode responded", job_id=job_id, file=file_path.name, status_code=str(response.status_code))
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail={"message": f"Transcoder API failed with status {response.status_code}", "response": response.text[:2000]})
    payload = response.json()
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise HTTPException(status_code=502, detail={"message": "Invalid response from transcoder API", "response": payload})
    log_info("remote transcode finished", job_id=job_id, file=file_path.name, variant_count=str(len(variants)))
    return variants


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/videos/process", response_model=VideoJobResponse)
def process_video(payload: ProcessVideoRequest) -> VideoJobResponse:
    job_id = str(uuid.uuid4())
    workspace = WORKDIR_ROOT / job_id
    workspace.mkdir(parents=True, exist_ok=True)
    create_job(job_id)
    single_episode: str | None = None

    try:
        log_info("pipeline started", job_id=job_id, source_url=payload.source_url, title=payload.title or "")
        files = download_videos(payload.source_url, workspace)
        update_job(job_id, "processing", title=payload.title or "", episode="")

        for index, file_path in enumerate(files):
            episode = resolve_episode(file_path)
            log_info("episode processing", job_id=job_id, file=file_path.name, episode=episode)
            if len(files) == 1 and index == 0:
                single_episode = episode
            variants = submit_to_transcoder(job_id, file_path, payload.title, episode)
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
