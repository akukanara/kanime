# Split Video Pipeline

Repo ini sekarang memakai `downloader_service.py` sebagai service utama all-in-one:

- download video dengan `yt-dlp`
- menerima paste cookies text atau fallback ke browser Chrome local
- kompres `480p/720p/1080p` dengan NVIDIA NVENC
- upload hasil ke MinIO/S3 internal
- simpan metadata ke SQLite
- sediakan frontend sederhana untuk start job, lihat list job, dan stream logs

## Main Service

Dependency sistem:

- `yt-dlp`
- Google Chrome profile yang bisa dibaca `yt-dlp --cookies-from-browser chrome`
- `ffmpeg`
- `ffprobe`

Konfigurasi pakai `.env`:

```bash
cp .env.example .env
```

Isi `.env`:

```env
S3_ENDPOINT_URL=http://minio.internal:9000
S3_ACCESS_KEY_ID=minio_access_key
S3_SECRET_ACCESS_KEY=minio_secret_key
S3_BUCKET_NAME=videos
S3_PUBLIC_BASE_URL=http://minio.internal:9000
NVENC_PRESET=p1
NVENC_CQ=25
```

Jalankan:

```bash
python3 downloader_service.py --dev
```

Endpoint:

`POST /videos/process`

Contoh body:

```json
{
  "title": "Welcome To Demon School! Iruma-kun Season 4",
  "source_url": "https://www.bilibili.tv/id/play/2343020"
}
```

`GET /videos/{job_id}`

Mengambil status job dan variant hasil akhir.

Target ukuran:

- `480p <= 100 MB`
- `720p <= 150 MB`
- `1080p <= 199 MB`

Encoder default:

- `hevc_nvenc`
- decode CUDA + scaling GPU (`scale_cuda`/`scale_npp`) untuk prioritas speed

## Install Python Dependencies

```bash
python3 -m pip install -r requirements.txt
```
