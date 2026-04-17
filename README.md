# Split Video Pipeline

Repo ini sekarang menyediakan 2 service API terpisah:

- `downloader_service.py`
  Tugasnya hanya download video dengan `yt-dlp --cookies-from-browser chrome`, parse episode dari nama file, lalu upload file mentah ke transcoder API.
- `transcoder_service.py`
  Tugasnya menerima file upload, kompres ke `480p/720p/1080p`, upload ke Catbox, lalu simpan metadata ke SQLite.

## 1. Downloader Service

Dependency sistem:

- `yt-dlp`
- Google Chrome profile yang bisa dibaca `yt-dlp --cookies-from-browser chrome`

Env penting:

```bash
export TRANSCODER_API_URL="http://ip-server-gpu:9000/transcode/upload"
export TRANSCODER_API_VERIFY_SSL="1"
```

Jika reverse proxy HTTPS di server transcoder masih bermasalah atau self-signed, sementara bisa pakai:

```bash
export TRANSCODER_API_VERIFY_SSL="0"
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

Mengambil status job downloader dan variant hasil akhir yang dikembalikan transcoder service.

## 2. Transcoder Service

Service ini dijalankan di mesin GPU NVIDIA.

Dependency sistem:

- `ffmpeg`
- `ffprobe`
- `curl`

Env opsional:

```bash
export CATBOX_USERHASH="userhash_akun_catbox_anda"
export NVENC_PRESET="p1"
export NVENC_CQ="25"
```

Jalankan:

```bash
python3 transcoder_service.py --dev
```

Endpoint:

`POST /transcode/upload`

Form field:

- `job_id`
- `title`
- `episode`
- `file`

Target ukuran:

- `480p <= 100 MB`
- `720p <= 150 MB`
- `1080p <= 199 MB`

Encoder default:

- `hevc_nvenc`
- decode CUDA + scaling GPU (`scale_cuda`/`scale_npp`) untuk prioritas speed

Upload ke Catbox akan di-retry sampai 3 kali sebelum request dianggap gagal.

## Install Python Dependencies

```bash
python3 -m pip install -r requirements.txt
```
