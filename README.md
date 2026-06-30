# Cat Vocalization Engine 🐱🔊

An advanced, digital signal processing (DSP) powered system for converting human speech into realistic cat vocalizations and interpreting cat meows into human-readable intent. The project features a high-fidelity rendering pipeline, a machine-learning-inspired dataset curation tool, a command-line interface, and a production-ready FastAPI REST service.

---

## 🌟 Key Features

* **Human-to-Cat Synthesis (DSP Render)**: Analyzes prosody, pitch, duration, and intensity of human vocal inputs and translates them into matching cat vocalizations using a curated database of cat sounds.
  * **Style Profiles**: Customize DSP rendering behavior with `natural`, `expressive`, or `subtle` styles.
* **Cat-to-Human Translation (Acoustic Interpreter)**: Analyzes acoustic features of meows, chirps, purrs, and trills to map them to discrete feline intents (e.g., attention, request, urgent, annoyed, lonely, calm, playful, content) with confidence scoring.
* **Text-to-Speech (TTS) Integration**: Automatically speaks meow translations aloud using high-quality neural voices (`edge-tts`) or localized system synthesis (`pyttsx3`).
* **Dataset Curation Pipeline**: Auto-segments, filters (rejects low quality/silence/noise), clusters (MFCC voice similarity), and classifies raw cat audio recordings into distinct vocal categories.
* **Interactive FastAPI Web Server**: Fully featured REST API with bearer token authentication, auto-generated Swagger UI, automated temp file cleanup, and a web-based upload page for manual testing.
* **Docker Support**: Containerized build with built-in health checks and service orchestration via Docker Compose.

---

## 📂 Project Architecture

```filepath
├── engine/                  # Core audio processing engine
│   ├── __init__.py
│   ├── audio_io.py          # Audio loading, sample rate validation, and ffmpeg fallbacks
│   ├── cat_interpreter.py   # Acoustic feature extraction and cat intent translation
│   ├── curated_dataset.py   # Loader, indexer, and fallback managers for cat clips
│   ├── dsp.py               # DSP primitives: pitch shift, time stretch, crossfades
│   ├── emotion_presets.py   # Feline emotion maps and acoustic characteristics
│   ├── prosody.py           # Onset-based speech segment and energy tracking
│   ├── renderer.py          # Human-to-cat alignment and clip-mixing logic
│   └── tts.py               # Text-to-speech rendering (edge-tts / pyttsx3)
├── tools/
│   └── curate_dataset.py    # Raw audio curation and classification pipeline
├── api.py                   # REST API server (FastAPI, Swagger documentation)
├── cli.py                   # Command-line interface for translation & rendering
├── Dockerfile               # Production container definition
├── docker-compose.yml       # Docker Compose service Orchestration
├── requirements.txt         # Project package dependencies
└── README.md                # Project documentation
```

---

## 🛠️ System Requirements & Installation

### Prerequisite System Libraries
This project requires system-level codecs to process various audio formats and synthesize text-to-speech.

* **Linux (Debian/Ubuntu)**:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1 espeak-ng
  ```
* **macOS**:
  ```bash
  brew install ffmpeg libsndfile
  ```
* **Windows**:
  Ensure `ffmpeg` is installed and added to your system `PATH`.

### Python Virtual Environment Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
2. Install Python packages:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

---

## 💻 CLI Usage

The Command-Line Interface (`cli.py`) handles conversion and translation.

### 1. Convert Human Speech to Cat Vocalizations
To translate a human voice recording into cat sounds, supply the path to the dataset folder:
```bash
python cli.py input.wav --dataset "./curated_cat_sounds" --out output.wav
```

**Additional Arguments:**
* `--style`: DSP intensity profiles: `natural` (default), `expressive` (more dynamic pitch/time stretching), or `subtle` (restrained alignment).
* `--sr`: Internal sample rate in Hz (default: `22050`).
* `--quiet`: Suppress segment-by-segment debug logs.

### 2. Translate Cat Vocalizations to Human English
Use the `--to-human` flag to interpret a cat's meows:
```bash
python cli.py cat_meow.wav --to-human
```

**Additional Translation Options:**
* `--json`: Outputs the complete translation response object, including per-segment signals, features, and confidence scores as JSON.
* `--speak`: Speaks the English translation aloud using the TTS system.
* `--voice-out <path>`: Saves the TTS voice translation as a WAV file.
* `--voice`: Choose neural TTS voice model: `en-US-GuyNeural` (default) or `en-GB-SoniaNeural`.

---

## 🌐 API Service (FastAPI)

The FastAPI server (`api.py`) exposes interactive REST endpoints for integration.

### Run Server Locally
Start the development server using uvicorn:
```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```
Access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

### Authentication
Protected endpoints require an HTTP Bearer Token.
1. Set the token as an environment variable on the server:
   ```bash
   export CAT_API_TOKEN="your-secure-secret-token"
   ```
2. Include the following header in your HTTP requests:
   ```http
   Authorization: Bearer your-secure-secret-token
   ```

### Core API Endpoints

* **`GET /health`** (Public)
  Checks API readiness and confirms if the default curated dataset is loaded.
* **`GET /`** (Public)
  A simple HTML upload interface for testing conversion and translation directly from the browser.
* **`POST /render`** (Protected)
  Converts uploaded human vocal files into a cat-like WAV file.
* **`GET /render-demo`** (Protected)
  Renders a short, pre-synthesized demo clip on the server to check engine functionality without uploading files.
* **`POST /translate-cat`** (Protected)
  Translates uploaded cat meows, returning a JSON report with intents, confidence, and acoustic measurements.
* **`POST /translate-cat-speech`** (Protected)
  Returns the translated cat audio synthesized directly as a spoken English WAV file.

---

## 🗄️ Dataset Curation Pipeline

If you want to train or curate your own feline vocal datasets, use the curation pipeline tool:
```bash
python tools/curate_dataset.py --dataset "/path/to/raw/cat/recordings" --output "./custom_curated_sounds"
```

### Pipeline Pipeline Stages:
1. **Audio Load**: Recursively scans directory for WAV, MP3, and M4A files.
2. **Onset Detection**: Segments raw recordings into isolated vocal events.
3. **Quality Filters**: Discards segments that are silent, clipped, overly noisy, too long, or too short.
4. **Scoring**: Ranks remaining candidates based on SNR (Signal-to-Noise Ratio), spectral clarity, and energy envelope.
5. **Clustering**: Utilizes MFCC features and voice similarity to filter outliers and group together clips from the dominant vocal source.
6. **Intent Classification**: Evaluates physical characteristics (pitch contour, duration, spectral centroid, autocorrelation) to classify the clips into six main categories:
   * `neutral_attention`
   * `calm_soft`
   * `lonely_falling`
   * `playful_chirp_trill`
   * `annoyed_urgent`
   * `purr`
7. **Metadata Output**: Generates `metadata.json` mapping files to their classified properties.

---

## 🐳 Docker Deployment

The application includes containerization assets for cloud deployments.

### Local Docker Build
1. Build the Docker image:
   ```bash
   docker build -t cat-sound-converter:latest .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 -e CAT_API_TOKEN="your_api_token" cat-sound-converter:latest
   ```

### Docker Compose Orchestration
Manage container builds and configurations easily:
```bash
# Start API container
CAT_API_TOKEN="your_token_here" docker-compose up -d --build

# Shutdown container
docker-compose down
```
The compose template utilizes built-in container-level healthchecks which poll `GET /health` every 30 seconds.
