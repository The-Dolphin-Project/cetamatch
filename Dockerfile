FROM python:3.11-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (cached layer) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Competition weights (cached layer) ---
# Downloaded from HuggingFace once at build time (~459 MB).
# This layer is only rebuilt when the URL or pip dependencies change,
# so day-to-day code deploys reuse it from Docker cache.
RUN python -c "\
import urllib.request, pathlib; \
dest = pathlib.Path('/app/models/efnv1b7_colab216.h5'); \
dest.parent.mkdir(parents=True, exist_ok=True); \
url = 'https://huggingface.co/yellowdolphin/happywhale-models/resolve/main/efnv1b7_colab216.h5'; \
print('Downloading competition weights (~459 MB) ...', flush=True); \
urllib.request.urlretrieve(url, dest); \
print(f'Saved  ({dest.stat().st_size // 1_048_576} MB)', flush=True); \
"

ENV MODEL_PATH=/app/models/efnv1b7_colab216.h5

# --- Application code (rebuilt on every deploy) ---
COPY app/ ./app/

RUN mkdir -p /app/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
