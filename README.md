# finFindR

Automated dorsal fin identification as a self-hosted REST API. Upload a photo of a
dolphin (or other cetacean) dorsal fin and get back the closest matches from a
labelled catalogue.

This project is a port of [haimeh/finFindR](https://github.com/haimeh/finFindR) —
originally written in R using MXNet — into a modern Python web service. The core
image processing pipeline (preprocessing, blob detection, Canny edge detection, and
A\* trailing-edge tracing) is a faithful port of the original C++ and R code. The
embedding network replaces the original MXNet model with a PyTorch ResNet50 backbone
fine-tuned with triplet loss, which requires significantly less training data thanks to
ImageNet pretraining.

---

## How it works

```
Photo upload (JPEG/PNG)
        │
        ▼
  Preprocessing       Resize to 300px, glare removal
        │
        ▼
  Fin cropping        Blob detection + bounding box
        │
        ▼
  Edge detection      Canny edge map  (port of imageToFeatureProcessing.cpp)
        │
        ▼
  A* trace            Weighted pathfinding along the trailing edge  (port of astar.cpp)
        │
        ▼
  Embedding           ResNet50 → 512-D L2-normalised vector
        │
        ▼
  Catalogue match     pgvector similarity search (Supabase)
        │
        ▼
  JSON response       embedding + trace + top-N matches
```

---

## Prerequisites

- Python 3.10+ (for local development and training)
- Docker (for containerised deployment)
- [Supabase](https://supabase.com) project (free tier works)
- [Fly.io](https://fly.io) account (optional — for cloud deployment)

---

## Supabase setup

Run the following once in your Supabase project's SQL editor:

```sql
-- 1. Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Fin embeddings table
CREATE TABLE fins (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    embedding   vector(512) NOT NULL,
    image_path  TEXT,
    added_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Index for fast approximate nearest-neighbour search
CREATE INDEX fins_embedding_idx
    ON fins USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);

-- 4. Row Level Security — deny all public access
ALTER TABLE fins ENABLE ROW LEVEL SECURITY;

-- 5. Similarity search function
CREATE OR REPLACE FUNCTION match_fins(
    query_embedding vector(512),
    match_count     int DEFAULT 10
)
RETURNS TABLE (id text, label text, distance float)
LANGUAGE sql
SECURITY INVOKER
AS $$
    SELECT id, label, (embedding <-> query_embedding)::float AS distance
    FROM   fins
    ORDER  BY embedding <-> query_embedding
    LIMIT  match_count;
$$;
```

Then create a **Storage bucket** named `finfinder` (private) to hold your model
weights file.

---

## Environment variables

Copy `.env.example` to `.env` and fill in your values:

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key (Project Settings → API) |
| `MODEL_PATH` | Local path for model weights (default: `/app/models/fin_embedder.pt`) |
| `SUPABASE_STORAGE_BUCKET` | Storage bucket name (default: `finfinder`) |
| `API_KEY` | Key required on all API requests — leave unset to disable auth |

---

## Local development

```bash
git clone https://github.com/your-org/finfinder.git
cd finfinder
pip install -r requirements.txt
cp .env.example .env   # fill in your Supabase credentials
uvicorn app.main:app --reload
```

API docs will be available at `http://localhost:8000/docs`.

Or with Docker:

```bash
docker compose up --build
```

---

## API reference

All endpoints except `/health` require an `X-API-Key` header when `API_KEY` is
configured on the server.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/identify` | Upload a fin photo → get embedding + top-N catalogue matches |
| `POST` | `/catalogue/add` | Add a labelled fin to the catalogue |
| `GET` | `/catalogue` | List all catalogue entries |
| `GET` | `/catalogue/{id}` | Get a single entry with its embedding |
| `DELETE` | `/catalogue/{id}` | Remove an entry |
| `GET` | `/health` | Liveness check + catalogue size (no auth required) |

### Identify a fin

```bash
curl -X POST http://localhost:8000/identify \
  -H "X-API-Key: your-key" \
  -F "file=@dolphin.jpg" \
  -F "top_n=5"
```

### Add a fin to the catalogue

```bash
curl -X POST http://localhost:8000/catalogue/add \
  -H "X-API-Key: your-key" \
  -F "file=@dolphin.jpg" \
  -F "label=Flipper" \
  -F "fin_id=flipper-001"
```

---

## Pre-trained model weights

Pre-trained weights are available on request. They were trained on a small catalogue
of bottlenose dolphins from [The Dolphin Project](https://www.dolphinproject.com)
using roughly 150 individuals with 2 dorsal fin photos each. The file is too large
for GitHub — once obtained, upload it to your Supabase Storage bucket at
`finfinder/models/fin_embedder.pt` and the app will download it automatically on
startup.

**These weights will likely underperform on other populations or species.** The
pretrained ResNet50 backbone still provides useful features out of the box, but for
best results you should train the model on your own labelled catalogue.

---

## Training your own model

### 1. Prepare your data

Organise your photos into one folder per individual, using the individual's ID as the
folder name. You need **at least 2 photos per individual** — 3–10 is ideal.

```
training_data/
  dolphin_001/
    left_dorsal.jpg
    right_dorsal.jpg
  dolphin_002/
    photo1.jpg
    photo2.jpg
    photo3.jpg
```

### 2. Train locally

Train on your local machine first — it's faster to iterate than rebuilding a container.

```bash
python -m app.training.train \
  --data-dir ./training_data \
  --output ./models/fin_embedder.pt \
  --epochs 50
```

Watch the output each epoch:

```
Epoch  1/50  train=0.2841  val=0.3102
Epoch  2/50  train=0.2213  val=0.2891
  → Saved best model to ./models/fin_embedder.pt
```

- **`train` loss** — error on photos the model is learning from. Should decrease.
- **`val` loss** — error on held-out photos. This is your honest quality measure.
- A val loss in the **0.05–0.20** range indicates a well-trained model.
- The best checkpoint is saved to `--output` whenever val loss hits a new low.

Additional options:

```
--epochs      Number of training epochs (default: 50)
--batch-size  Batch size (default: 32; reduce if you run out of memory)
--lr          Learning rate (default: 1e-4)
--val-split   Fraction of data held out for validation (default: 0.15)
```

### 3. Upload weights to Supabase Storage

Once you are satisfied with training, upload your `fin_embedder.pt` to the
`finfinder` Supabase Storage bucket at path `models/fin_embedder.pt`.

The running app will download the weights automatically on its next cold start if
the local file is not present.

---

## Deploying to Fly.io

```bash
# Install the CLI
brew install flyctl
flyctl auth login

# Create the app (run once)
flyctl launch

# Set your secrets
flyctl secrets set \
  SUPABASE_URL=https://your-project.supabase.co \
  SUPABASE_SERVICE_ROLE_KEY=your-service-role-key \
  API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Deploy
flyctl deploy
```

The included `fly.toml` configures 1 GB of RAM (required for PyTorch) and a
60-second startup grace period so the health check doesn't fire before the model
has finished loading.

For subsequent updates, `flyctl deploy` is all you need.

---

## Reporting bugs

Please open an issue on the [GitHub Issues](../../issues) page. Include the API
response, the Python version, and whether you are running locally or via Docker.

---

## Citation

If you use finFindR in published research, please cite both this port and the
original work:

```
Craig, S. (2026). finFindR Python port. GitHub.
https://github.com/The-Dolphin-Project/finfinderWeb

Haimeh, A. et al. finFindR: Automated identification of individual cetaceans
from dorsal fin photographs. GitHub.
https://github.com/haimeh/finFindR
```

---

## Acknowledgments

- **[haimeh/finFindR](https://github.com/haimeh/finFindR)** — the original R/C++
  implementation that this project ports.
- **[The Dolphin Project](https://www.thedolphinproject.org)** — provided the training
  catalogue of bottlenose dolphin dorsal fin photographs included with this release.

---

## License

MIT — see [LICENSE](LICENSE).
