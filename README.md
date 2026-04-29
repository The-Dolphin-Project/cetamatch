# CetaMatch

Automated dorsal fin identification as a self-hosted REST API. Upload a photo of a dolphin (or other cetacean) dorsal fin and get back the closest matches from a labelled catalogue.

CetaMatch combines two open-source efforts:

- **Image processing pipeline** — a faithful Python port of
  [haimeh/finFindR](https://github.com/haimeh/finFindR) (originally R/C++):
  preprocessing, blob detection, Canny edge detection, and A\* trailing-edge tracing.
- **Embedding model** — EfficientNet-B7 fine-tuned with Sub-center ArcFace loss from
  the [Happywhale Kaggle competition](https://www.kaggle.com/competitions/happy-whale-and-dolphin),
  trained on 15 587 individual cetacean identities across 30 species.

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
  Embedding           EfficientNet-B7 → 2560-D L2-normalised vector
        │
        ▼
  Catalogue match     pgvector similarity search (Supabase)
        │
        ▼
  JSON response       embedding + trace + top-N matches
```

---

## Prerequisites

- Python 3.11 (for local development)
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
    embedding   vector(2560) NOT NULL,
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
    query_embedding vector(2560),
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

---

## Environment variables

Copy `.env.example` to `.env` and fill in your values:

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key (Project Settings → API) |
| `API_KEY` | Key required on all API requests — leave unset to disable auth |

---

## Local development

```bash
git clone https://github.com/your-org/finfinder.git
cd finfinder
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your Supabase credentials
uvicorn app.main:app --reload
```

On first startup the EfficientNet-B7 competition weights (~459 MB) are downloaded
from HuggingFace automatically and cached in `./models/`.

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

## Embedding model

CetaMatch uses the **EfficientNet-B7 + Sub-center ArcFace** model from the
[Happywhale Kaggle competition](https://www.kaggle.com/competitions/happy-whale-and-dolphin),
trained by [yellowdolphin](https://huggingface.co/yellowdolphin/happywhale-models)
on 15 587 individual cetacean identities across 30 species.

Weights are downloaded automatically from HuggingFace on first run and cached
locally. The Docker image bakes them in at build time so cold starts need no
network access.

> **Note:** The model performs well across cetacean species out of the box.
> Accuracy will vary by photo quality and species. Fine-tuning is not currently
> supported.

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

The included `fly.toml` configures 2 GB of RAM (required for EfficientNet-B7 +
TensorFlow) and a 60-second startup grace period so the health check doesn't fire
before the model has finished loading.

For subsequent updates, `flyctl deploy` is all you need.

---

## Reporting bugs

Please open an issue on the [GitHub Issues](../../issues) page. Include the API
response, the Python version, and whether you are running locally or via Docker.

---

## Citation

If you use CetaMatch in published research, please cite both this port and the
original works:

```
Craig, S. (2026). CetaMatch. GitHub.
https://github.com/The-Dolphin-Project/CetaMatch

Haimeh, A. et al. finFindR: Automated identification of individual cetaceans
from dorsal fin photographs. GitHub.
https://github.com/haimeh/finFindR

Happywhale and Dolphin Identification. Kaggle competition, 2022.
https://www.kaggle.com/competitions/happy-whale-and-dolphin
```

---

## Acknowledgments

- **[haimeh/finFindR](https://github.com/haimeh/finFindR)** — the original R/C++
  image processing pipeline that this project ports.
- **[yellowdolphin](https://huggingface.co/yellowdolphin/happywhale-models)** — the
  Happywhale competition EfficientNet-B7 + Sub-center ArcFace weights used for embedding.
- **[Happywhale & Dolphin Kaggle competition](https://www.kaggle.com/competitions/happy-whale-and-dolphin)** —
  the dataset and competition that produced the embedding model.
- **[The Dolphin Project](https://www.thedolphinproject.org)** — operates the catalogue
  that this API was built to serve.

---

## License

MIT — see [LICENSE](LICENSE).
