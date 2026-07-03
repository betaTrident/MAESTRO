# MAESTRO

MAESTRO is an agentic fraud-detection pipeline for the Reply challenge datasets. It combines deterministic risk features with LLM judging to identify suspicious outgoing transactions across dataset folders containing:

- `transactions.csv`
- `users.json`
- `locations.json`
- `mails.json`
- `sms.json`
- optional `audio/`

## Architecture

The pipeline is designed around a compact orchestrator + LLM-judge flow:

1. Input loading
   - Loads transaction, user, location, email, SMS, and optional audio metadata.
   - Normalizes each dataset into the structures used by the risk pipeline.

2. Risk orchestration
   - Generates a Langfuse session ID through `generate_session_id()`.
   - Builds deterministic risk features for user-owned outgoing transactions.
   - Shortlists the highest-risk candidates before making LLM calls.

3. Primary LLM judge
   - Reviews shortlisted candidates with transaction details, profile stats, engineered risk features, recent activity, and communications context.
   - Returns a fraud decision, confidence score, and economic risk estimate.

4. Optional reviewer
   - Runs only for uncertain, economically relevant cases.
   - Improves decision stability without routing every candidate through another model call.

5. Output formatter
   - Writes fraud transaction IDs to `outputs/<dataset>_fraud_ids.txt`.
   - Writes run metadata to `outputs/<dataset>_fraud_meta.json`.

## Code Layout

- `main.py` - dataset loading, feature engineering, fraud scoring, and CLI entrypoint.
- `orchestator.py` - OpenRouter/Langfuse integration, LLM call wrapper, session tracking, and cost/latency reporting.
- `data/` - challenge datasets.
- `outputs/` - generated fraud IDs and run metadata.
- `instructions.txt` - challenge configuration notes and required environment variable names.

## Environment

Create a local `.env` file with your provider credentials:

```bash
OPENROUTER_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://challenges.reply.com/langfuse
```

Do not commit `.env`. The repository ignores it because it contains private API keys.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run all datasets, up to five by default:

```bash
python main.py
```

Run a single dataset:

```bash
python main.py --dataset dataset1
```

Use custom paths or worker settings:

```bash
python main.py --data-root data --output-dir outputs --max-datasets 5 --max-workers 4
```

## Outputs

Each dataset run produces:

- `outputs/<dataset>_fraud_ids.txt`
- `outputs/<dataset>_fraud_meta.json`

The console also prints `print_results()` metrics for each session, including latency, estimated cost, calls, and token usage. If Langfuse polling is delayed or unavailable, MAESTRO falls back to local per-session metrics.
