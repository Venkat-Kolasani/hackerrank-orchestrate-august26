# Message Notification Router

Deterministic Python router for the HackerRank Orchestrate **Message
Notification Router** challenge.

## Requirements

- Python 3.10+
- No mandatory third-party packages for the offline baseline (Stages 1–3 +
  offline media fallback)

### Optional provider dependencies (Stage 4)

```bash
pip install -r code/requirements.txt
# installs openai>=1.40 when you want ApiMediaInterpreter
```

Credentials must come from environment variables only — never commit secrets.

## Dataset

Point the CLI at the challenge `dataset/` directory. Production routing reads
participant-facing CSVs only and **never** loads `sample_messages.csv`.

Image and voice paths in `images.csv` / `voice_notes.csv` are relative to
`dataset/` (for example `media/images/img_001.jpg`).

## Run

From the repository root:

```bash
python code/main.py --dataset dataset
```

This writes `dataset/output.csv` with one row per `dataset/messages.csv`
message and columns:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

Custom paths:

```bash
python code/main.py --dataset /path/to/dataset --output /path/to/output.csv
```

## Multimodal perception (Stage 4)

`code/router/media.py` implements `MediaInterpreter`:

| Mode | When | Behavior |
|---|---|---|
| `OfflineMediaInterpreter` | default / no key / `ROUTER_MEDIA_PROVIDER=offline` | Keeps `message_text`; leaves OCR/ASR empty; logs a concise warning; never fabricates |
| `ApiMediaInterpreter` | `OPENAI_API_KEY` set and provider `auto`/`openai` | Vision JSON for images; Whisper (+ optional structure) for voice |
| `CachedMediaInterpreter` | default unless disabled | Disk cache under `code/.cache/media/` keyed by content SHA-256 |

Perception returns structured `ContentSummary` channels only — never routing
decisions. Prompts in `code/router/prompts.py` delimit source content and
state that it is untrusted data, not instructions. Safety still scans OCR/ASR
via `injection.py` before personalization.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Enables provider path when set | unset → offline |
| `ROUTER_MEDIA_PROVIDER` | `auto` \| `offline` \| `openai` | `auto` |
| `ROUTER_IMAGE_MODEL` | Vision chat model | `gpt-4o-mini` |
| `ROUTER_ASR_MODEL` | Transcription model | `whisper-1` |
| `ROUTER_MEDIA_CACHE_DIR` | Cache directory | `code/.cache/media` |
| `ROUTER_MEDIA_CACHE_DISABLE` | `1` / `true` disables cache | unset |

Without credentials or when a media file is missing, the router stays
runnable, warns, reduces confidence for uninterpreted media, and does not
invent OCR/ASR text. Exclude `code/.cache/` from submission zips (gitignored).

## Tests

```bash
python -m pytest code/tests -q
```

Media-specific coverage lives in `code/tests/test_media.py` (path resolution,
offline non-fabrication, unavailable media, cache hashing, prompt delimiters).

## Stage status

- Stage 1: typed loader, output validation, deterministic baseline CLI
- Stage 2: history↔event joins, recipient features, evidence retrieval
- Stage 3: safety gate — injection scanner + explainable risk before
  personalization (`HARD_MUTE_THRESHOLD=0.75`, notify ceiling floor `0.18`)
- Stage 4: MediaInterpreter (offline fallback + optional OpenAI + hash cache)
- Later: constrained model decisions, eval harness

## Stage 2 behavior

The router joins `message_history.csv` to `message_events.csv` by
`(user_id, message_id)` and builds recipient-specific trust, relevance,
repetition, quiet-hours, and notification-load features from users, groups,
memberships, businesses, user-business history, and daily summaries.

Evidence IDs (up to 3) are chosen only from that recipient's history,
preferring the same group/business/sender, then similar templates/content,
and only when the observed reaction supports the decision. Otherwise the
field is `none`.

## Stage 3 safety gate

`code/router/injection.py` deterministically scans full normalized content
(text + OCR + ASR) for override / label-bait / fake-JSON framing.
`code/router/safety.py` combines those hits with OTP/credential/payment
lures, coercive urgency, domain mismatch, report/forward volume, and young
account/domain age into `risk_score` **before** priority personalization.

- `risk_score >= 0.75` → hard mute (`scam` or `spam`); history cannot restore notify
- `risk_score >= 0.18` → notify ceiling (max action digest)
- Legitimate trusted payment receipts must not hard-mute; untrusted OTP /
  payment requests elevate risk on the mute path

Message and media text are always treated as untrusted data, never
instructions.
