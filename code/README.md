# Message Notification Router

Deterministic Python router for the HackerRank Orchestrate **Message
Notification Router** challenge.

## Requirements

- Python 3.10+
- No mandatory third-party packages for the offline baseline

### Optional provider dependencies

```bash
pip install "openai>=1.40.0"
```

`code/requirements.txt` documents this optional dependency. Offline routing
needs no install beyond Python 3.10+.

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
| `OfflineMediaInterpreter` | `ROUTER_MEDIA_PROVIDER=offline` | Keeps `message_text`; leaves OCR/ASR empty; logs a concise warning; never fabricates |
| `LocalMediaInterpreter` | `ROUTER_MEDIA_PROVIDER=local` (also `auto` without API key when tools exist) | Tesseract OCR for images; local Whisper ASR for voice |
| `ApiMediaInterpreter` | `OPENAI_API_KEY` set and provider `auto`/`openai` | Vision JSON for images; Whisper API (+ optional structure) for voice |
| `CachedMediaInterpreter` | default unless disabled | Disk cache under `code/.cache/media/` keyed by content SHA-256 |

Perception returns structured `ContentSummary` channels only — never routing
decisions. Dump every media file with:

```bash
PYTHONPATH=code python -m router.media local code/evaluation/diagnostics/media_content_summaries.json
```

Local OCR/ASR needs `brew install tesseract ffmpeg` and
`pip install pillow pytesseract openai-whisper`.decisions. Prompts in `code/router/prompts.py` delimit source content and
state that it is untrusted data, not instructions. Safety still scans OCR/ASR
via `injection.py` before personalization.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Enables provider path when set | unset → offline |
| `ROUTER_MEDIA_PROVIDER` | `auto` \| `offline` \| `openai` \| `local` | `auto` |
| `ROUTER_IMAGE_MODEL` | Vision chat model | `gpt-4o-mini` |
| `ROUTER_ASR_MODEL` | Transcription model | `whisper-1` |
| `ROUTER_LOCAL_WHISPER_MODEL` | Local Whisper size (`tiny`/`base`/…) | `base` |
| `ROUTER_MEDIA_CACHE_DIR` | Cache directory | `code/.cache/media` |
| `ROUTER_MEDIA_CACHE_DISABLE` | `1` / `true` disables cache | unset |
| `ROUTER_DECISION_PROVIDER` | `auto` \| `offline` \| `openai` | `auto` |
| `ROUTER_DECISION_MODEL` | Decision chat model | `gpt-4o-mini` |

Without credentials or when a media file is missing, the router stays
runnable, warns, reduces confidence for uninterpreted media, and does not
invent OCR/ASR text. Exclude `code/.cache/` from submission zips (gitignored).

## Tests

```bash
python -m pytest code/tests -q
```

## Evaluation

Sample labels are evaluation-only. Production `code/main.py` refuses
`sample_messages.csv`. Run:

```bash
python code/evaluation/main.py --dataset dataset
```

This strips label columns before routing, scores action/type macro-F1,
confusion matrices, evidence overlap, confidence buckets, and invalid outputs,
then writes diagnostics under `code/evaluation/diagnostics/` (gitignored,
outside `dataset/`).

## Stage status

- Stage 1: typed loader, output validation, deterministic baseline CLI
- Stage 2: history↔event joins, recipient features, evidence retrieval
- Stage 3: safety gate — injection scanner + explainable risk before
  personalization (`HARD_MUTE_THRESHOLD=0.75`, notify ceiling floor `0.18`)
- Stage 4: MediaInterpreter (offline fallback + optional OpenAI + hash cache)
- Stage 5: constrained decision layer with temperature-0 model path and
  deterministic fallback
- Stage 6: sample evaluation harness + aggregate tuning loop
- Stage 7: production audit + submission packaging

## Stage 5 contextual decision

`decide()` receives only normalized `ContentSummary`, deterministic features,
priority/safety summaries, and ranked evidence. It returns validated JSON
fields (`action`, `message_type`, `reason`, `confidence`, `evidence_message_ids`).

| Mode | When | Behavior |
|---|---|---|
| Deterministic fallback | default / no key / `ROUTER_DECISION_PROVIDER=offline` | Priority action + signal-based type/reason/confidence |
| OpenAI decision | `OPENAI_API_KEY` + provider `auto`/`openai` | Temperature `0` JSON; post-validated |

Safety is authoritative after any model response:

- hard mute cannot be weakened to digest/notify
- notify ceiling cannot be weakened to notify
- evidence IDs must come from the retriever allow-list (else `none`)
- generic boilerplate reasons are rejected (fallback used)

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
