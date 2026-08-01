# Message Notification Router

Deterministic Python router for the HackerRank Orchestrate **Message
Notification Router** challenge.

## Requirements

- Python 3.10+
- No mandatory third-party packages for the offline Stage 1 baseline

Optional provider SDKs will be documented when multimodal API perception is
added. Credentials must come from environment variables only — never commit
secrets.

## Dataset

Point the CLI at the challenge `dataset/` directory. Production routing reads
participant-facing CSVs only and **never** loads `sample_messages.csv`.

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

## Tests

```bash
python -m pytest code/tests -q
```

## Stage status

- Stage 1: typed loader, output validation, deterministic baseline CLI
- Stage 2: history↔event joins, recipient features, evidence retrieval
- Stage 3: safety gate — injection scanner + explainable risk before
  personalization (`HARD_MUTE_THRESHOLD=0.75`, notify ceiling floor `0.18`)
- Later: multimodal perception, constrained model decisions, eval harness

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