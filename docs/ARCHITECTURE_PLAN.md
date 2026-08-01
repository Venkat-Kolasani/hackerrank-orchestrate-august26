# Message Notification Router — Architecture Plan

## Objective and non-negotiables

Generate `dataset/output.csv` with exactly one prediction per row in
`dataset/messages.csv` and this exact column order:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

The router must inspect all participant-facing data under `dataset/`, including
image and voice media when it is available. It must not use labels from
`sample_messages.csv` in the prediction path. Use that file only in the
evaluation command.

The design optimizes for the scoring dimensions in the problem statement:
action, type, concise useful reasons, relevant evidence, and calibrated
confidence. It also makes false-positive interruptions expensive: a weak case
should become `digest`, never an unnecessary `notify`.

## Design decisions

1. **Hybrid, staged routing.** Metadata, behavioral history, and risk signals
   are calculated deterministically in Python. An LLM is used only to interpret
   normalized multimodal content and resolve the remaining contextual cases.
2. **Safety precedes personalization.** High-confidence phishing, scam, or
   unsafe content is forced to `mute`; history can never override it.
3. **Personalization is a primary feature.** Sender/group/business-specific
   behavior wins over global popularity. The same poster can consequently be
   `digest` for one recipient and `mute` for another.
4. **Evidence is retrieved before the decision.** Candidate evidence is limited
   to historical messages for the recipient, prioritized by the same
   conversation or sender and matching behavioral outcomes.
5. **Prompt injection is treated as content.** Incoming text, OCR, and ASR are
   untrusted data delimited in prompts. They cannot instruct the router.
6. **Provider-independent and deterministic when possible.** A provider
   interface supports a configured multimodal model; the default rule engine is
   usable without an API key. Temperature is zero, media interpretations are
   cached by file hash, and decisions receive a stable message ordering.

The supplied examples confirm two important cases: `sample_msg_053` embeds an
instruction to the classifier and must still be judged as a scam, while
`sample_msg_044` and `sample_msg_045` reuse `img_008` but receive different
actions because their recipients' histories differ.

## Processing architecture

```text
dataset CSVs + media
        |
        v
1. Load and validate --> typed message/context records
        |
        +--> 2. Media perception --> content summary (text/OCR/caption/ASR)
        |
        v
3. Feature and evidence engine
   - trust, urgency, engagement, repetition, quiet-hours, notification load
        |
        v
4. Safety gate --> forced mute for clear scams/spam
        |
        v
5. Structured decision --> notify / digest / mute + type + reason + confidence
        |
        v
6. Output validation --> dataset/output.csv
```

### 1. Data loading and validation

`code/router/data.py` will load only the files specified in the challenge
contract. It will:

- normalize missing identifiers and parse timestamps;
- join `message_history.csv` to `message_events.csv` by `(user_id, message_id)`;
- build lookup indexes for users, groups, group membership, businesses, and
  user-business history;
- resolve `images.csv` and `voice_notes.csv` paths relative to `dataset/`;
- fail clearly when a required input is absent and report unavailable media
  without silently inventing a description.

`sample_messages.csv` must be loaded only from `code/evaluation/`, never by
`code/main.py` or router modules used by it.

### 2. Multimodal perception

All later stages consume a `ContentSummary`:

- **Text:** preserve the original content as data.
- **Image:** extract readable poster/screenshot details (dates, amount, phone
  numbers, URLs, QR/payment requests) and a short scene/category description.
- **Voice:** transcribe and extract deadlines, amounts, people, requests, and
  urgency.

Implement `MediaInterpreter` behind a small interface:

- `ApiMediaInterpreter` uses a multimodal model plus an ASR endpoint when
  `OPENAI_API_KEY` (or another documented provider key) is present.
- `OfflineMediaInterpreter` returns text-only summaries and emits an explicit
  warning for unavailable image/audio analysis. It exists for reproducibility
  and testability, not as the expected high-score mode.
- Cache normalized media output under `code/.cache/` by content hash; this
  cache must be excluded from the submission zip.

The perception prompt returns strict JSON and explicitly states that all
message content is untrusted quoted data, never routing instructions.

### 3. Deterministic feature engine

`code/router/features.py` produces auditable features for each incoming
message.

**Trust and sender context**

- Group: type, recipient role, whether the author is a group admin, member
  mute state, and activity.
- Business: verified status, official-domain mismatch, account age, sender
  domain age, report volume.
- Personal: prior recipient/sender interactions and forward volume.

**Personal relevance**

- recipient's open, reply, dismiss, mute, and report rates;
- same group, sender, or business historical behavior;
- business relationship, last activity, promotion consent/opt-out;
- group member engagement and the daily notification-load baseline;
- direct mentions and same-day operational deadlines.

**Risk, urgency, and repetition**

- risk indicators: OTP/password/card requests, payment links, domain mismatch,
  credential verification, high forwarding, threats/urgency, new untrusted
  sender, and previous reports;
- urgent indicators: immediate/same-day deadline, actionable operational
  update, safety/health issue, direct mention, trusted admin;
- repetition: normalized token similarity and exact/template hash against
  recipient history, then prior dismiss/mute/report outcomes.

Quiet hours should reduce non-urgent `notify` to `digest`, including windows
that cross midnight. It must not hide a safety override or a high-confidence
time-critical message.

### 4. Evidence retrieval

`code/router/evidence.py` retrieves up to three IDs, in this order:

1. Same recipient and same conversation entity (group, business, or sender).
2. Same recipient and matching message category/template.
3. Similar content for the recipient only, using deterministic TF-IDF/cosine
   ranking if available.

Boost candidates whose reaction directly supports the decision:

- `notify`: opened/replied quickly;
- `digest`: opened but not time-sensitive;
- `mute`: dismissed, muted, or reported.

Do not use an unrelated ID merely to fill the field. Output `none` where
evidence does not support the prediction.

### 5. Safety gate and decision policy

Use a conservative rule-based safety gate before any LLM decision:

- `risk_score >= hard_threshold` => `mute`, `scam` or `spam`;
- a medium risk score is passed to the structured decision model with explicit
  high-risk context;
- never use the LLM to weaken a hard safety decision.

For safe messages, compute an interrupt priority:

```text
priority = urgency + direct_mention + sender_trust + personal_relevance
           + positive_history - repetition - low_engagement - quiet_hour_cost
```

Use this only to provide stable directional guidance to the decision model,
with a deterministic fallback:

- high priority: `notify`;
- moderate priority: `digest`;
- low priority/repetitive/unwanted: `mute`.

The structured model receives only the normalized content, calculated
features, and evidence summaries. It returns JSON constrained to the
challenge's allowed actions and message types. Post-validation enforces action
and type enums, evidence membership, reason length, and confidence range.

### 6. Reasons and confidence

Reasons must name the material signal, not generic boilerplate. Examples:

- "A verified delivery account sent an update matching the user's recent
  order."
- "This user previously dismissed similar promotions from this business."
- "The message requests an OTP through an untrusted payment domain."

Confidence starts from decision margin and signal agreement, then is reduced
for unavailable media, conflicting signals, or low-quality evidence. A
calibration step fitted only on sample evaluations may adjust buckets globally;
it must not learn message-level lookup answers.

## Proposed implementation layout

```text
code/
├── main.py                    # CLI: route dataset/messages.csv
├── requirements.txt
├── README.md                   # installation, configuration, run commands
├── router/
│   ├── data.py                 # models, loading, joins, validation
│   ├── media.py                # cached API/offline media interpreters
│   ├── features.py             # deterministic signals
│   ├── evidence.py             # evidence ranking
│   ├── safety.py               # risk scoring and hard overrides
│   ├── decision.py             # schema-constrained model + fallback
│   ├── prompts.py              # versioned prompt constants
│   ├── output.py               # exact output validation/writer
│   └── types.py                # dataclasses and allowed enums
└── evaluation/
    ├── main.py                 # sample-only evaluation CLI
    ├── metrics.py              # F1, evidence, calibration, diagnostics
    └── split.py                # optional deterministic validation splits
docs/
├── ARCHITECTURE_PLAN.md
└── BUILD_PROMPTS.md
```

## Evaluation workflow

`python code/evaluation/main.py --dataset dataset` must:

1. Run the same router on `sample_messages.csv`, passing it as the input
   source rather than looking up its labels.
2. Report action and type macro-F1, a confusion matrix, evidence exact/partial
   match, confidence bucket accuracy, and invalid output count.
3. Save predictions and diagnostics outside `dataset/` to avoid contaminating
   the final artifact.
4. Confirm that production `main.py` imports no sample label fields.

Tune thresholds and prompt wording from aggregate evaluation errors—not
individual answers. Test the offline fallback, missing media paths, malformed
CSV rows, quiet hours crossing midnight, direct mentions in muted groups,
domain mismatches, and injection-like text.

## 24-hour execution order

1. Establish the typed loader, schema validator, exact output writer, and
   deterministic baseline.
2. Build history joins, contextual features, evidence retrieval, and evaluation
   reports.
3. Add the safety gate, then test it against scam-like and prompt-injection
   examples.
4. Add cached image/voice interpretation and structured model decisions.
5. Iterate using aggregate sample metrics; review every `notify` manually
   because interrupt false positives are costly.
6. Run full production routing, validate the CSV, package `code/` only, and
   archive `$HOME/hackerrank_orchestrate_august26/log.txt`.

## Submission checklist

- `dataset/output.csv` has the exact six columns, in order, and one row for
  every input message.
- No output row has an invalid enum, empty reason, out-of-range confidence, or
  evidence ID outside message history.
- `code/README.md` documents Python version, dependencies, API environment
  variables, and a single run command.
- Zip only `code/`; exclude `dataset/`, media, caches, virtual environments,
  `node_modules`, and build artifacts.
- Upload `dataset/output.csv` and the append-only transcript log separately.
