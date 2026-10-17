# Forte Datathon — offline video analysis backend

A zero-shot CCTV observation engine using local **Qwen/Qwen2.5-VL-3B-Instruct**
or a vision model served by **Docker Model Runner (DMR)**, OpenCV, FFmpeg, FastAPI,
and Redis/RQ. It describes visible actions, then matches those
observations against a locally maintained catalogue of verified Kazakhstan rules.
It does not identify people or infer invisible transactions, intent, or authorization.

This repository already used a root-level `app/` layout, so that layout is retained
instead of adding a second `backend/` directory. No frontend source exists here.
The obsolete API imported nonexistent modules; it has been replaced, along with
its unused SQLite models. No migration is required for the new Redis job store.

## Architecture

```text
Next.js → POST /api/v1/analyze → shared upload storage + Redis/RQ
                                       ↓
                             persistent RQ worker
                                       ↓
FFprobe PTS → OpenCV decoding → baseline samples + motion candidates
                                       ↓
                     timestamped multi-image windows
                                       ↓
               local Qwen / HTTP → Docker Model Runner
                                       ↓
             optional denser windows → validated observations
                                       ↓
             conservative consolidation → verified local rules
                                       ↓
          original-frame JPEG evidence → Redis result + audit details
                                       ↓
                         GET /api/v1/jobs/{job_id}
```

The default `INFERENCE_BACKEND=local` runs Qwen inside the worker. With
`INFERENCE_BACKEND=dmr`, the worker sends sampled JPEG frames and prompts to the
configured model server over HTTP. Keep DMR on your deployment host/network for
local inference. Dependency, model, and inference-engine downloads are setup
operations; prepare them before offline use. Rule matching uses local files.

The default catalogue now contains conservative checks for paragraphs **20 and 22**
and non-violation context for paragraph **32** of National Bank Resolution No. 231
of 29 November 2019, [official act V1900019680](https://adilet.zan.kz/rus/docs/V1900019680).
The paragraph text was retrieved from Adilet's public API; the retrieved excerpts,
source URL, document ID, and content hash are recorded in
`rules/sources/V1900019680.json`. This verifies the source text, not model accuracy
or a legal conclusion about a recording. The earlier unrelated document URL is not used.
An empty catalogue (or one containing only context entries) still fails explicitly.

### What these rules can flag

| Rule | Required visible context | Insufficient on its own |
| --- | --- | --- |
| P20 counting | Receipt of the same cash, initial counting visibly still in progress as the counterparty leaves, and continued counting with both areas visible | Client simply not visible, or a later recount after completed service |
| P20 storage | Earlier receipt, its visible completion and a separate interaction, with the same earlier cash remaining on an open surface | Any cash on a desk during active handling |
| P22 departure | Established counter operator and participant roles, actual operator departure, positive signs of interrupted cash service, participant remaining | Empty chair, waiting person, occlusion, or merely no evidence that service ended |
| P32 packaging | Context for reviewing a banknote package handover | No visible counting; this never creates a violation by itself |

The vision model returns positive `facts` with supporting supplied-frame timestamps and material
`uncertainties`. Missing facts mean unknown. Every required fact must be supported
within one analyzed window, with no material uncertainty and confidence at least
0.8. This threshold is an engineering filter, not calibrated legal certainty. Cross-window
consolidation never combines partial fact sets into a newly satisfied rule. Sparse
frames cannot establish that counting never happened, so P32 is context-only.
This conservative approach can miss events whose necessary context spans longer
than a window; it does not invent unseen context to compensate.

The jury response remains unchanged. `/api/v1/jobs/{id}/details` additionally returns
`context_matches`, `regulatory_assessment`, and `review_required`. A `no_violation`
response means no supported rule match, even when `review_required` is true; it is
not proof of compliance. Review the details for uncertain or packaging-related cases.

After pulling this update, rebuild both processes so they use the new prompt,
schemas, and catalogue, then upload the video again (old failed jobs are not rerun):

```bash
docker compose up -d --build api worker
# GPU deployment: use the same override as your original startup command.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build api worker
```

Compose mounts `./rules` into both containers, while application code is baked into
the image. Updating the catalogue can therefore expose new fields to an older rule
schema. If startup reports rejected fields such as `kind`, `facts_all`, or
`require_no_uncertainties`, rebuild and recreate **both** services using the command
above; restarting existing containers does not update their code. Catalogue load
errors distinguish unreadable files from invalid content, with the path and details
in the logs. Image builds also validate the bundled catalogue as the runtime user.

## Installation and local development

Requires Python 3.11–3.13, FFmpeg/ffprobe on PATH, and Redis 7.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
cp .env.example .env
# Set API_KEY to a freshly generated secret, e.g. python -c 'import secrets; print(secrets.token_hex(32))'
# Review the bundled rule scope; stage the local checkpoint or configure DMR below.
docker compose up -d redis
uvicorn app.main:app --host 0.0.0.0 --port 8000
# In another terminal, using the same environment and working directory:
pip install -e '.[model]'  # Local backend only; DMR needs just the base installation.
python -m app.workers.analysis
```

The API requires a nonempty `API_KEY` at startup. `/health` checks Redis and reports
whether rules are configured; it does not promise GPU/model readiness. Swagger is
at `/docs`. Queue startup is independent of model initialization and DMR readiness.
One `SimpleWorker` process handles jobs sequentially. For the local backend, it
loads the model lazily and retains it across jobs; a forking worker would reload
weights. DMR manages model loading and residency in its own inference engine.

## Local backend: download the checkpoint in advance

Docker Compose includes a one-off `model-init` service. It downloads the pinned
`Qwen/Qwen2.5-VL-3B-Instruct` revision into `MODEL_WEIGHTS_DIR` (default `./weights`).
The worker waits for successful completion; API and worker retain offline loading.
Weights are approximately 7.5 GB and are stored outside the images. Initial setup
requires internet access and sufficient disk space for weights and Docker images.

To download independently of starting the backend:

```bash
docker compose run --rm --build model-init
# Or start everything; Compose runs model-init before starting the worker:
docker compose up -d --build
docker compose logs -f model-init
```

The initializer validates required files, records SHA-256 hashes in
`.forte-checkpoint.json`, and checks those hashes on subsequent runs without
contacting Hugging Face. This reads all weights and can take time on slow disks.
Interrupted downloads can resume by rerunning the initializer. A completed
checkpoint with corruption or a different revision fails: select a fresh
`MODEL_WEIGHTS_DIR` and recreate the services to avoid changing in-use weights.
`QWEN_MODEL_REVISION` must be a full commit SHA; the default is
`66285546d2b821cf421d4f5eb2576359d3770cd3`. Hashes detect changes after preparation;
they do not replace a real model inference smoke test.

For an air-gapped deployment, prepare the directory on a connected machine, copy
the entire weights directory **including hidden files and the completion marker**,
and set `MODEL_INIT_OFFLINE=1`. Also stage Docker images beforehand. The setup
container writes the mounted directory as root; inference mounts it read-only.

For non-Docker development, manual download is also supported:

```bash
pip install 'huggingface_hub[cli]'
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download \
  Qwen/Qwen2.5-VL-3B-Instruct --local-dir weights/Qwen2.5-VL-3B-Instruct
```

Copy the entire directory, including processor/tokenizer files and all safetensor
shards, to the offline machine. Set `QWEN_MODEL_PATH` to that local directory.
For reproducible evaluation pass `--revision 66285546d2b821cf421d4f5eb2576359d3770cd3`
and record that SHA with the benchmark. Compose still requires running model-init
to prepare its completion marker. Model loading uses `local_files_only=True`, disables
remote code, and enables `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`. A missing or
incompatible checkpoint fails clearly; there is no alternate-model fallback.

The adapter follows the official [Qwen multi-image interface](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct#quickstart):
interleaved timestamp text and images, Qwen chat template, compatible processor,
and generated-token-only decoding. Image sequences are intentional: each decoded
presentation timestamp remains explicit even for variable-frame-rate recordings.
The native video-tensor path is not enabled because a single FPS would not preserve
arbitrary VFR spacing. Real-checkpoint compatibility must be confirmed with the
smoke test below before deployment; no model execution is implied by mocked tests.

## Docker Model Runner backend

Enable or install DMR following the [Docker setup guide](https://docs.docker.com/ai/model-runner/get-started/).
Choose a **vision model that supports multiple images per request**, prepare it in
DMR, and use its full reference (namespace/name and preferably a pinned tag).
No default DMR model is assumed. The existing Qwen Hugging Face directory is not
automatically served by DMR: choose a supported engine and package/convert it as
needed. llama.cpp uses GGUF with a matching multimodal projector where required;
vLLM uses Safetensors. See [engines](https://docs.docker.com/ai/model-runner/inference-engines/)
and [model packaging](https://docs.docker.com/reference/cli/docker/model/package/).

After selecting an actual vision model reference, set it in your shell:

```bash
export MODEL_NAME='your-namespace/your-vision-model:tag'  # Replace with your real model.
docker model pull "$MODEL_NAME"
docker model list
```

Prepare the runner engine and model before going offline. Confirm image support
with the opt-in smoke test below; listing a model alone does not prove compatibility.

For a worker running directly on the host, set these in `.env`:

```dotenv
INFERENCE_BACKEND=dmr
MODEL_BASE_URL=http://localhost:12434/engines/v1
MODEL_NAME=your-namespace/your-vision-model:tag
MODEL_HTTP_TIMEOUT_SEC=120
```

`MODEL_BASE_URL` is the API base; the adapter appends `/chat/completions`. Docker
Desktop requires host TCP access enabled for host processes. DMR requires no API
key; the application's `API_KEY` still protects uploads and results. Endpoint details
are in the [DMR API reference](https://docs.docker.com/ai/model-runner/api-reference/).

For Docker Engine containers, use the DMR override (Compose **2.24.4+**):

```bash
# In .env, set MODEL_NAME to the prepared vision model and API_KEY to your API secret.
# Use this container URL instead of localhost, or leave MODEL_BASE_URL unset:
export MODEL_BASE_URL=http://model-runner.docker.internal:12434/engines/v1
docker compose -f docker-compose.yml -f docker-compose.dmr.yml up -d --build
```

The override selects `dmr`, builds `forte-analysis:dmr` with `INSTALL_MODEL=false`,
and removes the weights mount. It maps `model-runner.docker.internal` to the host
gateway. DMR must be running and reachable there. For Docker Desktop, set
`MODEL_BASE_URL=http://host.docker.internal:12434/engines/v1` and enable host TCP
access; this uses Desktop's host address independently of the Engine gateway mapping.
Do not combine the DMR override with `docker-compose.gpu.yml`: GPU access and
quantization belong to the runner, not the API or worker. Size worker concurrency
to the model server's capacity. The override connects to an existing runner;
it does not install DMR or pull models during application startup.

The adapter sends the observation prompt followed by alternating timestamp text
and JPEG data URLs. It requests JSON with `temperature=0` and a bounded output
length. Both backends validate the same observation schema and supplied timestamps.
Malformed observation JSON gets `GENERATION_RETRIES` attempts after the first call;
HTTP failures, malformed response envelopes, and incomplete completions fail the
job without generating a verdict or switching backends.

The total HTTP request timeout is the smaller of `MODEL_HTTP_TIMEOUT_SEC` and the
remaining job budget, including connection and response transfer. Retries share
that job budget. Client cancellation does not guarantee server-side generation
has stopped. Model loading and server queue time also consume the request budget.

Check the prepared model from the host with base/test dependencies installed:

```bash
MODEL_BASE_URL=http://localhost:12434/engines/v1 \
  RUN_DMR_MODEL_TEST=1 pytest -q tests/test_model_integration.py -k dmr
```

This sends two blank timestamped images through the actual adapter and expects no
events. It checks the multi-image request and response contract, not visual accuracy.
For troubleshooting, use `docker model logs` and check the model name, image support,
context capacity (prompt plus all images and output), and endpoint reachability.
Raise token/context limits deliberately if the server reports truncation or overflow.

## Configuration

All settings use environment variables or `.env`; `.env.example` lists defaults.

| Variables | Meaning |
| --- | --- |
| `API_KEY`, `CORS_ORIGINS` | Required secret; JSON list of allowed browser origins |
| `REDIS_URL`, `QUEUE_NAME` | Queue and result storage |
| `UPLOAD_DIR`, `RULES_PATH` | Shared recordings and validated JSON catalogue |
| `MAX_UPLOAD_MB` | File limit, default 512 MiB; multipart body has 64 KiB overhead allowance |
| `MAX_DURATION_SEC` | Maximum accepted video duration, default 150 seconds |
| `MAX_VIDEO_PIXELS`, `MAX_DECODED_FRAMES` | Decode resource bounds, defaults 8,294,400 and 18,000 |
| `BASE_SAMPLE_FPS`, `DETAILED_SAMPLE_FPS` | Whole-timeline baseline 1 FPS and refinement 3 FPS |
| `WINDOW_DURATION_SEC`, `WINDOW_OVERLAP_SEC` | Baseline windows, defaults 5 and 1 seconds |
| `MOTION_THRESHOLD` | Minimum changed-pixel fraction, default 0.04 |
| `MOTION_PIXEL_THRESHOLD` | Minimum grayscale pixel difference, default 25/255 |
| `MOTION_MERGE_GAP_SEC` | Merge nearby motion intervals, default 1 second |
| `INFERENCE_MAX_SIDE`, `ROI` | Resize longest side to 448; optional normalized crop `[left,top,right,bottom]` |
| `INFERENCE_BACKEND` | `local` (default) or `dmr` |
| `MODEL_BASE_URL`, `MODEL_NAME` | Required for DMR: HTTP(S) API base and full vision-model reference |
| `MODEL_HTTP_TIMEOUT_SEC` | DMR total request timeout, default 120 seconds; capped by remaining job budget |
| `QWEN_MODEL_PATH`, `MODEL_DEVICE` | Local backend only: existing checkpoint; `cpu` or `cuda` |
| `MODEL_QUANTIZATION` | Local backend only: `none`, `4bit`, or `8bit`; quantization requires CUDA |
| `MAX_NEW_TOKENS`, `MAX_FRAMES_PER_WINDOW` | Generation cap 512; input cap 12 |
| `GENERATION_RETRIES` | Retry malformed JSON once by default, max 2 |
| `MAX_REFINEMENT_WINDOWS` | Bound additional inspection, default 8 |
| `PROCESSING_BUDGET_SEC` | Default 300-second worker timeout; bounds stages, local generation, and DMR requests/retries |
| `QUEUE_TTL_SEC`, `RESULT_TTL_SEC` | Queue expiry 3600 seconds; result/failure retention 86400 seconds |
| `EVENT_MERGE_GAP_SEC` | Duplicate-event gap tolerance, default 0.2 seconds |

Settings reject overlap larger than the window, invalid ROIs, and baseline rates
that cannot fit the frame cap. Refinement windows shorten automatically to fit the
configured detailed sampling rate without dropping baseline observations.

## Video timing, motion, and evidence

FFprobe decodes presentation timestamps for the first video stream. Times are
normalized to its first decoded frame, not inferred by dividing a frame index by
FPS. OpenCV decodes that stream in presentation order; frame counts are checked.
Rotation metadata is intentionally not applied, so all ROI and evidence coordinates
refer to original decoded pixel orientation. Audio is neither decoded nor analyzed.

Sampling chooses the first decoded frame at or after each sampling target and
reports its actual timestamp. Baseline samples cover the complete timeline at the
configured rate; sparse sampling can still miss short actions. Sample thumbnails
are cached temporarily on disk and only one inference window is loaded into RAM.
Full-resolution frames are decoded one at a time. FFprobe frame metadata is small
but retained in memory. Corrupt or unsupported recordings fail instead of producing
a verdict from incomplete decoding.

Motion is measured on blurred grayscale frames resized to 160 pixels wide. The
score is the fraction of pixels whose difference exceeds the grayscale threshold,
after 3×3 morphological opening. Adjacent intervals merge and retain the peak score.
This is **not a probability of a violation**. Motion and initial event candidates
request extra windows; `IntervalProvider` allows future local detectors to propose
intervals in original-video seconds. Suspected events receive refinement priority.

JSON validation requires all event boundaries and evidence times to correspond to
supplied frames (up to 1 ms rounding, normalized back to exact PTS). Event types are
extensible descriptive snake_case labels, not a fixed legal taxonomy. Missing
fields, invalid confidence, truncation, and invented timestamps trigger a bounded
retry; persistent failure fails the job, preventing false all-clear results.

Consolidation requires matching action type and object set, overlapping/nearby
times, and description similarity ≥0.8. It keeps an existing description verbatim.
This transparent heuristic can still split paraphrases or merge similar repeated
actions; review it against annotated videos. Distinct times/descriptions remain
separate. Each event-rule match produces one potential violation. Multiple rules
may apply to one event and remain separate, with separate rule IDs in audit details.

Evidence is extracted once after matching, in a single decode pass across all
selected timestamps. It contains 1–3 original full-resolution JPEG frames per
matched event, with no ROI, overlay, generated content, or identity recognition.

## Adding verified Kazakhstan regulations

`rules/regulations.schema.json` defines the strict catalogue schema. The application
validates it at API and worker startup. Missing/malformed files prevent startup;
an empty valid catalogue allows startup but fails analysis verdicts explicitly.

Before adding a production rule, a qualified reviewer must verify the currently
applicable Kazakhstan act, exact section, quotation, applicability to this banking
context, and whether the condition is actually observable in video. Do not turn
object transfers, phones, or bags into violations by assumption.

Each entry requires:

- Stable `id`, `name`, `observable_event_types`, and `conditions`.
- `source.document`, `section`, official HTTPS `url`, `jurisdiction: "KZ"`,
  `verified_on`, `verified_by`, and a checked `excerpt`.
- `applicability` documenting why a purely visual match is sufficient to flag a
  **potential** violation in this deployment, and `explanation_template` as plain
  explanatory text (no executable template interpolation).

Supported conditions are `objects_all`, `min_duration_sec`, `min_confidence`,
`facts_all`, and `require_no_uncertainties`. `kind: context` entries never generate
violations. The default `kind` is `potential_violation`.
Unknown fields are rejected rather than silently ignored. Confidence is the model's
uncalibrated self-assessment. Deterministic matching cannot establish legal guilt.
The official-domain allowlist and verification fields validate catalogue structure;
they do **not** independently verify law or the truth of a reviewer's assertions.
Bundled rules cite the checked official source; their visual matching logic is an
engineering interpretation requiring human review. Unit-test rules are explicitly
fictional fixtures and must never be deployed.

Restart API and worker after editing the catalogue. Results retain a snapshot of
matched rules and source references in `/jobs/{id}/details`. New descriptive event
types and rule entries do not require model retraining. Rules requiring facts beyond
the supported visible conditions must wait for an appropriate implementation.

## API

Every analysis, status, and detail request requires `X-API-Key`. Never expose the
shared secret in a public Next.js client bundle; proxy requests from a server route.

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "X-API-Key: $API_KEY" -F 'file=@recording.mov'
# {"job_id":"<uuid>"}
curl http://localhost:8000/api/v1/jobs/<uuid> -H "X-API-Key: $API_KEY"
curl http://localhost:8000/api/v1/jobs/<uuid>/details -H "X-API-Key: $API_KEY"
```

Upload streaming is size-limited; extension validation happens immediately and
actual video validation occurs in the worker. Therefore corrupted `.mov` content
receives a job ID and later a `failed` status. Processing never runs in the request.
`processing` covers queued and running jobs. Completed responses preserve the jury
contract, with `type`, `timecode_sec`, `description`, and `evidence_frames`:

```json
{"job_id":"<uuid>","status":"done","verdict":"no_violation","violations":[]}
```

`no_violation` means no match within the configured rule catalogue and sampled
observations; it does not certify compliance with every Kazakhstan regulation.
Failures include a safe `error` and omit verdict/violations. Unknown or expired jobs
return 404; wrong/missing keys return 401. Redis outages return 503. Audit details
contain observations, matched source snapshots, motion intervals, and timing metrics.

Uploads are deleted after completed or failed jobs. Hard-killed workers or expired
queued jobs may leave files; periodically run `python -m scripts.cleanup` against
shared storage. It deletes old files only when no active job owns them. Redis has
bounded result retention. Keep Redis private: RQ is a trusted internal queue.

## Docker Compose and Coolify

```bash
cp .env.example .env
# Set API_KEY and review verified regulations; model-init stages weights automatically.
docker compose up -d --build
# With NVIDIA Container Toolkit and a suitable host NVIDIA driver:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
# Optional: use your existing frontend image, no frontend is rebuilt here.
FRONTEND_IMAGE=your-nextjs-image:tag docker compose --profile frontend up -d
```

The GPU override reserves one NVIDIA GPU for the worker. Set
`MODEL_QUANTIZATION=4bit` in `.env` and install the quantization extra for 4-bit mode.
API and worker share one image and upload volume; models and rules are read-only
bind mounts. Redis persists to its own volume and binds its host port only to loopback. Long-running services
restart automatically; model-init exits after setup and has no restart policy. Worker health allows an entire processing budget between
heartbeats because SimpleWorker does not heartbeat during a job.

For CPU-only infrastructure development or DMR, build with `INSTALL_MODEL=false`.
This supports API/queue/tests and DMR inference; local Qwen requires the model extra.
There is no runtime mock switch on the production API. Tests explicitly inject a
mock observer. On Coolify, use the same Compose services, shared volume, mounted
model/rule directories, environment values, and GPU reservation for local Qwen.
For DMR use the DMR override and a reachable prepared runner. Route the API
through your deployment proxy, which should also enforce the upload body limit.
For local Qwen, do not scale multiple workers onto the same GPU.

## GPU requirements and limitations

With the local backend, the 3B model can run with CUDA FP16/BF16 or bitsandbytes 4/8-bit quantization.
Choose an NVIDIA GPU with enough VRAM for weights **and** multi-image activations;
8–12 GB is a starting planning range for quantized, reduced-resolution experiments,
not a measured guarantee. CPU float32 is available for development but is likely
impractical for the five-minute target. Input frame count/resolution strongly affect
memory and latency. Quantization support depends on the installed CUDA/PyTorch/
bitsandbytes combination and must be tested on the deployment host.

This MVP uses a cooperative per-stage/token deadline plus RQ's Unix signal timeout.
DMR adds a total HTTP request timeout; server resource limits are managed by DMR.
A blocking native GPU operation can delay signal handling; the timeout is not a
hard real-time guarantee. Cold model loading counts against the job budget. For
long-running deployments, benchmark both cold and warm runs.

## Tests and performance measurement

```bash
pip install -e '.[test]'
pytest -q
ruff check app tests scripts
# Optional real local-checkpoint smoke test (no downloads):
RUN_LOCAL_MODEL_TEST=1 pytest -q tests/test_model_integration.py
# Optional real DMR multi-image smoke test (prepare model and set MODEL_NAME first):
MODEL_BASE_URL=http://localhost:12434/engines/v1 RUN_DMR_MODEL_TEST=1 pytest -q tests/test_model_integration.py -k dmr
# Real model benchmark after configuring verified rules and staging weights:
python -m scripts.benchmark recording.mov --repeat 3 > benchmark.json
# Mock timing, requires an explicitly supplied test catalogue; no model accuracy claim:
RULES_PATH=/path/to/your/test-catalogue.json python -m scripts.benchmark recording.mov --mock
```

Tests generate real MOVs locally, including VFR input, and validate OpenCV/FFmpeg
processing. Inference tests mock generated text; end-to-end tests use a deterministic
observer and fakeredis with a real RQ SimpleWorker. They require no model weights,
CUDA, or internet. DMR adapter tests use a mock HTTP transport and cover image order,
JSON validation/retries, failure handling, and deadlines. The local-model and DMR
smoke tests exercise real inference and skip by default.

The benchmark uses the selected backend and reports its model identifier. For DMR,
processing time includes model-server requests; `device` is null because the runner
owns device selection. Benchmark all supplied 10–145 second recordings; retain checkpoint SHA, GPU/driver,
Python/package versions, quantization, frame settings, cold/warm timing, errors,
and peak VRAM (`nvidia-smi`). The tool reports per-run wall time and the 300-second
comparison, excluding upload/network/queue wait. Measure submission-to-result time
separately with the API under expected load. A serial worker can accumulate queue
wait even when individual jobs meet the budget.

**No real-model inference, violation-detection accuracy, or five-minute GPU result
has been validated in this repository's CPU tests.** No dataset recordings or model weights were supplied during initial implementation.
Regulatory source text was subsequently checked for the bundled rules; that does
not validate the visual model or determine legal liability. Motion, zero-shot descriptions, and
self-reported confidence can all produce false positives or omissions. Occlusion,
small objects, frame sampling, lighting, and ROI crops limit observations. Human
review and labeled evaluation are required before relying on alerts.

## Validation recorded during implementation

Initial engine validation on 2026-10-17, Python 3.11.15: **61 tests passed, 1 skipped**
in 4.29 seconds (before the subsequent cash-regulation tests).
The skipped test requires a real local Qwen checkpoint. The suite emitted one
upstream Starlette warning about its legacy httpx TestClient transport. Ruff lint,
format checks, `git diff --check`, and CPU/GPU Compose configuration validation
passed. Docker images/services were not built or started for this validation.
API tests needed execution outside the restricted sandbox because even an empty
FastAPI TestClient stalled on local event-loop communication inside it.

Two synthetic, static, 160×120 MOVs at 10 decoded FPS were also measured with
`MockObserver` and a fictional test-only catalogue: a 10-second clip took 0.335 s
(10 baseline samples, 3 windows), and a 145-second clip took 0.547 s (145 samples,
36 windows). These measurements include preprocessing and pipeline orchestration,
but no neural inference, matched-event evidence, uploads, or queue wait. They are
not evidence that the real deployment meets the five-minute jury limit.

Cash-rule update validation: **77 tests passed, 1 real-model test skipped** in
3.05 seconds on the development environment (Python 3.12). The 16 new tests cover
the bundled catalogue, mandatory facts, uncertainties, timestamp grounding,
non-violating P32 context, and prevention of fabricated cross-window conditions.
After tightening P20 to exclude later recounting, all 16 cash-rule tests passed
again. Real Qwen inference and detection accuracy have not been tested by these tests.
