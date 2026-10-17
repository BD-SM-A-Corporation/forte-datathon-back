# Forte Datathon — offline video analysis backend

A zero-shot CCTV observation engine using local **Qwen/Qwen2.5-VL-3B-Instruct**
or a vision model served by **Docker Model Runner (DMR)**, OpenCV, FFmpeg, FastAPI,
and Redis/RQ. It describes visible actions, then matches those
observations against a locally maintained catalogue of verified Kazakhstan rules.
It does not identify people or infer invisible transactions, intent, or authorization.

## Запуск проекта с нуля и передача другому разработчику

Основной путь ниже: **Linux + NVIDIA, backend в Docker, frontend на хосте**.
На рабочем NixOS-хосте используется NVIDIA CDI и RTX 4060 Laptop 8 GB с `4bit`.
Python, PyTorch, FFmpeg и компилятор для Triton устанавливает Dockerfile;
для этого пути не нужен Python хоста (в частности, Python 3.14 не поддерживается
`pyproject.toml`). CPU/DMR и разработка без Docker описаны далее отдельно.

### 1. Что передать и установить

Разместите два репозитория рядом; имя родительского каталога произвольное:

```text
project/
  backend/    # этот репозиторий: Dockerfile, app/, rules/, docker-compose*.yml
  front/      # отдельный репозиторий: package.json, bun.lock, src/, public/
```

Передайте актуальные исходники **обоих** репозиториев, включая новые файлы, а не
только последний коммит, если изменения ещё не сохранены в Git. Веса модели,
датасет, `.env`, `.env.local`, `node_modules` и Docker volumes в Git не входят.
Ссылки на репозитории получите у владельца проекта; эта инструкция не требует
локальных путей или учётных данных автора. Нужен отдельный MOV для проверки.

Требования:

- Docker Engine с Compose plugin и доступом текущего пользователя к Docker.
- Рабочий драйвер NVIDIA и GPU-доступ из контейнеров: CDI **либо** NVIDIA runtime.
- Node.js не ниже 20.9 и Bun **1.4.2** (версия из `front/package.json`).
- Интернет для первой сборки, установки frontend и загрузки модели. После
  подготовки обработка видео выполняется локально.
- Место под Docker-образы, временные файлы сборки и примерно 7.5 GB весов;
  оставьте запас в десятки GB. Достаточность VRAM проверяется реальным анализом.

Проверки до запуска:

```bash
docker version
docker compose version
nvidia-smi
node --version
bun --version
```

На NixOS с уже настроенным NVIDIA-драйвером нужен
`hardware.nvidia-container-toolkit.enable = true;` в конфигурации системы.
После применения конфигурации проверьте
`systemctl status nvidia-container-toolkit-cdi-generator.service` и наличие
`/var/run/cdi/nvidia-container-toolkit.json`. Подробности — в разделе
[NVIDIA on NixOS](#nvidia-on-nixos-cdi). Для другого Linux с NVIDIA runtime
используйте альтернативный Compose-файл из следующего шага.

### 2. Настроить backend один раз

В терминале из каталога `project/backend`:

```bash
# Существующий .env не перезаписываем.
test -f .env || cp .env.example .env
openssl rand -hex 32
```

Сохраните сгенерированный секрет в `API_KEY` внутри `.env`. Для NixOS/CDI
установите следующие значения, заменяя существующие строки без дубликатов:

```dotenv
API_KEY=<ваш_новый_секрет>
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu-cdi.yml
INFERENCE_BACKEND=local
MODEL_DEVICE=cuda
MODEL_QUANTIZATION=4bit
MODEL_WEIGHTS_DIR=./weights
MODEL_INIT_OFFLINE=0
```

Для Linux с настроенным **NVIDIA runtime вместо CDI** измените только выбор файлов:

```dotenv
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
```

Выберите один вариант; не объединяйте оба GPU override. Значение
`MODEL_QUANTIZATION=none` из шаблона нужно явно заменить на `4bit` для указанной
8 GB конфигурации. `MODEL_DEVICE=cuda` само по себе GPU в контейнер не подключает.

**Все дальнейшие команды `docker compose` выполняйте из `backend/`.**
Compose читает `COMPOSE_FILE` из `.env`, поэтому обычный `up` сохраняет GPU-настройку.
Явный аргумент `-f` переопределяет этот выбор: не запускайте
`docker compose -f docker-compose.yml up` для GPU-worker. Переменная
`COMPOSE_FILE`, экспортированная в оболочке, также имеет приоритет над `.env`.
Файл `.env` не нужно исполнять через `source`: значения читает Compose.

### 3. Собрать образы, подготовить веса и запустить backend

```bash
docker compose config --quiet
docker compose build model-init api
docker compose run --rm --no-deps model-init
docker compose up -d --no-build redis api worker
docker compose ps -a
curl --fail http://localhost:8000/health
```

API и worker используют один образ `forte-analysis:local`; сборки `api` достаточно
для обоих. `model-init` скачивает закреплённую ревизию
`66285546d2b821cf421d4f5eb2576359d3770cd3` модели Qwen2.5-VL-3B-Instruct в
`weights/Qwen2.5-VL-3B-Instruct`, проверяет файлы и создаёт `.forte-checkpoint.json`.
При `up` сервис инициализации также проверяет веса, после чего завершается с
кодом `0`; это нормальное состояние. Прерванную загрузку можно продолжить
повторением команды `run`. Не меняйте каталог весов во время анализа.

Проверка **внутри рабочего контейнера**, включая реальную CUDA-операцию:

```bash
docker compose exec worker python -c 'import torch; assert torch.cuda.is_available(), "CUDA недоступна внутри worker"; print(torch.cuda.get_device_name(0)); print("CUDA test:", torch.ones(1, device="cuda").sum().item())'
```

Ожидается имя NVIDIA GPU и `CUDA test: 1.0`. Одного `nvidia-smi` на хосте
недостаточно. `/health` проверяет API/Redis/реестр, но не запускает модель.
Модель загружается при первом видео; готовность подтверждается анализом из шага 5.

### 4. Собрать и запустить frontend

В другом терминале из `project/front`:

```bash
test -f .env.local || cp .env.example .env.local
```

Заполните `.env.local`:

```dotenv
BACKEND_API_URL=http://127.0.0.1:8000
BACKEND_API_KEY=<тот_же_секрет_что_API_KEY_в_backend>
```

Ключ используется сервером Next.js; не переименовывайте его в `NEXT_PUBLIC_*`.
Затем:

```bash
bun install --frozen-lockfile
bun run build
bun start
```

Для разработки вместо последних двух команд: `bun dev`. Не запускайте dev и
production одновременно на одном порту и не выполняйте сборку в `.next` работающего
dev-сервера. `bun start` работает, пока запущен этот процесс; для постоянного
развёртывания нужен отдельный менеджер процессов. Docker Compose backend **не
собирает этот frontend**: его необязательный профиль `frontend` требует готового
образа, которого данная инструкция не создаёт.

Адреса: интерфейс — `http://localhost:3000`, Swagger — `http://localhost:8000/docs`,
API — `http://localhost:8000/api/v1`. Если Next.js находится в контейнере,
`127.0.0.1` указывает на сам контейнер: задайте адрес API, доступный из его сети
(например, `http://api:8000` в общей Compose-сети). При доступе с другого компьютера
используйте адрес хоста и проверьте firewall. Для публикации настройте доступ к
UI/прокси, HTTPS и лимит загрузки reverse proxy; локальная команда запуска этого
не делает. Подробности интерфейса: [README frontend](../front/README.md).

### 5. Проверить полный цикл на одном видео

В UI выберите MOV и нажмите «Начать анализ». Для проверки без frontend в Bash:

```bash
read -rsp 'API_KEY: ' FORTE_API_KEY
printf '\n'
curl --fail-with-body -sS http://localhost:8000/api/v1/analyze \
  -H "x-api-key: $FORTE_API_KEY" \
  -F 'file=@/absolute/path/to/video.mov;type=video/quicktime'
```

Подставьте свой путь к видео. Ответ содержит `job_id`; сохраните его и опрашивайте
статус до `done` или `failed`:

```bash
FORTE_JOB_ID='<job_id_из_ответа>'
curl --fail-with-body -sS "http://localhost:8000/api/v1/jobs/$FORTE_JOB_ID" \
  -H "x-api-key: $FORTE_API_KEY"
curl --fail-with-body -sS "http://localhost:8000/api/v1/jobs/$FORTE_JOB_ID/details" \
  -H "x-api-key: $FORTE_API_KEY"
unset FORTE_API_KEY
```

HTTP `200` при загрузке означает принятие задания, а не успешный анализ.
`details: null` во время обработки допустим. Пять минут — бюджет обработки одного
задания; ожидание очереди добавляется отдельно. Один worker обрабатывает видео
последовательно. При истечении ожидания UI проверьте статус через API.

Для диагностики:

```bash
docker compose logs --tail 100 worker
docker compose logs --tail 100 api model-init
# Непрерывный просмотр; выход Ctrl+C не останавливает контейнер:
docker compose logs -f worker
```

Успешный запуск подтверждают `status: done`, заполненные `details`, доступные
наблюдения и проверки условий в UI. `no_violation` означает отсутствие
подтверждённого совпадения с настроенными правилами, а не доказательство соблюдения.
Смотрите `rule_checks`, `verification_summary`, пропущенные проверки и неопределённости.
Самооценка модели не является вероятностью нарушения; итог проверяет человек.

### 6. Перезапуск и обновление без потери GPU

Из `backend/`, с сохранённым `COMPOSE_FILE` в `.env`:

```bash
# Перезапуск тех же контейнеров, без изменения образов и переменных:
docker compose restart api worker

# После изменения исходников / Dockerfile: собрать и пересоздать оба процесса.
docker compose build api
docker compose up -d --no-build api worker

# После изменения .env или GPU-конфигурации, без изменения кода:
docker compose up -d --no-deps --force-recreate api worker

# Остановить, сохранив контейнеры, очередь и результаты:
docker compose stop
# Запустить снова с сохранённым выбором GPU:
docker compose up -d --no-build redis api worker
```

`restart` не подхватывает изменения `.env` и исходников. После обновления кода
нужны сборка и пересоздание; после обновления правил — перезапуск обоих процессов
для сброса кеша. Русский и английский реестры должны быть согласованы по SHA-256.
Worker может завершать текущее задание до 310 секунд перед остановкой.
После перезапуска повторите CUDA-проверку из шага 3. Failed-задания автоматически
не повторяются; видео нужно загрузить заново. Для frontend после изменения кода:
остановите процесс, выполните `bun install --frozen-lockfile`, `bun run build`
и `bun start`; после изменения `.env.local` также перезапустите сервер.

### 7. Частые ошибки

| Симптом | Проверка и действие |
| --- | --- |
| `CUDA requested but no CUDA device is available` | Проверьте `COMPOSE_FILE` в `.env`, CDI/runtime на хосте и CUDA внутри worker. Пересоздайте worker с нужным override; перезапуск контейнера без GPU не добавляет устройство. |
| GPU пропал после `up` | Контейнер мог быть пересоздан только по базовому файлу. Команда ниже показывает использованные Compose-файлы; в списке должен быть GPU override. |
| `Job OK` после `analysis_failed` | RQ успешно завершил функцию, вернувшую объект ошибки. Смотрите `status`/`error` API и `analysis_failed`, а не только `Job OK`. |
| `401` | Ключ запроса или `BACKEND_API_KEY` не совпадает с `API_KEY`. После изменения конфигурации пересоздайте backend и перезапустите frontend. |
| UI сообщает о недоступности backend | Проверьте `/health`, `docker compose ps` и `BACKEND_API_URL` с точки зрения сервера Next.js. |
| `model-init` завершился с ошибкой | Проверьте его логи, доступ к Hugging Face, место на диске и путь к весам. При неполной загрузке повторите `run`; офлайн-режим требует уже подготовленного каталога. |
| `CUDA out of memory` | Проверьте `MODEL_QUANTIZATION=4bit`, другие GPU-процессы и отсутствие второго worker на той же GPU. Настройки числа/размера кадров влияют и на память, и на качество. |
| Ошибка проверки реестра / перевода | Передайте оба JSON из одного состояния исходников; после изменений кода пересоберите API и worker. Не отключайте проверку SHA ради запуска. |
| `Model output failed validation` | GPU уже мог работать, но ответ модели не прошёл схему или проверку времён. Смотрите `invalid_model_output`, лимит токенов и причину завершения в логах. Это не исправляется добавлением GPU override. |
| `404` для прежнего job_id | Результаты хранятся 86400 секунд по умолчанию; загруженный файл удаляется после завершения/ошибки. Загрузите видео снова. |

Команда для проверки реально применённых файлов (без вывода API-ключа):

```bash
docker inspect "$(docker compose ps -q worker)" \
  --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}'
```

### 8. Перенос на другую машину и офлайн-запуск

Сохраните исходники обоих репозиториев и версии их коммитов, а также весь
`weights/`, включая скрытые файлы и `.forte-checkpoint.json` в каталоге модели.
На новой машине создайте свои `.env`/`.env.local`, настройте NVIDIA и повторите
шаги выше. Для заранее подготовленных весов без интернета поставьте
`MODEL_INIT_OFFLINE=1`. Если новая машина совсем без сети, заранее перенесите
также Docker-образы backend, model-init и Redis через `docker save`/`docker load`
и подготовьте frontend с его зависимостями для той же платформы. Одних весов
недостаточно для офлайн-сборки; Dockerfile и `bun install` обычно требуют сети.

История заданий находится в Docker volume Redis, временные загрузки — в другом
volume; исходники их не переносят. Не используйте `docker compose down -v` при
обычном обновлении: эта команда удаляет volumes. Если нужна история, отдельно
подготовьте резервную копию Redis. Исходное видео API не хранит после анализа.

Проверка на рабочем хосте 2026-10-17: второй проход завершил X-1 за 227 секунд и
Х-10 за 219 секунд (без ожидания очереди); по 4 проверки условий. Это проверка
работоспособности цепочки, **не качества детекции**: положительная метка X-1
осталась неподтверждённой. Для приёмки нового развёртывания нужен свой полный
тест из шага 5, а для оценки качества — отдельная размеченная выборка.

---

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

The catalogue contains **11 entries: 4 potential-match checks and 7 review contexts**.
It uses National Bank Resolution No. 231 of 29 November 2019,
[V1900019680](https://adilet.zan.kz/rus/docs/V1900019680), and Resolution No. 14
of 24 February 2020, [V2000020075](https://adilet.zan.kz/rus/docs/V2000020075).
Texts and current-status flags were checked through Adilet's official API on
2026-10-17. Excerpts, retrieval URLs, document IDs, and HTML hashes are recorded
in `rules/sources/`. This verifies source text, not recognition accuracy or a
legal conclusion about a recording. Research scope and exclusions are recorded
in `rules/sources/research_scope.json` against sections 6–7 and FAQ 30–32 of the
provided ForteBank case. The case supplies no exhaustive list of violation types.
An empty catalogue (or one containing only context entries) still fails explicitly.

### What these rules can flag

| Rule | Required visible context | Insufficient on its own |
| --- | --- | --- |
| P20 counting | Receipt of the same cash, initial counting visibly still in progress as the counterparty leaves, and continued counting with both areas visible | Client simply not visible, or a later recount after completed service |
| P20 storage | Earlier receipt, its visible completion and a separate interaction, with the same earlier cash remaining on an open surface | Any cash on a desk during active handling |
| P22 departure | Established counter operator and participant roles, actual operator departure, positive signs of interrupted cash service, participant remaining | Empty chair, waiting person, occlusion, or merely no evidence that service ended |
| P20 clothing storage | Same previously received banknotes tracked after receipt completion into a visible operator clothing pocket | Reaching below the desk, touching clothing, an obscured object or unverified cash origin |
| P32 packaging | Context for reviewing a banknote package handover | No visible counting; this never creates a violation by itself |
| P12 personal items | Context for phone use and item placement in the cash area | Ownership, camera capability, permitted medical items and zone boundaries require verification |
| P19 receipt / P31 issuance | Context for transfer direction, counting, document handover and participant presence | Sparse frames cannot prove a required step was skipped; P32 packaging exceptions apply |
| P24 collection bag | Context for visible bag handover, inspection and seal area | Clothing alone cannot identify a collector; obscured seals and unseen paperwork remain unknown |
| P35 client movement | Context for receiving cash, counting and leaving the counter | Counting may occur in a designated room under another worker's supervision |
| No.14 P8 service entrance | Context for doorway crossings and door movement | A closable entrance is an equipment requirement; an open door alone is not a violation |

The vision model returns positive `facts` with supporting supplied-frame timestamps and material
`uncertainties`. Missing facts mean unknown. Every required fact must be supported
within one analyzed window, with no material uncertainty and confidence at least
0.8. This threshold is an engineering filter, not calibrated legal certainty. Cross-window
consolidation never combines partial fact sets into a newly satisfied rule. Sparse
frames cannot establish that counting never happened, so P32 is context-only.
This conservative approach can miss events whose necessary context spans longer
than a window; it does not invent unseen context to compensate.

The required jury response fields are preserved. `/api/v1/jobs/{id}/details` also returns
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
| `MODEL_RULES_PATH` | English model catalogue, default `rules/regulations.en.json`; must match the Russian catalogue |
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
| `MAX_NEW_TOKENS`, `MAX_FRAMES_PER_WINDOW` | Generation cap 512 (local Qwen retries allow twice this); input cap 12 |
| `GENERATION_RETRIES` | Retry malformed JSON once by default, max 2 |
| `MAX_REFINEMENT_WINDOWS` | Bound additional inspection, default 8 |
| `RULE_VERIFICATION_ENABLED`, `MAX_RULE_CHECKS` | Independent visual fact checks, enabled; at most 4 candidate/rule pairs |
| `VERIFICATION_CONTEXT_SEC` | Include up to 6 seconds before and after the candidate |
| `VERIFICATION_RESERVE_SEC` | Reserve 140 seconds from optional refinement for fact checks |
| `VERIFICATION_CHECK_TIMEOUT_SEC`, `VERIFICATION_MAX_NEW_TOKENS` | Per-check limits: 40 seconds and 768 tokens |
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
intervals in original-video seconds. Refinement windows are spread across the
timeline so the beginning cannot consume every available slot.

After observation, a second pass checks each required fact for selected candidate/rule
pairs against actual frames, including surrounding context. Candidate selection
prioritizes different rules and separated times. The verifier receives visual
questions and the interval, never the first pass's description or confidence as
evidence. Each fact is confirmed, contradicted, or unknown, with an explanation and
grounded timestamps. A separate check establishes whether the facts concern the
same service episode. This is another pass of the same model, not independent ground truth.

With verification enabled, only fully supported checks can produce potential rule
matches. Original observations remain unchanged. `/details` exposes `rule_checks`
and `verification_summary`, including unknown facts, timeouts and skipped candidates.
These states never establish compliance. The legacy `verdict: no_violation` means
no supported match and can include incomplete checks; use the details and mandatory
human-review assessment. The frontend explicitly shows insufficient evidence.

JSON validation requires all event boundaries and evidence times to correspond to
supplied frames (up to 1 ms rounding, normalized back to exact PTS). Event types are
extensible descriptive snake_case labels, not a fixed legal taxonomy. Missing
fields, invalid confidence, truncation, and invented timestamps trigger a bounded
retry; persistent failure fails the job, preventing false all-clear results.

The local Qwen adapter also constrains decoding with `lm-format-enforcer`: its
per-window JSON schema permits only supplied timestamps for event boundaries,
evidence, and facts. The tokenizer vocabulary index is cached with the model.
Generation limits descriptions to 240 characters and individual uncertainty notes
to 120 characters, so verbose text does not consume the budget before the JSON
can close. Stored/API observations retain their older, larger text limits.
Local retries allow twice `MAX_NEW_TOKENS`, still within the same job deadline.
Post-generation validation still checks event ordering, evidence bounds, and all
other observation requirements. Validation failures report a category
(`invalid_json`, `output_token_limit`, `schema_mismatch`, or `ungrounded_timestamp`);
worker logs include the attempt, window bounds, generated token count and stop
reason without recording the generated scene text. Truncated JSON is never repaired
into a successful analysis or treated as an empty observation list.

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

`rules/regulations.json` remains the Russian source for deterministic matching,
API responses, rule names, explanations and official excerpts shown by the frontend.
`rules/regulations.en.json` contains an English working translation of all 11 rules,
including names, source descriptions/excerpts, applicability and explanations.
Its `translation_note` explicitly identifies the excerpts as unofficial translations;
source verification metadata applies to the Russian originals.

The worker and real-model benchmark read `observation_guidance` from the English
catalogue for both local Qwen and DMR. Only these concise action instructions go into
the observation prompt; the full legal texts are not added to every inference window.
These instructions were already English before the split, so the split itself does
not improve detection accuracy or fix empty `facts` responses.

Worker startup checks that both files have identical IDs, rule kinds, event types,
conditions, source URLs, jurisdictions and verification dates. The English file's
`translated_from_sha256` must match the exact bytes of the Russian file. After editing
the Russian catalogue, review/update the translation and then update this hash
(`sha256sum rules/regulations.json`). Updating the hash alone does not verify semantic
equivalence. A missing, outdated or structurally divergent translation stops worker
startup rather than silently using a different set of rules. Custom catalogues need
a corresponding `MODEL_RULES_PATH`; injected test settings may set it to `None`.

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
- `observation_guidance` with concise visible-action instructions. The worker feeds
  these local registry targets to both Qwen and DMR. Legal conclusions still come
  from grounded rule matching and human review, not model-generated legal text.

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

`GET /api/v1/rules` requires the same key and returns the catalogue loaded by the API,
`configured`, and `review_required`. The UI reads it through its server proxy;
loading, actual emptiness, and network failure are distinct states. Restart API and
worker together after editing rules so the displayed and inference catalogues agree.

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
contract, with `type`, `timecode_sec`, `description`, and `evidence_frames`.
Every completed result now includes a preliminary assessment:

```json
{
  "job_id": "<uuid>",
  "status": "done",
  "verdict": "no_violation",
  "violations": [],
  "assessment": {
    "status": "no_supported_match",
    "review_required": true,
    "decision_status": "pending_human_review"
  }
}
```

`no_violation` means no match within the configured rule catalogue and sampled
observations; it does not certify compliance with every Kazakhstan regulation.
`verdict` is deprecated and retained for compatibility. Clients should display
`assessment.status`: `potential_violation` or `no_supported_match`. Both require
human review, including when no events were extracted. This API does not record or
automatically produce a human's final decision.

Each candidate in `violations` includes `confidence` (0–1),
`confidence_kind: "self_reported_observation"`, and `uncertainties`. The score is
the model's self-assessment of the supporting action, **not** a calibrated
probability of a violation or of compliance. The UI displays it as an index out
of 100. Legacy results can lack a score; missing scores must not become zero.
There is no aggregate decision probability until a classifier is calibrated and
evaluated on held-out labelled videos. Even a score of 1 does not bypass review.
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

### NVIDIA on NixOS (CDI)

`MODEL_DEVICE=cuda` selects CUDA in PyTorch; it does not expose the host GPU to
Docker. If the host's `nvidia-smi` works but the worker reports
`CUDA requested but no CUDA device is available`, check GPU access inside the worker.

For NixOS, enable the toolkit in your system configuration and apply it with your
usual `nixos-rebuild switch` command:

```nix
hardware.nvidia-container-toolkit.enable = true;
```

Check `systemctl status nvidia-container-toolkit-cdi-generator.service` and confirm
that `/var/run/cdi/nvidia-container-toolkit.json` exists. See the
[NixOS CUDA container documentation](https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/cuda.section.md#running-docker-or-podman-containers-with-cuda-support).
Use the CDI override instead of the NVIDIA runtime override above; do not combine
the two. On an 8 GB GPU, set `MODEL_QUANTIZATION=4bit` in `.env` (an explicit `none`
overrides the default). From this directory, recreate the existing worker:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu-cdi.yml up -d --no-deps worker
docker compose -f docker-compose.yml -f docker-compose.gpu-cdi.yml exec worker \
  python -c 'import torch; print("CUDA:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

The check should print `CUDA: True` and the GPU name. For an initial deployment,
omit `--no-deps` and add `--build` so Redis, model initialization, and image building
run normally. Continue using the CDI override on subsequent Compose updates.
Upload the video again after fixing access; failed jobs are not retried automatically.

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
