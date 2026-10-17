# Forte Datathon — Backend API & ML Video Analytics

Микросервис FastAPI и ML-пайплайн для видеоаналитики кассовой зоны в соответствии с нормативными актами НБРК.

## Архитектура
- **FastAPI + Uvicorn**: асинхронный сервер для приёма видеофайлов и выдачи результатов
- **Очередь/Воркер**: асинхронный фоновый пайплайн обработки с хранением состояния в SQLite (`storage/db.sqlite3`)
- **ML-пайплайн**:
  - `video.py`: декодирование через ffmpeg (~5 FPS)
  - `detector.py`: YOLO-pose (скелет без распознавания лиц) + детекция объектов кассовой зоны (деньги, ящик, сейф)
  - `tracker.py`: трекинг кассира
  - `features.py`: относительные пространственные признаки (рука–деньги, кассир–кассовый узел)
  - `classifier.py`: state-machine классификатор нарушений по правилам и временным интервалам
  - `evidence.py`: кадры-доказательства с разметкой в base64 и JPEG
  - `rules_engine.py`: нормативная база из `rules/rules.yaml` со ссылками на adilet.zan.kz

## Запуск в среде Nix (`shell.nix`)

В корневой директории проекта активируйте nix-shell:
```bash
nix-shell
```

В директории `backend`:
```bash
# Создание venv и установка зависимостей через uv
uv venv
source .venv/bin/activate
uv pip install -e .

# Запуск сервера
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Запуск в Docker Compose (для сервера / VM)

```bash
docker compose up -d --build
```

Сервер будет доступен на `http://localhost:8000`.
- Интерактивная документация Swagger: `http://localhost:8000/docs`
- Проверка работоспособности: `http://localhost:8000/health`

## Основные API-эндпоинты

- `POST /api/v1/analyze`: Загрузка видеофайла (без ограничений Vercel, напрямую на бэкенд)
- `GET /api/v1/jobs/{job_id}`: Проверка статуса анализа, вердикт (`violation` / `no_violation`), список нарушений и кадры-доказательства
- `GET /api/v1/rules`: Список нормативных правил НБРК
- `GET /api/v1/frames/{job_id}/{violation_id}`: Получение кадра-доказательства
