# Предиктор отклонений городского транспорта

Docker-решение из трёх модулей: **ML-ядро** (CatBoost/FastAPI), **Backend**
(NDTP, REST/SSE, PostgreSQL) и **BI-дашборд** (React/TypeScript/Leaflet).
Прогнозируется подписанное отклонение для первой ещё не посещённой остановки
в окне **10–15 минут до планового прибытия**. Плюс — опоздание, минус — опережение.

## Быстрый запуск

Нужны Git и работающий Docker Engine с Compose. Данные для демонстрации
и обученные модели включены в репозиторий; переобучение не требуется.

```bash
git clone https://github.com/mdvdv177/hackaton_v2.git
cd hackaton_v2
docker compose up --build -d --wait
```

Откройте **http://localhost:8088**, выберите «Диспетчерский replay», скорость ×1
и нажмите «Запустить replay». Появятся позиции ТС, прогнозы и журнал инцидентов.
Если доступна кнопка «Продолжить», сохранённый прогон находится на паузе.

| Интерфейс | Адрес |
|---|---|
| Дашборд | http://localhost:8088 |
| Backend Swagger | http://localhost:8010/docs |
| Backend OpenAPI | http://localhost:8010/openapi.json |
| Состояние, очереди, задержки | http://localhost:8010/api/v1/system/status |
| Prometheus-метрики | http://localhost:8010/metrics |
| Приём NDTP TCP | `localhost:9211` |

Порты и пароль PostgreSQL переопределяются через `.env` по примеру `.env.example`.
ML и PostgreSQL доступны только внутри Docker-сети. Сборке нужен интернет для
зависимостей; карта работает со схемой маршрутов и при недоступных фоновых тайлах.
Остановка с сохранением БД: `docker compose down`. Команда с `--volumes` удаляет БД.
На macOS при отсутствии Compose в PATH доступен `python3 -m scripts.stack up`.

## Материалы для сдачи

- **[Submission CSV](artifacts/submission.csv)** — 151 validate-ID, UTF-8,
  разделитель `;`, колонки `sample_id;prediction`, все значения конечны.
- **[Инструкция жюри](JURY.md)** — исторический поток, настоящий NDTP,
  прогнозы, алерты и метрики.
- **[Производительность и дополнительные возможности](PERFORMANCE.md)**.
- **[Sphinx и Swagger на GitHub Pages](https://mdvdv177.github.io/hackaton_v2/)**.
- [Архитектура](docs/architecture.md), [импорт сценариев](docs/scenarios.md),
  [подробная демонстрация](docs/demo.md), [проверки](docs/validation.md).

## Качество и ограничения

| Профиль | Проверочные точки | MAE, с |
|---|---:|---:|
| Official, предоставленный `cur_dev_s` | 353 | 70,61 |
| Stream, причинные признаки, все контрольные точки | 353 | 80,58 |
| Stream, свежие и ещё не посещённые цели | 340 | 81,58 |

Test и validate используют телеметрию одного периода; скрытый score и качество
новых дней не измерены. [Отчёт official](artifacts/report.json),
[отчёт stream](artifacts/stream_report.json). Настройка 50 конфигураций CatBoost
не подтвердила требуемый выигрыш; текущая модель сохранена. [Результат](docs/tuning.md).

Часовая нагрузка отложена по указанию пользователя. Реальная геометрия перевозчика
не предоставлена, человеческий тест понятности не проведён. Линии без импорта
геометрии обозначены как схема, объяснения модели — как гипотезы.

## Разработка

Python 3.12+ (проверено на 3.13), Node.js 22+; команды запускаются из корня.

```bash
make install
make test
make frontend
make docs
make submit
```

`make docs` создаёт Sphinx и статические Swagger/OpenAPI в `docs/_build/html/`.
Публикация Pages выполняется workflow при push в `main`.
[Обучение, проверки и локальная разработка](docs/development.md).

Код: `predictor/` — признаки; `ml/` — обучение и инференс; `backend/` — поток и API;
`frontend/` — дашборд; `tests/` и `scripts/` — проверки и инструменты.
