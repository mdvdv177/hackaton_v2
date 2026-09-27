# Разработка и воспроизведение

Команды выполняются из корня репозитория. Нужны Python 3.12+ (проверено на 3.13)
и Node.js 22+. Для браузерных проверок используется Google Chrome; альтернативно
установите Chromium командой `.venv/bin/python -m playwright install chromium`
и задайте `PLAYWRIGHT_CHANNEL=chromium`.

```bash
make install
make test
make frontend
make docs
.venv/bin/python scripts/dev.py
```

Локальный frontend — http://127.0.0.1:5173, Backend использует SQLite.
При одновременной работе с Docker задавайте свободные порты. `make docs`
строит Sphinx и обе OpenAPI-схемы без запуска Backend, NDTP или БД.
Для просмотра результата: `.venv/bin/python -m http.server 8099 --directory docs/_build/html`.

## Обучение и submission

```bash
make train
make train-stream
make submit
```

Команды обучения перезаписывают соответствующие модели и отчёты в `artifacts/`.
Official использует предоставленный `cur_dev_s`; stream получает отклонение
от общего причинного наблюдателя. Настройка CatBoost выполняется отдельно,
без замены действующих моделей: [протокол и результат](tuning.md).

## Приёмочные проверки

```bash
make acceptance-ui
make acceptance-cold
make acceptance-faults
make acceptance-warm
.venv/bin/python scripts/check_demo_ndtp.py
```

Эти команды создают собственные Docker-проекты и очищают их после проверки.
`acceptance-warm` занимает около пяти минут и не заменяет часовой протокол.
Поставленный эмулятор требует отдельного Docker-архива организаторов;
его запуск описан в [инструкции жюри](../JURY.md).

Часовая нагрузка отложена по указанию пользователя. `make acceptance-load`
остаётся отдельной ручной командой и не вызывается тестами, сборкой документации
или GitHub Actions. Старые `smoke.py`, `benchmark.py`, `check_resilience.py`
могут переключать основной прогон; они предназначены для ручной диагностики.

## Состав репозитория

В Git включены исходники, тесты, Docker-конфигурация, данные для воспроизведения,
готовые модели, submission и проверяемые итоговые отчёты. Локальные инструкции
для агента, исходные задания/PDF, `.env`, окружения, кэши, сборки, архив Docker-образа
и промежуточные дубли экспериментов исключены через `.gitignore`.
Локальные копии исключённых материалов сохраняются; история Git не переписывается.
