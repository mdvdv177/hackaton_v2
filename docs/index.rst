Предиктор отклонений транспорта
==========================================================================================

Три независимых модуля: ML-сервис, Backend и веб-дашборд. PostgreSQL хранит
телеметрию, прогнозы и отметки просмотра. Прогноз относится к первой ещё не
посещённой остановке в окне 10–15 минут до её планового прибытия.

Материалы для жюри
------------------------------------------------------------------------------------------

* `Репозиторий и быстрый запуск <https://github.com/mdvdv177/hackaton_v2>`_
* `Инструкция жюри <https://github.com/mdvdv177/hackaton_v2/blob/main/JURY.md>`_
* `Производительность и дополнительные возможности <https://github.com/mdvdv177/hackaton_v2/blob/main/PERFORMANCE.md>`_
* `Submission CSV <https://github.com/mdvdv177/hackaton_v2/blob/main/artifacts/submission.csv>`_
* `Архитектура <https://github.com/mdvdv177/hackaton_v2/blob/main/docs/architecture.md>`_

Документация API
------------------------------------------------------------------------------------------

* `Backend Swagger <api/backend/index.html>`_
* `Backend OpenAPI JSON <api/backend/openapi.json>`_
* `ML Swagger <api/ml/index.html>`_
* `ML OpenAPI JSON <api/ml/openapi.json>`_

Это статическая справка с актуальными схемами обоих FastAPI-приложений.
Для выполнения запросов запустите Docker-стек и откройте Backend Swagger
на http://localhost:8010/docs. ML доступен только внутри Docker-сети.
Кнопки выполнения запросов на публичной странице отключены.

Swagger UI загружает JavaScript и CSS из CDN; JSON-схемы доступны отдельно.
Сборка документации не запускает сервер, NDTP listener или базу данных.

.. toctree::
   :maxdepth: 2
   :caption: Документация по коду

   modules
