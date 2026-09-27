Предиктор отклонений транспорта
============================================================

Система состоит из независимых Backend, ML-сервиса и веб-дашборда.
Инструкции запуска находятся в README.md корня проекта.
Описание архитектуры — в docs/architecture.md, сценарии проверки —
в docs/demo.md. Спецификация REST API генерируется FastAPI на /docs
и /openapi.json каждого Python-сервиса.

Общие признаки и входные данные
------------------------------------------------------------

.. automodule:: predictor.features
   :members:

.. automodule:: predictor.data
   :members:

.. automodule:: predictor.observer
   :members:

.. automodule:: predictor.stream_features
   :members:

Протокол NDTP
-------------

.. automodule:: backend.ndtp
   :members:

Безопасный импорт Backend
------------------------------------------------------------

.. automodule:: backend.data
   :members:

.. automodule:: backend.scenarios
   :members:

.. automodule:: backend.storage
   :members: Store
