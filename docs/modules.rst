Модули решения
==========================================================================================

Причинные признаки и входные данные
------------------------------------------------------------------------------------------

Обучение и Backend используют общие преобразования. Время отсечения ограничивает
доступную телеметрию; фактические будущие прибытия не входят в признаки.

.. automodule:: predictor.features
   :members:

.. automodule:: predictor.data
   :members:

.. automodule:: predictor.observer
   :members:

.. automodule:: predictor.stream_features
   :members:

ML: обучение и инференс
------------------------------------------------------------------------------------------

Official использует предоставленное текущее отклонение, stream — оценку общего
наблюдателя. Сервис загружает готовые CatBoost-модели и возвращает подписанные
секунды, вероятность опоздания и факторы прогноза. Настройка гиперпараметров
создаёт отдельные результаты и не заменяет действующие модели автоматически.

.. automodule:: ml.model
   :members:

.. automodule:: ml.train
   :members: train, temporal_folds

.. automodule:: ml.train_stream
   :members: train

.. automodule:: ml.submit
   :members:

.. automodule:: ml.tune
   :members: run_tuning, make_inner_split, choose_finalist

Backend: NDTP, сценарии и хранение
------------------------------------------------------------------------------------------

NDTP-парсер принимает бинарную телеметрию. Dispatcher связывает её со сценарием,
вызывает ML и публикует снимки через REST/SSE. Store сохраняет события,
checkpoint, прогнозы и отметки просмотра; перед публикацией проверяется горизонт.

.. automodule:: backend.ndtp
   :members:

.. automodule:: backend.data
   :members:

.. automodule:: backend.scenarios
   :members:

.. automodule:: backend.storage
   :members: Store

.. automodule:: backend.service
   :members: Dispatcher

.. automodule:: backend.app
   :members: create_app, ReplayStart, LiveStart
