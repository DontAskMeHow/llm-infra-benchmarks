# Пулы данных для качественных замеров

Каталог: `artifacts/gb10/quality/datasets/` (в git — только компактные JSONL-срезки;
сырьё/полные датасеты держим локально, в git не кладём).

Формат QA-пула (jsonl): `{"id": str, "prompt": str, "answer": str, "type": "mcq|free|code"}`.

| Файл | Источник | Состав | Лицензия/условия |
| --- | --- | --- | --- |
| gpqa_diamond_30.jsonl | зеркало hendrydong/gpqa_diamond_mc (offset 60; оригинал Idavidrein/gpqa на HF под гейтом) | 30 вопросов MC `\boxed{A-D}` (физика 11/химия 15/биология 4) | открытое зеркало |
| mmlu_pro_25.jsonl | TIGER-Lab/MMLU-Pro (test, offset 200; демо-эталоны offset 100) | 25 вопросов, 10 вариантов, 5-shot демо вшиты в промпт | открытый |
| math_20.jsonl (фаза 2) | lighteval/MATH-Hard / MATH-500 | открытый ответ, `\boxed{}` | открытый |
| ce_doc.txt | самодельный детерминированный текст (филлер-слова) или wikitext/C4-срезка | ≥512K слов | открытый |

Правило срезок: фиксировать смещение/seed выборки в имени файла или первом поле
`"meta"`, чтобы прогоны были воспроизводимы. Пересобирать пул — только новым файлом,
старые результаты не перетирать.

Скачивание: datasets-server API без токена:
`https://datasets-server.huggingface.co/rows?dataset=Idavidrein/gpqa&config=gpqa_diamond&split=train&offset=0&length=20`
(поля `question`, `correct_answer`, варианты — в `query_row`). Формат подсказки для MCQ:
вопрос + варианты буквами, просьба «Ответь одной буквой варианта».
