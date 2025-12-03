# Компрессоры документов LangChain

## Обзор

Компрессоры документов из LangChain позволяют улучшить качество RAG системы, фильтруя и сжимая найденные документы перед передачей в LLM.

## Преимущества использования компрессоров

1. **Улучшение качества ответов**: Фильтрация нерелевантных документов уменьшает шум в контексте
2. **Экономия токенов**: Меньше документов = меньше токенов для LLM
3. **Быстрее генерация**: Меньший контекст обрабатывается быстрее
4. **Более точные ответы**: LLM получает только релевантную информацию

## Использование

### Базовое использование

```python
from rag_system import SysAdminRAG

# Инициализация с компрессией
rag = SysAdminRAG(
    use_document_compression=True,
    compression_similarity_threshold=0.76
)

rag.load_index()
result = rag.ask("Как настроить DNS?", k=10)  # k=10, но после компрессии останется меньше
```

### Через командную строку

```bash
# Включить компрессию
python main.py --mode chat --use-compression

# С настройкой порога
python main.py --mode chat --use-compression --compression-threshold 0.8
```

## Как это работает

### Процесс компрессии

1. **Поиск**: ColBERT находит k релевантных документов
2. **Компрессия**: `EmbeddingsFilter` фильтрует документы по семантической схожести
3. **Генерация**: LLM получает только отфильтрованные документы

### EmbeddingsFilter

`EmbeddingsFilter` использует эмбеддинги для вычисления семантической схожести между запросом и каждым документом:

- Используется модель `paraphrase-multilingual-MiniLM-L12-v2` (поддерживает русский язык)
- Документы с схожестью ниже порога отфильтровываются
- Остаются только наиболее релевантные документы

### DocumentCompressorPipeline

`DocumentCompressorPipeline` позволяет комбинировать несколько компрессоров:

```python
from document_compressor import DocumentCompressor

compressor = DocumentCompressor(
    use_compression=True,
    similarity_threshold=0.76,
    embeddings_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

## Настройка параметров

### similarity_threshold

Порог схожести определяет, какие документы остаются:

- **0.5-0.7**: Мягкая фильтрация (больше документов, больше контекста)
- **0.7-0.8**: Умеренная фильтрация (рекомендуется)
- **0.8-0.9**: Строгая фильтрация (только очень релевантные документы)
- **0.9+**: Очень строгая фильтрация (может отфильтровать слишком много)

**Рекомендации:**
- Для общих вопросов: `0.7-0.75`
- Для специфических вопросов: `0.75-0.8`
- Для очень точных ответов: `0.8-0.85`

### Выбор модели эмбеддингов

По умолчанию используется `paraphrase-multilingual-MiniLM-L12-v2`, которая:
- Поддерживает русский язык
- Быстрая и эффективная
- Хорошо работает с технической документацией

Альтернативные модели:
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (лучше качество, медленнее)
- `intfloat/multilingual-e5-base` (отличное качество для многоязычных задач)

## Примеры использования

### Пример 1: Без компрессии

```python
rag = SysAdminRAG(use_document_compression=False)
rag.load_index()
result = rag.ask("Как настроить Active Directory?", k=5)
# Используются все 5 найденных документов
```

### Пример 2: С компрессией

```python
rag = SysAdminRAG(
    use_document_compression=True,
    compression_similarity_threshold=0.76
)
rag.load_index()
result = rag.ask("Как настроить Active Directory?", k=10)
# Найдено 10 документов, после компрессии осталось, например, 6
```

### Пример 3: Настройка в коде

```python
from rag_system import SysAdminRAG
from document_compressor import DocumentCompressor

# Создаем компрессор с кастомными настройками
compressor = DocumentCompressor(
    use_compression=True,
    similarity_threshold=0.8,
    embeddings_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Используем в RAG системе
rag = SysAdminRAG(use_document_compression=True)
rag.compressor = compressor  # Заменяем стандартный компрессор
rag.load_index()
```

## Сравнение с/без компрессии

### Без компрессии
- ✅ Все найденные документы используются
- ✅ Больше контекста для LLM
- ❌ Может содержать нерелевантную информацию
- ❌ Больше токенов, медленнее

### С компрессией
- ✅ Только релевантные документы
- ✅ Меньше токенов, быстрее
- ✅ Лучшее качество ответов
- ❌ Может отфильтровать важную информацию (если порог слишком высокий)

## Рекомендации

1. **Начните с компрессии отключенной** для понимания базового поведения
2. **Включите компрессию** и сравните результаты
3. **Настройте порог** в зависимости от ваших данных:
   - Если ответы слишком общие → увеличьте порог
   - Если важная информация теряется → уменьшите порог
4. **Экспериментируйте с k**: при компрессии можно использовать больше k (например, 10-15), так как нерелевантные будут отфильтрованы

## Troubleshooting

### Компрессор не инициализируется

**Проблема**: `Предупреждение: не удалось инициализировать компрессор`

**Решение**:
1. Убедитесь, что установлен langchain: `pip install langchain langchain-community`
2. Убедитесь, что установлен sentence-transformers: `pip install sentence-transformers`
3. Проверьте доступность модели эмбеддингов

### Слишком много документов отфильтровывается

**Проблема**: После компрессии остается очень мало документов

**Решение**: Уменьшите `compression_threshold`:
```python
rag = SysAdminRAG(
    use_document_compression=True,
    compression_similarity_threshold=0.65  # Было 0.76
)
```

### Компрессия не работает

**Проблема**: Документы не фильтруются

**Решение**:
1. Проверьте, что компрессия включена: `use_document_compression=True`
2. Проверьте логи на наличие ошибок
3. Убедитесь, что LangChain установлен корректно

## Дополнительные компрессоры

В будущем можно добавить другие компрессоры из LangChain:

- `LLMChainExtractor`: Использует LLM для извлечения релевантных частей
- `DocumentCompressorPipeline`: Комбинация нескольких компрессоров
- Кастомные компрессоры для специфических задач

Пример расширенного pipeline:

```python
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor
)

# Комбинация фильтрации и извлечения
pipeline = DocumentCompressorPipeline(
    transformers=[
        EmbeddingsFilter(...),  # Сначала фильтруем
        LLMChainExtractor(...)  # Потом извлекаем релевантные части
    ]
)
```

