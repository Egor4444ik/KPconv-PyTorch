# 🚀 БЫСТРЫЙ СТАРТ: Организация LAS данных для KPConv

## САМЫЙ БЫСТРЫЙ СПОСОБ (3 команды)

```bash
# 1. Найти ваши LAS файлы
bash find_las_files.sh

# 2. Интерактивная организация (рекомендуется)
python3 setup_las_data_interactive.py

# 3. Проверить результаты
python3 -c "
import numpy as np
from pathlib import Path
for f in Path('Data/S3DIS').glob('Area_*/forest_1/Annotations/*.txt'):
    data = np.loadtxt(f)
    print(f'{f.parent.parent.parent.name}: {f.name} - {len(data)} points')
"
```

---

## ЕСЛИ ИНТЕРАКТИВНЫЙ СКРИПТ НЕ РАБОТАЕТ

### Способ 1: Прямое указание файлов

Отредактируйте `Data/organize_las_data.py` (строки 80-89):

```python
las_mapping = {
    'Area_1': [
        ('/абсолютный/путь/к/Area_classified.las', 'forest_1'),
    ],
    'Area_2': [
        ('/абсолютный/путь/к/LAS_Uchastok_2_color.las', 'forest_1'),
    ],
    'Area_3': [
        ('/абсолютный/путь/к/LAS_Uchastok_2.las', 'forest_1'),
    ],
}
```

Затем:
```bash
cd Data
python3 organize_las_data.py
```

### Способ 2: Использование продвинутого организатора

Если базовый не работает или вам нужно объединить файлы:

```python
from advanced_las_organizer import AdvancedLASOrganizer

# Один файл с классификацией
organizer = AdvancedLASOrganizer()
organizer.merge_las_by_classification('Data/S3DIS/Area_classified.las', 'Area_1')

# Несколько файлов
organizer.merge_multiple_files_to_area({
    'color_file': 'Data/S3DIS/LAS_Uchastok_2_color.las',
    'classification': 'Data/S3DIS/Area_classified.las'
}, 'Area_2')

# Разделение по высоте (без классификации)
from advanced_las_organizer import ManualClassificationSplitter
splitter = ManualClassificationSplitter()
splitter.split_by_height_heuristic('Data/S3DIS/LAS_Uchastok_2.las', 'Area_3')
```

---

## ПРОВЕРКА РЕЗУЛЬТАТОВ

Ваша структура должна быть:

```
Data/S3DIS/
├── Area_1/
│   └── forest_1/
│       └── Annotations/
│           ├── ground_1.txt      ✅
│           ├── bush_1.txt        ✅
│           └── tree_1.txt        ✅
│
├── Area_2/
│   └── forest_1/
│       └── Annotations/
│           ├── ground_1.txt      ✅
│           ├── bush_1.txt        ✅
│           └── tree_1.txt        ✅
│
├── Area_3/
│   └── forest_1/
│       └── Annotations/
│           ├── ground_1.txt      ✅
│           ├── bush_1.txt        ✅
│           └── tree_1.txt        ✅
│
├── original_ply/
└── input_0.030/
```

**Проверить количество файлов:**
```bash
find Data/S3DIS/Area_*/forest_1/Annotations -name "*.txt" | wc -l
# Должно быть минимум 3 (по одному файлу каждого класса)
```

---

## ОБНОВЛЕНИЕ train_S3DIS.py

**ВАЖНО**: Обновить строки 100-107 в `train_S3DIS.py`

### Вариант 1: Если у вас 3 Areas

```python
# Proportion of validation scenes
self.cloud_names = ['Area_1', 'Area_2', 'Area_3']
lasConverter(self.cloud_names).toTxt()
self.all_splits = [0, 0, 1]
self.validation_split = 1
```

### Вариант 2: Если у вас 2 Areas

```python
self.cloud_names = ['Area_1', 'Area_2']
lasConverter(self.cloud_names).toTxt()
self.all_splits = [0, 1]
self.validation_split = 1
```

### Вариант 3: Если у вас 6 Areas (все)

```python
self.cloud_names = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
lasConverter(self.cloud_names).toTxt()
self.all_splits = [0, 0, 0, 0, 0, 1]
self.validation_split = 1
```

---

## ЗАПУСК ОБУЧЕНИЯ

```bash
# Убедитесь, что вы в корне проекта
cd /Users/egorsmirnov/Documents/work/Laboratory/ForestTaxation/KPConv-PyTorch

# Запустить обучение
python3 train_S3DIS.py

# В отдельном терминале, мониторить прогресс
tail -f results/Log_*/training.txt
```

---

## ВОЗМОЖНЫЕ ПРОБЛЕМЫ И РЕШЕНИЯ

### ❌ `ModuleNotFoundError: No module named 'laspy'`

```bash
pip install laspy[lazrs]
```

### ❌ `FileNotFoundError: '[Errno 2] No such file or directory'`

**Решение**: Убедитесь, что:
1. Полный путь к LAS файлу корректный
2. Файл не открыт в другой программе
3. Используйте абсолютные пути, а не относительные

### ❌ `Нет классификации в LAS файле`

**Решение**: Используйте `ManualClassificationSplitter`:

```python
from Data.advanced_las_organizer import ManualClassificationSplitter
splitter = ManualClassificationSplitter()
splitter.split_by_height_heuristic('Data/S3DIS/LAS_Uchastok_2.las',
                                  ground_threshold=0.5,
                                  bush_threshold=3.0)
```

### ❌ `Warning: only X area has data, need at least 2`

**Решение**: Убедитесь, что заполнены минимум 2 Area (для обучения и валидации)

### ❌ Файлы весят очень много, обработка долгая

**Решение**: Это нормально для больших облаков точек. Оставьте скрипт работать, или:
1. Используйте несколько файлов разных размеров
2. Уменьшите размер файла в QGIS перед обработкой

---

## ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ

### Структура данных S3DIS ожидает:

- **Area_X** - районы/регионы
- **forest_1** - конкретный лес/помещение в районе
- **Annotations** - папка с аннотациями по классам
- **{class}_{num}.txt** - файлы точек, разделённые по классам

### Формат .txt файлов:

Каждый файл содержит точки в формате:
```
X Y Z R G B
X Y Z R G B
...
```

Где:
- X, Y, Z - координаты (float)
- R, G, B - цвета 0-255 (int)
- Разделитель: пробел

---

## БЫСТРАЯ ДИАГНОСТИКА

```bash
# Проверить структуру
ls -la Data/S3DIS/Area_1/forest_1/Annotations/

# Проверить первые строки файла
head -5 Data/S3DIS/Area_1/forest_1/Annotations/ground_1.txt

# Проверить количество точек
wc -l Data/S3DIS/Area_1/forest_1/Annotations/ground_1.txt

# Проверить, что файлы не пусты
find Data/S3DIS -name "*.txt" -size 0

# Посчитать всего точек
find Data/S3DIS -name "*.txt" -exec wc -l {} + | tail -1
```

---

## ЕСЛИ ВСЁ РАБОТАЕТ

```bash
# Проверить готовность данных
python3 -c "
from datasets.S3DIS import S3DISDataset
from utils.config import Config
config = Config()
config.num_classes = 3
dataset = S3DISDataset(config, set='training', use_potentials=False)
print(f'✅ Dataset loaded: {len(dataset)} clouds')
print(f'✅ Total points: {dataset.input_colors}')
"

# Запустить обучение!
python3 train_S3DIS.py
```

---

**Created**: 2026-06-16  
**For**: KPConv-PyTorch with LAS Forest Data  
**Status**: ✅ Ready to use
