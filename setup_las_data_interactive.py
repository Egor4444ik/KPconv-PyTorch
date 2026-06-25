#!/usr/bin/env python3
"""
Скрипт для быстрого преобразования LAS в S3DIS структуру.
Выполняет все необходимые шаги за раз.
"""

import sys
import os
from pathlib import Path

def print_banner(text):
    """Красивый вывод"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_dependencies():
    """Проверить требуемые пакеты"""
    print_banner("🔍 ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    
    required = ['laspy', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - НЕ УСТАНОВЛЕН")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Установите недостающие пакеты:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True

def find_las_files():
    """Поиск LAS файлов"""
    print_banner("🔍 ПОИСК LAS ФАЙЛОВ")
    
    search_paths = [
        Path('.'),
        Path('Data'),
        Path('Data/S3DIS'),
        Path.home() / 'Downloads',
    ]
    
    found_files = {}
    
    for search_path in search_paths:
        if search_path.exists():
            for las_file in search_path.glob('*.las'):
                found_files[str(las_file)] = las_file
                print(f"  ✅ {las_file}")
    
    if not found_files:
        print("  ❌ LAS файлы не найдены")
        return None
    
    return found_files

def interactive_setup():
    """Интерактивная настройка"""
    print_banner("⚙️  ИНТЕРАКТИВНАЯ НАСТРОЙКА")
    
    las_files = find_las_files()
    if not las_files:
        print("\n❌ Не найдено LAS файлов. Поместите их в текущую папку.")
        return None
    
    print(f"\n✅ Найдено {len(las_files)} файлов")
    
    # Выбор файлов
    files_list = list(las_files.keys())
    mapping = {}
    area_num = 1
    
    for file_path in files_list:
        print(f"\n📄 Файл: {Path(file_path).name}")
        
        area_input = input(f"  В какой Area? (по умолчанию Area_{area_num}): ").strip()
        area_name = f"Area_{area_input}" if area_input else f"Area_{area_num}"
        
        if area_name not in mapping:
            mapping[area_name] = []
        
        mapping[area_name].append((file_path, 'forest_1'))
        area_num += 1
    
    # Выбрать validation Area
    areas = list(mapping.keys())
    print(f"\n📋 Available Areas: {', '.join(areas)}")
    val_area = input(f"Какой Area использовать для валидации? (по умолчанию {areas[-1]}): ").strip()
    if not val_area:
        val_area = areas[-1]
    
    print(f"\n✅ Конфигурация:")
    print(f"  Training Areas: {[a for a in areas if a != val_area]}")
    print(f"  Validation Area: {val_area}")
    
    return mapping

def run_organization(mapping):
    """Запустить организацию данных"""
    print_banner("🚀 ОРГАНИЗАЦИЯ ДАННЫХ")
    
    try:
        # Динамический импорт
        sys.path.insert(0, str(Path('Data').absolute()))
        from organize_las_data import LASDataOrganizer
        
        organizer = LASDataOrganizer()
        organizer.organize_las_files(mapping)
        
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_structure():
    """Проверить структуру"""
    print_banner("✔️  ПРОВЕРКА СТРУКТУРЫ")
    
    base_path = Path('Data/S3DIS')
    all_ok = True
    
    for area_dir in sorted(base_path.glob('Area_*')):
        anno_dir = area_dir / 'forest_1' / 'Annotations'
        if anno_dir.exists():
            files = list(anno_dir.glob('*.txt'))
            if files:
                print(f"  ✅ {area_dir.name}: {len(files)} файлов")
                for f in files:
                    try:
                        import numpy as np
                        data = np.loadtxt(f)
                        print(f"     • {f.name}: {len(data)} точек")
                    except:
                        print(f"     • {f.name}: ⚠️  ошибка чтения")
            else:
                print(f"  ⚠️  {area_dir.name}: пуста")
                all_ok = False
        else:
            print(f"  ⚠️  {area_dir.name}: структура не создана")
            all_ok = False
    
    return all_ok

def show_next_steps():
    """Показать следующие шаги"""
    print_banner("📋 СЛЕДУЮЩИЕ ШАГИ")
    
    print("""
1. Если структура ✅, вы готовы к обучению!

2. Обновите train_S3DIS.py:

   ```python
   # Найдите строку около линии 100:
   self.cloud_names = ['Area_1', 'Area_2']
   
   # Измените на ваши Areas (например):
   self.cloud_names = ['Area_1', 'Area_2', 'Area_3']
   self.all_splits = [0, 0, 1]  # Последняя для валидации
   self.validation_split = 1
   ```

3. Запустите обучение:

   ```bash
   python3 train_S3DIS.py
   ```

4. Мониторьте процесс:

   ```bash
   # В отдельном терминале
   tail -f results/Log_*/training.txt
   ```
    """)

def main():
    """Главная функция"""
    print_banner("🌲 KPConv S3DIS LAS ДАННЫЕ - БЫСТРАЯ НАСТРОЙКА")
    
    # Шаг 1: Проверить зависимости
    if not check_dependencies():
        sys.exit(1)
    
    # Шаг 2: Интерактивная настройка
    mapping = interactive_setup()
    if not mapping:
        sys.exit(1)
    
    # Шаг 3: Организация данных
    proceed = input("\nПродолжить организацию? (y/n): ").strip().lower()
    if proceed != 'y':
        print("❌ Отменено")
        sys.exit(1)
    
    if not run_organization(mapping):
        sys.exit(1)
    
    # Шаг 4: Проверка структуры
    if verify_structure():
        print("✅ Структура создана успешно!")
    else:
        print("⚠️  Некоторые Area могут быть пусты")
    
    # Шаг 5: Показать следующие шаги
    show_next_steps()
    
    print_banner("✅ ГОТОВО!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Отменено пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
