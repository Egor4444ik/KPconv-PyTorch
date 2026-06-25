"""
Скрипт для организации LAS файлов в структуру S3DIS.
Распределяет точки по классам (ground, bush, tree) в соответствии с LAS классификацией.
"""

import laspy
import numpy as np
import os
from pathlib import Path

class LASDataOrganizer:
    """Организует LAS файлы в структуру S3DIS с аннотациями по классам"""
    
    # Маппинг LAS классификации на классы леса
    LAS_CLASS_MAPPING = {
        2: 'ground',      # Ground
        3: 'bush',        # Low vegetation / Bush
        4: 'tree',        # Medium vegetation / Tree
        5: 'tree',        # High vegetation / Tree
        6: 'tree',        # Building (for point clouds, consider as structure)
    }
    
    def __init__(self, base_path='Data/S3DIS'):
        self.base_path = Path(base_path)
        
    def organize_las_files(self, las_files_mapping):
        """
        Организует LAS файлы в структуру S3DIS.
        
        Args:
            las_files_mapping: dict вида {
                'Area_1': [('path/to/file1.las', 'forest_1'), ...],
                'Area_2': [('path/to/file2.las', 'forest_1'), ...],
            }
        
        Example:
            mapping = {
                'Area_1': [('Area_classified.las', 'forest_1')],
                'Area_2': [('LAS_Uchastok_2.las', 'forest_1')],
            }
            organizer.organize_las_files(mapping)
        """
        
        for area_name, file_list in las_files_mapping.items():
            for las_file, region_name in file_list:
                self._process_las_file(area_name, region_name, las_file)
    
    def _process_las_file(self, area_name, region_name, las_file_path):
        """Обрабатывает один LAS файл и распределяет точки по классам"""
        
        las_file_path = Path(las_file_path)
        if not las_file_path.exists():
            print(f"❌ Файл не найден: {las_file_path}")
            return
        
        print(f"\n📂 Обработка: {las_file_path}")
        print(f"   Area: {area_name}, Region: {region_name}")
        
        # Создать структуру папок
        annotation_dir = self.base_path / area_name / region_name / 'Annotations'
        annotation_dir.mkdir(parents=True, exist_ok=True)
        
        # Прочитать LAS файл
        try:
            las = laspy.read(las_file_path)
        except Exception as e:
            print(f"❌ Ошибка чтения LAS: {e}")
            return
        
        # Получить координаты и цвета
        points = np.vstack((las.x, las.y, las.z)).T
        
        # Попытаться получить цвета
        try:
            red = (las.red // 256).astype(np.uint8)
            green = (las.green // 256).astype(np.uint8)
            blue = (las.blue // 256).astype(np.uint8)
        except:
            print("⚠️  Цвета не найдены, используются нули")
            red = np.zeros(len(points), dtype=np.uint8)
            green = np.zeros(len(points), dtype=np.uint8)
            blue = np.zeros(len(points), dtype=np.uint8)
        
        # Получить классификацию
        if hasattr(las, 'classification'):
            classification = las.classification
        else:
            print("⚠️  Классификация не найдена в LAS, используется класс 2 (ground) для всех")
            classification = np.full(len(points), 2, dtype=np.uint8)
        
        # Распределить точки по классам
        point_data = np.column_stack((
            points[:, 0], points[:, 1], points[:, 2],
            red, green, blue
        ))
        
        class_counts = {}
        class_files = {}
        
        for class_num, class_name in self.LAS_CLASS_MAPPING.items():
            mask = classification == class_num
            if np.any(mask):
                class_points = point_data[mask]
                class_counts[class_name] = len(class_points)
                class_files[class_name] = class_points
        
        # Если нет классификации, выбрать все как ground
        if not class_counts:
            class_counts['ground'] = len(point_data)
            class_files['ground'] = point_data
        
        # Сохранить файлы по классам
        class_counter = {cls: 1 for cls in class_counts.keys()}
        
        for class_name, class_points in class_files.items():
            file_num = class_counter[class_name]
            filename = annotation_dir / f"{class_name}_{file_num}.txt"
            
            fmt = ["%.8f", "%.8f", "%.8f", "%d", "%d", "%d"]
            np.savetxt(filename, class_points, fmt=fmt, delimiter=" ")
            print(f"   ✅ {class_name}_{file_num}.txt: {len(class_points)} точек")
            
            class_counter[class_name] += 1
        
        print(f"   📊 Всего точек: {len(points)}")
        print(f"   📊 Распределение: {class_counts}")


def find_las_files(directory, pattern="*.las"):
    """Поиск LAS файлов в директории"""
    directory = Path(directory)
    return list(directory.glob(pattern))


if __name__ == "__main__":
    import sys
    
    # ========== КОНФИГУРАЦИЯ ==========
    # Указать, где находятся ваши LAS файлы
    # и в какие Area их поместить
    
    # Вариант 1: Автоматический поиск (если LAS файлы в Data/S3DIS)
    # las_files = find_las_files("Data/S3DIS", "*.las")
    
    # Вариант 2: Явное указание файлов
    las_mapping = {
        'Area_1': [
            ('Data/S3DIS/Area_classified.las', 'forest_1'),
        ],
        'Area_2': [
            ('Data/S3DIS/LAS_Uchastok_2_color.las', 'forest_1'),
        ],
        'Area_3': [
            ('Data/S3DIS/LAS_Uchastok_2.las', 'forest_1'),
        ],
    }
    
    print("\n" + "="*60)
    print("🌲 Организатор LAS файлов для S3DIS структуры")
    print("="*60)
    
    organizer = LASDataOrganizer()
    
    # Если передан аргумент, использовать как путь к файлу
    if len(sys.argv) > 1:
        custom_file = sys.argv[1]
        if Path(custom_file).exists():
            las_mapping = {
                'Area_1': [(custom_file, 'forest_1')]
            }
            print(f"\n📌 Используется файл: {custom_file}")
    
    organizer.organize_las_files(las_mapping)
    
    print("\n" + "="*60)
    print("✅ Готово!")
    print("="*60 + "\n")
