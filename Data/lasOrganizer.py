"""
Расширенный организатор LAS для сложных случаев:
- Слияние нескольких файлов в один Area
- Использование информации из разных источников
- Маппинг классов вручную
"""

import laspy
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional

class AdvancedLASOrganizer:
    """Продвинутая организация LAS данных с гибким маппингом классов"""
    
    def __init__(self, base_path='Data/S3DIS'):
        self.base_path = Path(base_path)
    
    def merge_las_by_classification(self, 
                                   classified_las: str, 
                                   area_name: str = 'Area_1',
                                   region_name: str = 'forest_1'):
        """
        Используйте один LAS с готовой классификацией для разделения по классам.
        
        Args:
            classified_las: путь к LAS с полноценной классификацией
            area_name: название Area
            region_name: название региона (папки внутри Area)
        
        Example:
            organizer.merge_las_by_classification('Area_classified.las')
        """
        
        las_path = Path(classified_las)
        if not las_path.exists():
            print(f"❌ Файл не найден: {las_path}")
            return
        
        print(f"\n📂 Обработка классифицированного LAS: {las_path}")
        
        las = laspy.read(las_path)
        points = np.vstack((las.x, las.y, las.z)).T
        
        # Получить RGB
        try:
            colors = np.column_stack([
                (las.red // 256).astype(np.uint8),
                (las.green // 256).astype(np.uint8),
                (las.blue // 256).astype(np.uint8),
            ])
        except:
            colors = np.zeros((len(points), 3), dtype=np.uint8)
        
        # Получить классификацию
        if hasattr(las, 'classification'):
            classification = las.classification
        else:
            print("⚠️  Нет классификации, используется класс 2 для всех")
            classification = np.full(len(points), 2, dtype=np.uint8)
        
        # Сохранить по классам
        self._save_by_classes(
            points, colors, classification,
            area_name, region_name
        )
    
    def merge_multiple_files_to_area(self,
                                    files: Dict[str, str],
                                    area_name: str = 'Area_1',
                                    region_name: str = 'forest_1'):
        """
        Объедините несколько файлов и распределите по классам.
        
        Args:
            files: {
                'color_file': 'путь_с_цветами.las',
                'height_data': 'путь_с_высотой.las',  # опционально
                'classification': 'путь_с_классами.las'  # опционально
            }
        
        Example:
            organizer.merge_multiple_files_to_area({
                'color_file': 'LAS_Uchastok_2_color.las',
                'classification': 'Area_classified.las'
            })
        """
        
        print(f"\n📦 Объединение нескольких файлов в {area_name}/{region_name}")
        
        all_points = []
        all_colors = []
        all_classes = []
        
        # Прочитать основной файл (обычно color)
        color_file = files.get('color_file') or files.get('main_file')
        if not color_file:
            print("❌ Требуется 'color_file' или 'main_file'")
            return
        
        las = laspy.read(color_file)
        points = np.vstack((las.x, las.y, las.z)).T
        
        try:
            colors = np.column_stack([
                (las.red // 256).astype(np.uint8),
                (las.green // 256).astype(np.uint8),
                (las.blue // 256).astype(np.uint8),
            ])
        except:
            colors = np.zeros((len(points), 3), dtype=np.uint8)
        
        all_points.append(points)
        all_colors.append(colors)
        
        # Попытаться получить классификацию
        if hasattr(las, 'classification'):
            all_classes.append(las.classification)
        else:
            all_classes.append(None)
        
        # Если есть файл с классификацией
        class_file = files.get('classification')
        if class_file:
            las_class = laspy.read(class_file)
            if len(las_class) == len(las):
                all_classes[-1] = las_class.classification
                print(f"✅ Загружена классификация из {class_file}")
        
        # Объединить
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        merged_classes = all_classes[0] if all_classes[0] is not None else np.full(len(merged_points), 2)
        
        print(f"📊 Объединено: {len(merged_points)} точек")
        
        # Сохранить
        self._save_by_classes(merged_points, merged_colors, merged_classes,
                            area_name, region_name)
    
    def _save_by_classes(self, points, colors, classification,
                        area_name, region_name):
        """Сохранить точки разделённые по классам"""
        
        # Классы лесных данных
        classes_map = {
            'ground': [2],
            'bush': [3],
            'tree': [4, 5, 6]
        }
        
        # Создать структуру
        anno_dir = self.base_path / area_name / region_name / 'Annotations'
        anno_dir.mkdir(parents=True, exist_ok=True)
        
        # Разделить и сохранить
        class_counter = {cls: 1 for cls in classes_map.keys()}
        
        for class_name, class_nums in classes_map.items():
            mask = np.isin(classification, class_nums)
            if np.any(mask):
                class_points = points[mask]
                class_colors = colors[mask]
                
                data = np.column_stack([class_points, class_colors])
                
                filename = anno_dir / f"{class_name}_{class_counter[class_name]}.txt"
                np.savetxt(filename, data, 
                          fmt=["%.8f", "%.8f", "%.8f", "%d", "%d", "%d"],
                          delimiter=" ")
                
                print(f"   ✅ {filename.name}: {len(class_points)} точек")
                class_counter[class_name] += 1


class ManualClassificationSplitter:
    """Разделение файла на классы вручную (если нет классификации)"""
    
    @staticmethod
    def split_by_height_heuristic(las_file: str,
                                 area_name: str = 'Area_1',
                                 ground_threshold: float = 0.5,
                                 bush_threshold: float = 3.0):
        """
        Разделить по высоте (эвристика):
        - ground: z < ground_threshold
        - bush: ground_threshold <= z < bush_threshold  
        - tree: z >= bush_threshold
        
        Args:
            las_file: путь к LAS файлу
            ground_threshold: граница для земли (метры)
            bush_threshold: граница для кустарников (метры)
        
        Example:
            splitter = ManualClassificationSplitter()
            splitter.split_by_height_heuristic('LAS_Uchastok_2.las',
                                             ground_threshold=0.3,
                                             bush_threshold=2.5)
        """
        
        las = laspy.read(las_file)
        points = np.vstack((las.x, las.y, las.z)).T
        
        try:
            colors = np.column_stack([
                (las.red // 256).astype(np.uint8),
                (las.green // 256).astype(np.uint8),
                (las.blue // 256).astype(np.uint8),
            ])
        except:
            colors = np.zeros((len(points), 3), dtype=np.uint8)
        
        # Минимизировать Z для relative height
        z_min = points[:, 2].min()
        relative_z = points[:, 2] - z_min
        
        # Классификация по высоте
        classification = np.zeros(len(points), dtype=np.uint8)
        classification[relative_z < ground_threshold] = 2  # ground
        mask_bush = (relative_z >= ground_threshold) & (relative_z < bush_threshold)
        classification[mask_bush] = 3  # bush
        classification[relative_z >= bush_threshold] = 4  # tree
        
        print(f"📊 Распределение по высоте:")
        print(f"   Ground (z < {ground_threshold}): {np.sum(classification == 2)}")
        print(f"   Bush ({ground_threshold} <= z < {bush_threshold}): {np.sum(classification == 3)}")
        print(f"   Tree (z >= {bush_threshold}): {np.sum(classification == 4)}")
        
        # Сохранить
        organizer = AdvancedLASOrganizer()
        organizer._save_by_classes(points, colors, classification,
                                  area_name, 'forest_1')


# Примеры использования
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌲 ПРОДВИНУТЫЙ ОРГАНИЗАТОР LAS ФАЙЛОВ")
    print("="*70 + "\n")
    
    # Пример 1: Использовать один файл с классификацией
    print("Пример 1: Один файл с классификацией")
    print("─" * 70)
    organizer = AdvancedLASOrganizer()
    # organizer.merge_las_by_classification('Data/S3DIS/Area_classified.las')
    
    # Пример 2: Объединить несколько файлов
    print("\nПример 2: Объединить несколько файлов")
    print("─" * 70)
    # organizer.merge_multiple_files_to_area({
    #     'color_file': 'Data/S3DIS/LAS_Uchastok_2_color.las',
    #     'classification': 'Data/S3DIS/Area_classified.las'
    # }, area_name='Area_1')
    
    # Пример 3: Разделить по высоте (если нет классификации)
    print("\nПример 3: Разделить по высоте (эвристика)")
    print("─" * 70)
    # splitter = ManualClassificationSplitter()
    # splitter.split_by_height_heuristic('Data/S3DIS/LAS_Uchastok_2.las',
    #                                    ground_threshold=0.3,
    #                                    bush_threshold=2.5)
    
    print("\n" + "="*70)
    print("💡 Раскомментируйте примеры выше для использования")
    print("="*70 + "\n")
