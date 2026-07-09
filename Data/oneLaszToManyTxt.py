import laspy
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import re
import os
import math

from Data.lasColoring import ObjColloring

class lasToTxt:
    def __init__(self, base_path='Data/S3DIS', Areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']):
        self.Areas = Areas
        self.base_path = Path(base_path)
        self.folder_paths = [Path(f"Data/S3DIS/{Area}") for Area in self.Areas]
        self.S3DIS_path = Path(f"Data/S3DIS")
        self.class_names = {1: 'Unassigned', 2: 'Ground', 3: 'LowerBushe', 4: 'HighBushe', 5: 'Tree', 7: 'Noise', 12: 'Overlap or Reserved'}
        self.las_file_name = self.file_finding()


    def file_finding(self):
        folder_path = self.S3DIS_path
        if folder_path.exists():
            laspattern = re.compile(r'([^\\/]+)(?=\.las$)')
            laZpattern = re.compile(r'([^\\/]+)(?=\.laz$)')

            for file in os.listdir(folder_path):
                match = laspattern.search(file) or laZpattern.search(file)
                if match:
                    print('las/laz file founded:', file)
                    
                    return folder_path.joinpath(file)
        else:
            print("folder_path doesnt exists")


    def _split_bbox(self, x_min, x_max, y_min, y_max):
        n = len(self.Areas)
        cols = int(math.ceil(math.sqrt(n * (x_max - x_min) / (y_max - y_min))))
        if cols > n:
            cols = n
        rows = int(math.ceil(n / cols))
        x_edges = np.linspace(x_min, x_max, cols + 1)
        y_edges = np.linspace(y_min, y_max, rows + 1)
        areas = []
        for i in range(rows):
            for j in range(cols):
                areas.append((x_edges[j], x_edges[j+1], y_edges[i], y_edges[i+1]))
        return areas[:n]

    
    def one_to_many_by_classes(self, region_name: str = 'forest_1'):
        # 1. Сначала определяем границы облака (легкая операция)
        with laspy.open(self.las_file_name) as las:
            x_min, x_max = las.x.min(), las.x.max()
            y_min, y_max = las.y.min(), las.y.max()
            bboxes = self._split_bbox(x_min, x_max, y_min, y_max)
            
            has_color = hasattr(las, 'red')
            if has_color:
                print('Las object has real colors.')
            else:
                print('Las object is not colored. Procedural coloring will be applied.')

            fmt_spec = ["%.8f", "%.8f", "%.8f", "%d", "%d", "%d", "%f", "%f", "%f", "%f"]

            # 2. Обрабатываем КАЖДУЮ ЗОНУ ОТДЕЛЬНО, загружая только нужные точки
            for area_idx, (x0, x1, y0, y1) in enumerate(bboxes):
                area_name = self.Areas[area_idx]
                if area_name == 'Area_1':
                    continue
                annot_dir = self.base_path / area_name / 'Annotations'
                annot_dir.mkdir(parents=True, exist_ok=True)
                
                print(f'Processing {area_name}...')
                
                # Загружаем только точки в текущей зоне
                zone_mask = (las.x >= x0) & (las.x < x1) & (las.y >= y0) & (las.y < y1)
                
                if zone_mask.sum() < 100:
                    print(f'{area_name}: too few points ({zone_mask.sum()}), skipping.')
                    continue
                
                print(f'{area_name}: {zone_mask.sum()} points in zone')
                
                # Извлекаем данные только для этой зоны
                zx = np.asarray(las.x[zone_mask])
                zy = np.asarray(las.y[zone_mask])
                zz = np.asarray(las.z[zone_mask])
                zcls = np.asarray(las.classification[zone_mask])
                zi = np.asarray(las.intensity[zone_mask], dtype=np.float32)
                zrn = np.asarray(las.return_number[zone_mask], dtype=np.float32)
                znr = np.asarray(las.number_of_returns[zone_mask], dtype=np.float32)
                
                if has_color:
                    zr = np.asarray(las.red[zone_mask], dtype=np.uint8)
                    zg = np.asarray(las.green[zone_mask], dtype=np.uint8)
                    zb = np.asarray(las.blue[zone_mask], dtype=np.uint8)
                else:
                    zr = np.zeros_like(zx, dtype=np.uint8)
                    zg = np.zeros_like(zx, dtype=np.uint8)
                    zb = np.zeros_like(zx, dtype=np.uint8)
                
                # 3. Для каждого класса внутри зоны
                for cls_id, cls_name in self.class_names.items():
                    cmask = (zcls == cls_id)
                    if cmask.sum() == 0:
                        continue
                    
                    cx, cy, cz = zx[cmask], zy[cmask], zz[cmask]
                    cr, cg, cb = zr[cmask], zg[cmask], zb[cmask]
                    ci = zi[cmask]
                    crn = zrn[cmask]
                    cnr = znr[cmask]

                    del cmask
                    
                    del zx, zy, zz, zcls, zi, zrn, znr, zr, zg, zb
                    
                    # Собираем данные класса
                    cls_data = np.column_stack((
                        cx, cy, cz, cr, cg, cb, ci, crn, cnr,
                        np.full(cx.shape, cls_id, dtype=np.float32)
                    ))

                    del crn, cnr

                    # 4. Кластеризация
                    if cls_id == 2:  # Ground – один объект
                        instances = [cls_data]
                        print(f'  {cls_name}: 1 instance')
                    else:
                        coords = np.column_stack((cx, cy, cz))
                        del cx, cy, cz
                        clustering = DBSCAN(eps=0.3, min_samples=10).fit(coords)
                        del coords
                        labels = clustering.labels_
                        instances = []
                        for lbl in np.unique(labels):
                            if lbl == -1:
                                continue
                            instances.append(cls_data[labels == lbl])
                        print(f'  {cls_name}: {len(instances)} instances')
                    
                    del cls_data
                    # 5. Раскраска и сохранение
                    for inst_id, inst_pts in enumerate(instances, start=1):
                        if not has_color:
                            inst_x = inst_pts[:, 0]
                            inst_y = inst_pts[:, 1]
                            inst_z = inst_pts[:, 2]
                            new_r, new_g, new_b = self.color_instance(inst_x, inst_y, inst_z, cls_id)
                            inst_pts[:, 3] = new_r
                            inst_pts[:, 4] = new_g
                            inst_pts[:, 5] = new_b
                        
                        fname = annot_dir / f'{cls_name}_{inst_id}.txt'
                        np.savetxt(fname, inst_pts, fmt=fmt_spec, delimiter=' ')
                    
                    # Очищаем память
                    del instances
                
                # Очищаем данные зоны из памяти
                
                
                print(f'{area_name} done.')
    
    def color_instance(self, x, y, z, class_id):
        """
        Генерирует RGB‑цвета для одного экземпляра (дерева, куста и т.д.)
        с учётом высоты и класса.
        """
        n = len(x)
        r = np.zeros(n, dtype=np.uint8)
        g = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=np.uint8)

        # Нормализуем высоту в пределах экземпляра (0 – низ, 1 – верх)
        if n > 1 and z.max() > z.min():
            z_norm = (z - z.min()) / (z.max() - z.min())
        else:
            z_norm = np.zeros(n)

        if class_id in (3, 4, 5):   # Растительность
            # Цвета: низ – коричневый (ствол), верх – зелёный (крона)
            bottom = np.array([139, 90, 43])   # Brown
            if class_id == 3:       # Low vegetation
                top = np.array([34, 139, 34])
            elif class_id == 4:     # Medium vegetation
                top = np.array([0, 128, 0])
            else:                   # High vegetation (trees)
                top = np.array([0, 100, 0])

            r = (bottom[0] * (1 - z_norm) + top[0] * z_norm).astype(np.uint8)
            g = (bottom[1] * (1 - z_norm) + top[1] * z_norm).astype(np.uint8)
            b = (bottom[2] * (1 - z_norm) + top[2] * z_norm).astype(np.uint8)

            # Лёгкий шум
            r = np.clip(r + np.random.randint(-10, 11, n), 0, 255).astype(np.uint8)
            g = np.clip(g + np.random.randint(-15, 16, n), 0, 255).astype(np.uint8)
            b = np.clip(b + np.random.randint(-10, 11, n), 0, 255).astype(np.uint8)

        elif class_id == 2:   # Земля – коричневый с вариациями
            base = np.array([139, 90, 43])
            noise = np.random.randint(-15, 16, n)
            r = np.clip(np.full(n, base[0], dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            g = np.clip(np.full(n, base[1], dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            b = np.clip(np.full(n, base[2], dtype=np.int16) + noise, 0, 255).astype(np.uint8)

        elif class_id == 7:   # Шум – равномерный RGB
            third = n // 3
            colors = np.zeros((n, 3), dtype=np.uint8)
            colors[:third, 0] = np.random.randint(220, 255, third)
            colors[:third, 1] = np.random.randint(0, 20, third)
            colors[:third, 2] = np.random.randint(0, 20, third)
            if n >= 2*third:
                colors[third:2*third, 0] = np.random.randint(0, 20, third)
                colors[third:2*third, 1] = np.random.randint(220, 255, third)
                colors[third:2*third, 2] = np.random.randint(0, 20, third)
                colors[2*third:, 0] = np.random.randint(0, 20, n-2*third)
                colors[2*third:, 1] = np.random.randint(0, 20, n-2*third)
                colors[2*third:, 2] = np.random.randint(220, 255, n-2*third)
            else:
                # если точек меньше, просто случайно распределяем
                colors[third:, 0] = np.random.randint(0, 20, n-third)
                colors[third:, 1] = np.random.randint(0, 20, n-third)
                colors[third:, 2] = np.random.randint(220, 255, n-third)
            perm = np.random.permutation(n)
            r = colors[perm, 0]
            g = colors[perm, 1]
            b = colors[perm, 2]

        elif class_id == 12:  # Overlap – оранжевый с полосами
            base_r = np.random.randint(200, 255, n)
            base_g = np.random.randint(150, 210, n)
            base_b = np.random.randint(50, 130, n)
            scan_mask = np.random.random(n) < 0.3
            if scan_mask.any():
                pattern = np.sin(np.arange(n) * 0.5) > 0
                base_r = np.where(scan_mask & pattern, np.minimum(base_r + 30, 255), base_r)
                base_g = np.where(scan_mask & pattern, np.minimum(base_g + 30, 255), base_g)
                base_b = np.where(scan_mask & ~pattern, np.maximum(base_b - 40, 0), base_b)
            r, g, b = base_r, base_g, base_b

        else:   # Остальные классы – серый
            r[:] = 169
            g[:] = 169
            b[:] = 169

        return r, g, b