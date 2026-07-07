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
        self.las = self.file_finding()


    def file_finding(self):

        #for folder_path in self.folder_paths:
            folder_path = self.S3DIS_path
            if folder_path.exists():
                laspattern = re.compile(r'([^\\/]+)(?=\.las$)')
                laZpattern = re.compile(r'([^\\/]+)(?=\.laz$)')

                for file in os.listdir(folder_path):
                    match = laspattern.search(file) or laZpattern.search(file)
                    if match:
                        print('las/laz file founded:', file)
                        las =laspy.read(folder_path.joinpath(file))
                        
                        return las
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
        # 1. Загружаем все данные один раз
        x_all = np.asarray(self.las.x)
        y_all = np.asarray(self.las.y)
        z_all = np.asarray(self.las.z)
        classification_all = np.asarray(self.las.classification)
        intensity_all = np.asarray(self.las.intensity, dtype=np.float64)
        return_num_all = np.asarray(self.las.return_number, dtype=np.float64)
        num_returns_all = np.asarray(self.las.number_of_returns, dtype=np.float64)

        # Проверяем наличие цвета
        has_color = hasattr(self.las, 'red')
        if has_color:
            r_all = np.asarray(self.las.red, dtype=np.uint8)
            g_all = np.asarray(self.las.green, dtype=np.uint8)
            b_all = np.asarray(self.las.blue, dtype=np.uint8)
            print('Las object has real colors.')
        else:
            # Создаём нулевые массивы, реальный цвет будет назначен позже для каждого экземпляра
            r_all = np.zeros_like(x_all, dtype=np.uint8)
            g_all = np.zeros_like(x_all, dtype=np.uint8)
            b_all = np.zeros_like(x_all, dtype=np.uint8)
            print('Las object is not colored. Procedural coloring will be applied after clustering.')

        # 2. Границы облака для разбиения на зоны
        x_min, x_max = x_all.min(), x_all.max()
        y_min, y_max = y_all.min(), y_all.max()
        bboxes = self._split_bbox(x_min, x_max, y_min, y_max)

        fmt_spec = ["%.8f", "%.8f", "%.8f", "%d", "%d", "%d", "%f", "%f", "%f", "%f"]

        # 3. Обработка каждой зоны
        for area_idx, (x0, x1, y0, y1) in enumerate(bboxes):
            area_name = self.Areas[area_idx]
            annot_dir = self.base_path / area_name / 'Annotations'
            annot_dir.mkdir(parents=True, exist_ok=True)

            # Маска зоны
            zone_mask = (x_all >= x0) & (x_all < x1) & (y_all >= y0) & (y_all < y1)
            if zone_mask.sum() < 100:
                print(f'{area_name}: слишком мало точек, пропускаем.')
                continue

            # Вырезаем данные зоны
            zx = x_all[zone_mask]
            zy = y_all[zone_mask]
            zz = z_all[zone_mask]
            zr = r_all[zone_mask]
            zg = g_all[zone_mask]
            zb = b_all[zone_mask]
            zi = intensity_all[zone_mask]
            zrn = return_num_all[zone_mask]
            znr = num_returns_all[zone_mask]
            zcls = classification_all[zone_mask]

            # 4. Для каждого класса внутри зоны
            for cls_id, cls_name in self.class_names.items():
                cmask = (zcls == cls_id)
                if cmask.sum() == 0:
                    continue

                cx, cy, cz = zx[cmask], zy[cmask], zz[cmask]
                ci = zi[cmask]
                crn = zrn[cmask]
                cnr = znr[cmask]
                # Пока берём цвета из зоны (для настоящих цветов они уже есть, для процедурных пока нули)
                cr = zr[cmask]
                cg = zg[cmask]
                cb = zb[cmask]

                cls_data = np.column_stack((
                    cx, cy, cz, cr, cg, cb, ci, crn, cnr,
                    np.full(cx.shape, cls_id, dtype=np.float64)
                ))

                # 5. Разделение на экземпляры
                if cls_id == 2:  # Ground – один объект
                    instances = [cls_data]
                    print(f'{area_name}/{cls_name}: 1 (Ground) instance')
                else:            # растительность и прочее – кластеризуем
                    coords = np.column_stack((cx, cy, cz))
                    clustering = DBSCAN(eps=0.3, min_samples=10).fit(coords)
                    labels = clustering.labels_
                    instances = []
                    for lbl in np.unique(labels):
                        if lbl == -1:    # шум DBSCAN пропускаем
                            continue
                        instances.append(cls_data[labels == lbl])
                    print(f'{area_name}/{cls_name}: {len(instances)} instances found')

                # 6. Раскрашиваем каждый экземпляр (только если исходный файл без цвета)
                for inst_id, inst_pts in enumerate(instances, start=1):
                    if not has_color:
                        # Вызываем функцию раскраски для этого экземпляра
                        inst_x = inst_pts[:, 0]
                        inst_y = inst_pts[:, 1]
                        inst_z = inst_pts[:, 2]
                        new_r, new_g, new_b = self.color_instance(inst_x, inst_y, inst_z, cls_id)
                        inst_pts[:, 3] = new_r
                        inst_pts[:, 4] = new_g
                        inst_pts[:, 5] = new_b

                    # Сохраняем
                    fname = annot_dir / f'{cls_name}_{inst_id}.txt'
                    np.savetxt(fname, inst_pts, fmt=fmt_spec, delimiter=' ')
                    print(f'  {inst_pts.shape[0]} points saved to {fname}')

            print(f'{area_name} processing done.')
    
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