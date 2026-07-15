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
        self.Areas = [f'Area_{i}' for i in range(1, 25)]
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
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        x_edges = np.linspace(x_min, x_max, cols + 1)
        y_edges = np.linspace(y_min, y_max, rows + 1)
        areas = []
        for i in range(rows):
            for j in range(cols):
                if len(areas) >= n:
                    break
                areas.append((x_edges[j], x_edges[j+1], y_edges[i], y_edges[i+1]))
        return areas

    
    def one_to_many_by_classes(self, region_name: str = 'forest_1'):
        # Удаляем старые Annotations, оставшиеся не в комнатах
        for area in self.Areas:
            bad_annot = self.base_path / area / 'Annotations'
            if bad_annot.is_dir():
                import shutil
                shutil.rmtree(bad_annot)
                print(f'Removed leftover {bad_annot}')

        with laspy.open(str(self.las_file_name)) as reader:
            # Надёжное вычисление границ по всем точкам
            print('Computing real boundaries from points...')
            x_min = y_min = float('inf')
            x_max = y_max = float('-inf')

            las = reader.read()
            print("las classifications:", np.unique(las.classification))
            del las

            for chunk in reader.chunk_iterator(2_000_000):
                x_min = min(x_min, chunk.x.min())
                x_max = max(x_max, chunk.x.max())
                y_min = min(y_min, chunk.y.min())
                y_max = max(y_max, chunk.y.max())
            print(f'Bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]')
            reader.seek(0)  # вернуться в начало

            dim_names = [dim.name.lower() for dim in reader.header.point_format.dimensions]
            has_color = 'red' in dim_names and 'green' in dim_names and 'blue' in dim_names
            print('Has colors:', has_color)

            bboxes = self._split_bbox(x_min, x_max, y_min, y_max)

            print('Zones')
            print(f'  {self.Areas}')
            print('Zone boundaries:')
            for i, bbox in enumerate(bboxes):
                print(f'  {self.Areas[i]}: {bbox}')

            fmt_spec = ["%.8f", "%.8f", "%.8f", "%d", "%d", "%d", "%f", "%f", "%f", "%f"]

            for area_idx, (x0, x1, y0, y1) in enumerate(bboxes):
                area_name = self.Areas[area_idx]
                annot_dir = self.base_path / area_name / region_name / 'Annotations'
                annot_dir.mkdir(parents=True, exist_ok=True)
                print(f'Processing {area_name}/{region_name}...')

                reader.seek(0)  # начать чтение файла заново для этой зоны
                zx_list, zy_list, zz_list, zcls_list, zi_list, zrn_list, znr_list = [], [], [], [], [], [], []
                if has_color:
                    zr_list, zg_list, zb_list = [], [], []

                for chunk in reader.chunk_iterator(2_000_000):
                    mask = (chunk.x >= x0) & (chunk.x < x1) & (chunk.y >= y0) & (chunk.y < y1)

                    if mask.sum() == 0:
                        continue
                    zx_list.append(np.asarray(chunk.x[mask]))
                    zy_list.append(np.asarray(chunk.y[mask]))
                    zz_list.append(np.asarray(chunk.z[mask]))
                    zcls_list.append(np.asarray(chunk.classification[mask]))
                    zi_list.append(np.asarray(chunk.intensity[mask], dtype=np.float32))
                    zrn_list.append(np.asarray(chunk.return_number[mask], dtype=np.float32))
                    znr_list.append(np.asarray(chunk.number_of_returns[mask], dtype=np.float32))
                    if has_color:
                        zr_list.append(np.asarray(chunk.red[mask], dtype=np.uint8))
                        zg_list.append(np.asarray(chunk.green[mask], dtype=np.uint8))
                        zb_list.append(np.asarray(chunk.blue[mask], dtype=np.uint8))

                if area_name in os.listdir(self.S3DIS_path):
                        if len(os.listdir(self.S3DIS_path.joinpath(f"{area_name}/forest_1/Annotations")))>0:

                            print(f'  {area_name}, already exists')
                            for cls_id, cls_name in self.class_names.items():
                                #zcls = np.concatenate(zcls_list)
                                cmask = (zcls_list == cls_id)
                                print(f'  {cls_name}: {len(instances)} instances')
                            continue

                if not zx_list:
                    print(f'{area_name}/{region_name}: no points, skipping.')
                    continue

                zx = np.concatenate(zx_list)
                zy = np.concatenate(zy_list)
                zz = np.concatenate(zz_list)
                zcls = np.concatenate(zcls_list)
                zi = np.concatenate(zi_list)
                zrn = np.concatenate(zrn_list)
                znr = np.concatenate(znr_list)
                if has_color:
                    zr = np.concatenate(zr_list)
                    zg = np.concatenate(zg_list)
                    zb = np.concatenate(zb_list)
                else:
                    zr = np.zeros_like(zx, dtype=np.uint8)
                    zg = np.zeros_like(zx, dtype=np.uint8)
                    zb = np.zeros_like(zx, dtype=np.uint8)

                print(f'{area_name}/{region_name}: {len(zx)} points in zone')

                for cls_id, cls_name in self.class_names.items():
                    cmask = (zcls == cls_id)
                    if cmask.sum() == 0:
                        continue
                    cx, cy, cz = zx[cmask], zy[cmask], zz[cmask]
                    cr, cg, cb = zr[cmask], zg[cmask], zb[cmask]
                    ci, crn, cnr = zi[cmask], zrn[cmask], znr[cmask]

                    cls_data = np.column_stack((
                        cx, cy, cz, cr, cg, cb, ci, crn, cnr,
                        np.full(cx.shape, cls_id, dtype=np.float32)
                    ))

                    if cls_id == 2:
                        instances = [cls_data]
                        print(f'  {cls_name}: 1 instance')
                    else:
                        coords = np.column_stack((cx, cy, cz))
                        clustering = DBSCAN(eps=0.3, min_samples=10).fit(coords)
                        labels = clustering.labels_
                        instances = []
                        for lbl in np.unique(labels):
                            if lbl == -1:
                                continue
                            instances.append(cls_data[labels == lbl])
                        print(f'  {cls_name}: {len(instances)} instances')

                    for inst_id, inst_pts in enumerate(instances, start=1):
                        if not has_color:
                            inst_x, inst_y, inst_z = inst_pts[:, 0], inst_pts[:, 1], inst_pts[:, 2]
                            new_r, new_g, new_b = self.color_instance(inst_x, inst_y, inst_z, cls_id)
                            inst_pts[:, 3] = new_r
                            inst_pts[:, 4] = new_g
                            inst_pts[:, 5] = new_b
                        fname = annot_dir / f'{cls_name}_{inst_id}.txt'
                        np.savetxt(fname, inst_pts, fmt=fmt_spec, delimiter=' ')

                    del cls_data, instances, cmask

                del zx, zy, zz, zcls, zi, zrn, znr, zr, zg, zb
                print(f'{area_name}/{region_name} done.')
    
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