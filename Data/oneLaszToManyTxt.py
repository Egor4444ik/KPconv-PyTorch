import laspy
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import re
import os
import math


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

    
    def one_to_many_by_classes (self, region_name: str = 'forest_1'):
        
        data = {}

        unique_classes = np.unique(self.las.classification)
        for cls in unique_classes:
            print('Unique classes:', unique_classes)

            mask = cls==self.las.classification

            x =self.las.x[mask]
            y =self.las.y[mask]
            z =self.las.z[mask]

            las_attrs = self.las.__dict__
            print('las file attributes:', las_attrs)
            has_color = hasattr(self.las, 'red')
            if has_color:
                print('Las object is collored')
                r = np.asarray(self.las.red[mask]).astype(np.uint8)
                g = np.asarray(self.las.green[mask]).astype(np.uint8)
                b = np.asarray(self.las.blue[mask]).astype(np.uint8)
            else:
                print('Las object is not collored')

            intensity = np.asarray(self.las.intensity[mask]).astype(np.float64)
            return_num = np.asarray(self.las.return_number[mask]).astype(np.float64)
            num_returns = np.asarray(self.las.number_of_returns[mask]).astype(np.float64)
            classification = np.asarray(self.las.classification[mask]).astype(np.float64)

            data = np.column_stack((x, y, z, r, g, b, intensity, return_num, num_returns, classification))
            
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            bboxes = self._split_bbox(x_min, x_max, y_min, y_max)

        fmt_spec = ["%.8f", "%.8f", "%.8f", "%d", "%d", "%d", "%f", "%f", "%f", "%f"]

        # Обрабатываем каждую зону
        for area_idx, (x0, x1, y0, y1) in enumerate(bboxes):
            area_name = self.Areas[area_idx]   # берём имя из списка
            # Создаём папку Annotations внутри зоны (как в S3DIS)
            annot_dir = self.base_path / area_name / 'Annotations'
            annot_dir.mkdir(parents=True, exist_ok=True)

            # Маска точек в зоне
            zone_mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            if zone_mask.sum() < 100:
                print(f'{area_name}: слишком мало точек, пропускаем.')
                continue

            zx, zy, zz = x[zone_mask], y[zone_mask], z[zone_mask]
            zr, zg, zb = r[zone_mask], g[zone_mask], b[zone_mask]
            zi = intensity[zone_mask]
            zrn = return_num[zone_mask]
            znr = num_returns[zone_mask]
            zcls = classification[zone_mask]

            # Для каждого класса внутри зоны
            for cls_id, cls_name in self.class_names.items():
                cmask = (zcls == cls_id)
                if cmask.sum() == 0:
                    continue
                cx, cy, cz = zx[cmask], zy[cmask], zz[cmask]
                cr, cg, cb = zr[cmask], zg[cmask], zb[cmask]
                ci = zi[cmask]
                crn = zrn[cmask]
                cnr = znr[cmask]
                cls_data = np.column_stack((cx, cy, cz, cr, cg, cb, ci, crn, cnr,
                                            np.full(cx.shape, cls_id, dtype=np.float64)))

                # Разделение на экземпляры
                if cls_id == 2:          # Ground – один объект
                    instances = [cls_data]
                    print(f'{area_name}/{cls_name}: 1 (Ground) instance')
                else:                    # растительность – кластеризуем
                    coords = np.column_stack((cx, cy, cz))
                    clustering = DBSCAN(eps=0.3, min_samples=10).fit(coords)
                    labels = clustering.labels_
                    instances = []
                    for lbl in np.unique(labels):
                        if lbl == -1:    # шум пропускаем
                            continue
                        instances.append(cls_data[labels == lbl])
                    print(f'{area_name}/{cls_name}: {len(instances)} instances founded')

                # Сохраняем каждый экземпляр с нужным именем
                for inst_id, inst_pts in enumerate(instances, start=1):
                    fname = annot_dir / f'{cls_name}_{inst_id}.txt'
                    np.savetxt(fname, inst_pts, fmt=fmt_spec, delimiter=' ')
                    print(f'  {inst_pts.shape[0]} points saved to {fname}')

            print('clustering done.')
            
            '''fmt_spec = ["%.8f", "%.8f", "%.8f", "%d", "%d", "%d", "%f", "%f", "%f", "%f"]

            cls_name = self.class_names.get(cls, f'class_{cls}')

            #DBSCAN processing
            coords = np.column_stack((x, y, z))

            clustering = DBSCAN(eps=0.3, min_samples=10).fit(coords)
            labels = clustering.labels_

            instance_list = []
            instance_ids = []
            for lbl in np.unique(labels):
                if lbl == -1:    # noise continue
                    continue
                instance_list.append(data[labels == lbl])
                instance_ids.append(lbl + 1)   # nummering labels after one
            print(f'  Found {len(instance_list)} instances for class {cls_name}')

            max_instance_id = max(instance_ids)
            areas_count = len(self.Areas)
            for inst_id, inst_data in zip(instance_ids, instance_list):
                    area_num = np.clip(int((inst_id/max_instance_id)*areas_count)+1, 1, areas_count)
                    output_dir = Path(f"Data/S3DIS/Area_{area_num}"/region_name)
                    fname = output_dir / f'{cls_name}_{inst_id}.txt'
                    np.savetxt(fname, inst_data, fmt=fmt_spec, delimiter=" ")
                    print(f'  Saved {fname} ({inst_data.shape[0]} points)')

            np.savetxt(self.folder_paths.joinpath(f"{cls}_.txt"), data, fmt=fmt_spec, delimiter=" ")

            print(f"{cls}.txt saved")
        return data, np.unique(self.las.classification)
    '''