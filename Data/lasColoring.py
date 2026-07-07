import numpy as np

class ObjColloring:
    def __init__(self, las, mask):
        """
        las: полный LAS-объект
        mask: булев массив для выбранных точек (например, конкретного класса)
        """
        self.mask = mask
        # Классификация только для отобранных точек
        self.classification = np.asarray(las.classification[mask])
        # Цвета инициализируем нулями нужного размера
        n = mask.sum()
        self.r = np.zeros(n, dtype=np.uint8)
        self.g = np.zeros(n, dtype=np.uint8)
        self.b = np.zeros(n, dtype=np.uint8)

    def colloring(self):
        # Базовые натуральные цвета по ASPRS
        class_colors = {
            0: (128, 128, 128),    # Unclassified
            1: (169, 169, 169),
            2: (139, 90, 43),      # Ground – коричневый
            3: (34, 139, 34),      # Low vegetation – лесной зелёный
            4: (0, 128, 0),        # Medium vegetation – зелёный
            5: (0, 100, 0),        # High vegetation – тёмно-зелёный
            6: (178, 34, 34),      # Building
            7: None,               # Noise – обработаем отдельно
            8: (255, 255, 0),
            9: (0, 0, 255),
            10: (128, 0, 128),
            11: (105, 105, 105),
            12: None,              # Overlap – отдельно
            13: (0, 255, 255),
            14: (0, 255, 255),
            15: (255, 0, 255),
            16: (255, 192, 203),
            17: (128, 128, 128),
            18: (255, 0, 0),
        }

        # Сначала задаём базовые цвета для всех классов, кроме 7 и 12
        for cls_id, (cr, cg, cb) in class_colors.items():
            if cls_id in (7, 12):  # особые классы
                continue
            mask_cls = (self.classification == cls_id)
            if mask_cls.any():
                self.r[mask_cls] = cr
                self.g[mask_cls] = cg
                self.b[mask_cls] = cb

        # === Вариации для растительности (классы 3,4,5) ===
        veg_mask = np.isin(self.classification, [3, 4, 5])
        if veg_mask.any():
            # Зелёный канал: вариация ±20
            g_var = np.random.randint(-20, 21, size=veg_mask.sum())
            self.g[veg_mask] = np.clip(
                self.g[veg_mask].astype(np.int16) + g_var, 0, 255
            ).astype(np.uint8)
            # Красный канал: немного коричневого для стволов (±15)
            r_var = np.random.randint(-10, 16, size=veg_mask.sum())
            self.r[veg_mask] = np.clip(
                self.r[veg_mask].astype(np.int16) + r_var, 0, 255
            ).astype(np.uint8)

        # === Вариации для земли (класс 2) ===
        ground_mask = (self.classification == 2)
        if ground_mask.any():
            noise = np.random.randint(-15, 16, size=ground_mask.sum())
            for ch in (self.r, self.g, self.b):
                ch[ground_mask] = np.clip(
                    ch[ground_mask].astype(np.int16) + noise, 0, 255
                ).astype(np.uint8)

        # === Класс 7: Low Point (Noise) – равномерно RGB ===
        noise_mask = (self.classification == 7)
        if noise_mask.any():
            n = noise_mask.sum()
            third = n // 3
            # Первая треть – красный
            self.r[noise_mask] = np.concatenate([
                np.random.randint(220, 255, third),
                np.random.randint(0, 20, n - third)
            ])[:n]
            self.g[noise_mask] = np.concatenate([
                np.random.randint(0, 20, third),
                np.random.randint(220, 255, third),
                np.random.randint(0, 20, n - 2*third)
            ])[:n]
            self.b[noise_mask] = np.concatenate([
                np.random.randint(0, 20, 2*third),
                np.random.randint(220, 255, n - 2*third)
            ])[:n]
            # Перемешиваем, чтобы RGB-шум был случайным
            perm = np.random.permutation(n)
            self.r[noise_mask] = self.r[noise_mask][perm]
            self.g[noise_mask] = self.g[noise_mask][perm]
            self.b[noise_mask] = self.b[noise_mask][perm]

        # === Класс 12: Overlap – оранжевый с узором ===
        overlap_mask = (self.classification == 12)
        if overlap_mask.any():
            n = overlap_mask.sum()
            # Базовый оранжевый
            self.r[overlap_mask] = np.random.randint(200, 255, n)
            self.g[overlap_mask] = np.random.randint(150, 210, n)
            self.b[overlap_mask] = np.random.randint(50, 130, n)
            # Эффект "строк сканирования" (30%)
            scan_mask = np.random.random(n) < 0.3
            if scan_mask.any():
                scan_idx = np.where(overlap_mask)[0][scan_mask]
                pattern = np.sin(np.arange(len(scan_idx)) * 0.5) > 0
                self.r[scan_idx[pattern]] = np.minimum(self.r[scan_idx[pattern]] + 30, 255)
                self.g[scan_idx[pattern]] = np.minimum(self.g[scan_idx[pattern]] + 30, 255)
                self.b[scan_idx[~pattern]] = np.maximum(self.b[scan_idx[~pattern]] - 40, 0)

        return self.r, self.g, self.b
    
def color_instance(x, y, z, class_id):
    """
    Раскрашивает точки одного экземпляра в зависимости от класса и высоты.
    Возвращает массивы r, g, b (uint8).
    """
    n = len(x)
    r = np.zeros(n, dtype=np.uint8)
    g = np.zeros(n, dtype=np.uint8)
    b = np.zeros(n, dtype=np.uint8)

    # Нормализуем высоту внутри экземпляра (0 = низ, 1 = верх)
    if n > 1 and z.max() > z.min():
        z_norm = (z - z.min()) / (z.max() - z.min())
    else:
        z_norm = np.zeros(n)

    if class_id in (3, 4, 5):   # Растительность: градиент от коричневого (ствол) к зелёному (крона)
        # Ствол: коричневый (139,90,43) внизу
        # Крона: зелёный (0,128,0) для средней, (0,100,0) для высокой, (34,139,34) для низкой
        if class_id == 3:    # Low vegetation (кусты)
            top_color = np.array([34, 139, 34])   # лесной зелёный
        elif class_id == 4:  # Medium vegetation
            top_color = np.array([0, 128, 0])     # зелёный
        else:                # High vegetation (деревья)
            top_color = np.array([0, 100, 0])     # тёмно-зелёный

        bottom_color = np.array([139, 90, 43])     # коричневый

        # Интерполяция между низом и верхом по z_norm
        for i in range(3):
            channel = (bottom_color[i] * (1 - z_norm) + top_color[i] * z_norm).astype(np.uint8)
            if i == 0:
                r = channel
            elif i == 1:
                g = channel
            else:
                b = channel

        # Небольшой шум для реалистичности
        r = np.clip(r + np.random.randint(-10, 11, n), 0, 255).astype(np.uint8)
        g = np.clip(g + np.random.randint(-15, 16, n), 0, 255).astype(np.uint8)
        b = np.clip(b + np.random.randint(-10, 11, n), 0, 255).astype(np.uint8)

    elif class_id == 2:   # Земля – коричневый с вариациями
        base = np.array([139, 90, 43])
        r = np.full(n, base[0], dtype=np.uint8)
        g = np.full(n, base[1], dtype=np.uint8)
        b = np.full(n, base[2], dtype=np.uint8)
        noise = np.random.randint(-15, 16, n)
        r = np.clip(r.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif class_id == 7:   # Шум – RGB равномерно
        third = n // 3
        # Создаем три блока: красный, зелёный, синий
        colors = np.zeros((n, 3), dtype=np.uint8)
        # Красный
        colors[:third, 0] = np.random.randint(220, 255, third)
        colors[:third, 1] = np.random.randint(0, 20, third)
        colors[:third, 2] = np.random.randint(0, 20, third)
        # Зелёный
        colors[third:2*third, 0] = np.random.randint(0, 20, third)
        colors[third:2*third, 1] = np.random.randint(220, 255, third)
        colors[third:2*third, 2] = np.random.randint(0, 20, third)
        # Синий
        colors[2*third:, 0] = np.random.randint(0, 20, n - 2*third)
        colors[2*third:, 1] = np.random.randint(0, 20, n - 2*third)
        colors[2*third:, 2] = np.random.randint(220, 255, n - 2*third)
        # Перемешиваем
        perm = np.random.permutation(n)
        r = colors[perm, 0]
        g = colors[perm, 1]
        b = colors[perm, 2]

    elif class_id == 12:  # Перекрытие – оранжевый с узором сканирования
        base_r = np.random.randint(200, 255, n)
        base_g = np.random.randint(150, 210, n)
        base_b = np.random.randint(50, 130, n)
        # Добавляем полосы на 30% точек
        scan_mask = np.random.random(n) < 0.3
        if scan_mask.any():
            # Используем sin для создания полос
            pattern = np.sin(np.arange(n) * 0.5) > 0
            base_r = np.where(scan_mask & pattern, np.minimum(base_r + 30, 255), base_r)
            base_g = np.where(scan_mask & pattern, np.minimum(base_g + 30, 255), base_g)
            base_b = np.where(scan_mask & ~pattern, np.maximum(base_b - 40, 0), base_b)
        r, g, b = base_r, base_g, base_b

    else:   # Остальные классы – плоский серый/цвет по умолчанию
        r[:] = 169
        g[:] = 169
        b[:] = 169

    return r, g, b