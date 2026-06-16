import laspy
import numpy as np
import re
import os
from pathlib import Path

class lasConverter:

    def __init__(self, Areas):

        self.folder_paths = [Path(f"Data/S3DIS/{Area}") for Area in Areas]

    def toTxt(self):
        for folder_path in self.folder_paths:
            if folder_path.exists():
                pattern = re.compile(r'([^\\/]+)(?=\.las$)')

                for file in os.listdir(folder_path):
                    match = pattern.search(file)
                    if match:
                        las = laspy.read(folder_path.joinpath(file))

                        file_name = match.group(1)

                        x = las.x
                        y = las.y
                        z = las.z

                        r = (las.red // 256).astype(np.uint8)
                        g = (las.green // 256).astype(np.uint8)
                        b = (las.blue // 256).astype(np.uint8)

                        intensity = las.intensity.astype(np.float64)
                        return_num = las.return_number.astype(np.float64)
                        num_returns = las.number_of_returns.astype(np.float64)
                        classification = las.classification.astype(np.float64)

                        data = np.column_stack((x, y, z, r, g, b, intensity, return_num, num_returns, classification))

                        # TXT с соответствием формату (пробел-разделитель и точность)
                        # %f задает вывод с плавающей точкой (6 знаков), %d или %.0f для целых.
                        # Чтобы строго повторить ваш шаблон, используем список форматов для каждой колонки:
                        fmt_spec = ["%.8f", "%.8f", "%.8f", "%d", "%d", "%d", "%f", "%f", "%f", "%f"]

                        np.savetxt(folder_path.joinpath(f"{file_name}.txt"), data, fmt=fmt_spec, delimiter=" ")

                        print(f"{file_name}.txt saved")