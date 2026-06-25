import laspy
import numpy as np
from pathlib import Path
import re
import os


class LASOrganizer:
    def __init__(self, base_path='Data/S3DIS', Areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']):
        self.base_path = Path(base_path)
        self.folder_paths = [Path(f"Data/S3DIS/{Area}") for Area in Areas]

    def file_finding(self):
        for folder_path in self.folder_paths:
            if folder_path.exists():
                laSpattern = re.compile(r'([^\\/]+)(?=\.las$)')
                laZpattern = re.compile(r'([^\\/]+)(?=\.laz$)')

                for file in os.listdir(folder_path):
                    match = laSpattern.search(file) or laZpattern.search(file)
                    if match:
                        las = laspy.read(folder_path.joinpath(file))
                        
                        return las
                    
                        file_name = match.group(1)
                        np.savetxt(folder_path.joinpath(f"{file_name}.txt"), data, fmt=fmt_spec, delimiter=" ")

                        print(f"{file_name}.txt saved")

        
    
    
    def one_las_to_cls_num_txt_files(self, 
                                   one_las: str, 
                                   area_name: str = 'Area_1',
                                   region_name: str = 'forest_1'):
        one_las = self.file_finding()
        las = laspy.read(one_las)

        data = {}

        for cls in np.unique(las.classification):
            mask = cls==las.classification

            x = las.x[mask]
            y = las.y[mask]
            z = las.z[mask]

            r = las.red.astype(np.uint8)[mask]
            g = las.green.astype(np.uint8)[mask]
            b = las.blue.astype(np.uint8)[mask]

            intensity = las.intensity.astype(np.float64)[mask]
            return_num = las.return_number.astype(np.float64)[mask]
            num_returns = las.number_of_returns.astype(np.float64)[mask]
            classification = las.classification.astype(np.float64)[mask]

            data[cls] = np.column_stack((x, y, z, r, g, b, intensity, return_num, num_returns, classification))

        return data, np.unique(las.classification)