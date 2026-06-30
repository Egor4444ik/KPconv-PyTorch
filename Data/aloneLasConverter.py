import laspy
import numpy as np
from pathlib import Path
import re
import os


class LASOrganizer:
    def __init__(self, base_path='Data/S3DIS', Areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']):
        self.base_path = Path(base_path)
        self.folder_paths = [Path(f"Data/S3DIS/{Area}") for Area in Areas]
        self.S3DIS_path = Path(f"Data/S3DIS")

    def file_finding(self):

        
        #for folder_path in self.folder_paths:
            folder_path = self.S3DIS_path
            if folder_path.exists():
                laSpattern = re.compile(r'([^\\/]+)(?=\.las$)')
                laZpattern = re.compile(r'([^\\/]+)(?=\.laz$)')

                for file in os.listdir(folder_path):
                    match = laSpattern.search(file) or laZpattern.search(file)
                    if match:
                        print('las/laz file fined:', file)
                        las = laspy.read(folder_path.joinpath(file))
                        
                        return las
                    
                        file_name = match.group(1)
                        np.savetxt(folder_path.joinpath(f"{file_name}.txt"), data, fmt=fmt_spec, delimiter=" ")

                        print(f"{file_name}.txt saved")
            else:
                print("folder_path doesnt exists")

        
    
    
    def one_las_to_cls_num_txt_files(self, 
                                   area_name: str = 'Area_1',
                                   region_name: str = 'forest_1'):
        
        las = self.file_finding()

        data = {}

        for cls in np.unique(las.classification):
            mask = cls==las.classification

            x = las.x[mask]
            y = las.y[mask]
            z = las.z[mask]

            r = np.asarray(las.red[mask]).astype(np.uint8)
            g = np.asarray(las.green[mask]).astype(np.uint8)
            b = np.asarray(las.blue[mask]).astype(np.uint8)

            intensity = np.asarray(las.intensity[mask]).astype(np.float64)
            return_num = np.asarray(las.return_number[mask]).astype(np.float64)
            num_returns = np.asarray(las.number_of_returns[mask]).astype(np.float64)
            classification = np.asarray(las.classification[mask]).astype(np.float64)

            data[cls] = np.column_stack((x, y, z, r, g, b, intensity, return_num, num_returns, classification))

        return data, np.unique(las.classification)