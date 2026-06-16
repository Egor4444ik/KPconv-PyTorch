#REQUIREMENTS INSTALL

*python(Windows)/python3(MacOs/Linux) 3.11 - 3.13 versions
*pip(Windows)/pip3(MacOs/Linux) 25 - 26 versions

*MacOs/Linux python3 enviropment and requirements install

```terminal
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

*Windows python enviropment and requirements install

```terminal
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

*Compilling c++ script (MS Visual C++ extention required)
sh compile_wrappers.sh

#Data preporation

Txt or las point cloud format are distributed in Data/S3DIS/Area_i

In which from their will be able train/val data (val can be alone path only)

Uses Areas defines in 105 subline of S3DIS.py module are changing:

```python
# Proportion of validation scenes
self.cloud_names = ['Area_1', 'Area_2']
lasConverter(self.cloud_names).toTxt()
self.all_splits = [0, 1]
self.validation_split = 1
```

This you setted two areas only and choosing second from their as val

#Running

```terminal
python3 train_S3DIS.py
```