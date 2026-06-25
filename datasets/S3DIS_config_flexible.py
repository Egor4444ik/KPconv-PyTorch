"""
Гибкая конфигурация S3DIS для работы с неполным датасетом
(когда есть только несколько Areas вместо полных 6)
"""

# Используйте эту конфигурацию вместо стандартной в train_S3DIS.py

from utils.config import Config
import numpy as np

class S3DISConfigFlexible(Config):
    """
    Гибкая конфигурация S3DIS, позволяющая работать с любым количеством Areas.
    
    Использование:
        config = S3DISConfigFlexible(available_areas=['Area_1', 'Area_2'], 
                                     validation_areas=['Area_2'])
    """
    
    ####################
    # Dataset parameters
    ####################
    
    dataset = 'S3DIS'
    num_classes = None
    dataset_task = ''
    input_threads = 4
    
    #########################
    # Architecture definition
    #########################
    
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']
    
    ###################
    # KPConv parameters
    ###################
    
    num_kernel_points = 15
    in_radius = 1.2
    first_subsampling_dl = 0.03
    conv_radius = 2.5
    deform_radius = 5.0
    KP_extent = 1.2
    KP_influence = 'linear'
    aggregation_mode = 'sum'
    first_features_dim = 128
    in_features_dim = 5
    modulated = False
    use_batch_norm = True
    batch_norm_momentum = 0.02
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0
    deform_lr_factor = 0.1
    repulse_extent = 1.2
    
    #####################
    # Training parameters
    #####################
    
    max_epoch = 500
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0
    batch_num = 6
    epoch_steps = 500
    validation_size = 50
    checkpoint_gap = 50
    
    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_color = 0.8
    
    segloss_balance = 'none'
    saving = True
    saving_path = None
    
    # Дополнительные параметры
    class_w = []
    deform_layers = None
    num_layers = None
    
    def __init__(self, available_areas=None, validation_areas=None):
        """
        Инициализация гибкой конфигурации.
        
        Args:
            available_areas: список доступных Areas (например, ['Area_1', 'Area_2'])
            validation_areas: список Areas для валидации
        
        Example:
            config = S3DISConfigFlexible(
                available_areas=['Area_1', 'Area_2', 'Area_3'],
                validation_areas=['Area_3']
            )
        """
        super(S3DISConfigFlexible, self).__init__()
        
        if available_areas is None:
            # По умолчанию использовать все 6 Areas
            available_areas = [f'Area_{i}' for i in range(1, 7)]
        
        if validation_areas is None:
            # По умолчанию использовать последнюю Area для валидации
            validation_areas = [available_areas[-1]] if available_areas else []
        
        self.available_areas = available_areas
        self.validation_areas = validation_areas
        
        # Класс метки: 0=ground, 1=bush, 2=tree (для лесных данных)
        self.num_classes = 3

print("""
═══════════════════════════════════════════════════════════════
📋 S3DIS CONFIG FLEXIBLE - Инструкция
═══════════════════════════════════════════════════════════════

Использование в train_S3DIS.py:
─────────────────────────────────

1. Измените import:
   from datasets.S3DIS import S3DISConfig
   на:
   from datasets.S3DIS_config import S3DISConfigFlexible as S3DISConfig

2. Модифицируйте инициализацию:
   
   config = S3DISConfig()
   
   на:
   
   config = S3DISConfig(
       available_areas=['Area_1', 'Area_2', 'Area_3'],
       validation_areas=['Area_3']
   )

3. Полная структура папок:
   
   Data/S3DIS/
   ├── Area_1/
   │   └── forest_1/
   │       └── Annotations/
   │           ├── ground_1.txt
   │           ├── bush_1.txt
   │           └── tree_1.txt
   ├── Area_2/
   │   └── forest_1/
   │       └── Annotations/
   │           ├── ground_1.txt
   │           ├── bush_1.txt
   │           └── tree_1.txt
   └── Area_3/
       └── forest_1/
           └── Annotations/
               ├── ground_1.txt
               ├── bush_1.txt
               └── tree_1.txt

═══════════════════════════════════════════════════════════════
""")
