import os
import torch
from enum import Enum

DRIVE_BASE_DIR = '/content/drive'
PROJECT_DIR = 'real_time_da_semseg'
DATA_DIR = f'{PROJECT_DIR}/loveDA_dataset'
TRAIN_ZIP = f'{DATA_DIR}/train.zip'
VAL_ZIP = f'{DATA_DIR}/validation.zip'
TEST_ZIP = f'{DATA_DIR}/test.zip'
TRAIN_DIR = f'{DATA_DIR}/train'
VAL_DIR = f'{DATA_DIR}/validation'
TEST_DIR = f'{DATA_DIR}/test'
RURAL_PATH = "Rural"
URBAN_PATH = "Urban"
IMG_PATH = "images_png"
MASK_PATH = "masks_png"
PRETRAINED_WEIGHTS_DIR = f'{DRIVE_BASE_DIR}/MyDrive/{PROJECT_DIR}/pretrained_weights'
DEEPLAB_V2_CHP_DIR = f"{DRIVE_BASE_DIR}/MyDrive/{PROJECT_DIR}/checkpoints/deeplab_v2/"
DEEPLAB_V2_WEIGHTS = f'{PRETRAINED_WEIGHTS_DIR}/DeepLab_resnet_pretrained_imagenet.pth'
DEEPLAB_V2_ID = "1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v"
PIDNET_S_WEIGHTS = f"{PRETRAINED_WEIGHTS_DIR}/pidnet_s_imagenet_pretrained.pth"
PIDNET_S_ID = "1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-"
STDC1_WEIGHTS = f"{PRETRAINED_WEIGHTS_DIR}/STDC1_pretrained_weights.pth"
STDC1_ID = "1DFoXcV42zy-apUcMh5P8WhsXMRJofgl8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SemanticLoss(str, Enum):
    CE = "cross_entropy"
    OHEM = "ohem"
    FOCAL = "focal"
    
    def __str__(self):
        return self.value
    
    def values():
        return [SemanticLoss.CE, SemanticLoss.OHEM, SemanticLoss.FOCAL]

class Domain(str, Enum):
    RURAL = "rural"
    URBAN = "urban"
    
    def __str__(self):
        return self.value
    
    def values():
        return [Domain.RURAL, Domain.URBAN]

class ModelType(str, Enum):
    DEEPLAB_V2 = "deeplab_v2"
    PIDNET_S = "pidnet_s"
    PIDNET_M = "pidnet_m"
    PIDNET_L = "pidnet_l"
    BISENET_V1 = "bisenet_v1"
    BISENET_V1_RT = "bisenet_v1_rt"
    STDC1 = "stdc1"
    STDC2 = "stdc2"
    
    def __str__(self):
        return self.value
    
    def values():
        return [
            ModelType.DEEPLAB_V2, ModelType.PIDNET_S, ModelType.PIDNET_M, ModelType.PIDNET_L,
            ModelType.BISENET_V1, ModelType.BISENET_V1_RT, ModelType.STDC1, ModelType.STDC2
        ]
        
class AdaptationMethod(str, Enum):
    ADDA = "adda"
    DACS = "dacs"
    
    def __str__(self):
        return self.value
    
    def values():
        return [AdaptationMethod.ADDA, AdaptationMethod.DACS]

categories = {
    'BARREN': (0.003921568859368563, (159, 129, 183)),       # Lilla
    'AGRICULTURE': (0.027450980618596077, (255, 195, 128)),  # Arancione
    'BUILDING': (0.007843137718737125, (255, 0, 0)),         # Rosso
    'WATER': (0.01568627543747425, (0, 0, 255)),             # Blu
    'ROAD': (0.0117647061124444, (255, 255, 0)),             # Giallo
    'BG': (0.019607843831181526, (255, 255, 255)),           # Bianco
    'FOREST': (0.0235294122248888, (0, 255, 0))              # Verde
}

categories = dict(sorted(categories.items(), key=lambda item: item[1][0]))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

urban_percentage = [48.5, 21.2, 9.3, 3.7, 7.6, 7.9, 1.9]
rural_percentage = [42.9, 3.7, 2.6, 11.6, 3.6, 5.0, 30.5]