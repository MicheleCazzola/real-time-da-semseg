from enum import Enum

class SemanticLoss(str, Enum):
    CE = "cross_entropy"
    OHEM = "ohem"
    FOCAL = "focal"
    
    def __str__(self):
        return self.value
    
    def values():
        return [SemanticLoss.CE.value, SemanticLoss.OHEM.value, SemanticLoss.FOCAL.value]

class Domain(str, Enum):
    RURAL = "rural"
    URBAN = "urban"
    
    def __str__(self):
        return self.value
    
    def values():
        return [Domain.RURAL.value, Domain.URBAN.value]

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
            ModelType.DEEPLAB_V2.value, ModelType.PIDNET_S.value, ModelType.PIDNET_M.value, ModelType.PIDNET_L.value,
            ModelType.BISENET_V1.value, ModelType.BISENET_V1_RT.value, ModelType.STDC1.value, ModelType.STDC2.value
        ]
        
class AdaptationMethod(str, Enum):
    ADDA = "adda"
    ADDA_MULTI = "adda_multi"
    DACS = "dacs"
    IAST = "iast"
    
    def __str__(self):
        return self.value
    
    def values():
        return [AdaptationMethod.ADDA.value, AdaptationMethod.ADDA_MULTI.value, AdaptationMethod.DACS.value, AdaptationMethod.IAST.value]