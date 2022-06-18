import numpy as np
from sklearn.datasets import load_breast_cancer

from clasificadores import particion_entr_prueba
from clasificadores import RegresionLogisticaMiniBatch
from clasificadores import rendimiento_p2
from clasificadores import normaliza

from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

X_cancer,y_cancer=cancer.data, cancer.target

Xe_cancer, Xp_cancer, ye_cancer, yp_cancer = particion_entr_prueba(X_cancer,y_cancer)

lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)

lr_cancer.entrena(Xe_cancer, ye_cancer)

print("Test Regresion logistica sobre los datos del cancer.")
print("Rendimiento: ",rendimiento_p2(lr_cancer, normaliza(Xe_cancer), ye_cancer))
