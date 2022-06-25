# Inteligencia Artificial para la Ciencia de los Datos
# Implementación de clasificadores 
# Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================
import sys
import numpy as np
import argparse
import random
import math

# Importamos los datos que vamos a usar en los ejemplos posteriores.
from jugar_tenis import X_tenis
from jugar_tenis import y_tenis
from votos import datos as X_votos
from votos import clasif as y_votos
from credito import datos_con_la_clase
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
iris=load_iris()
X_iris=iris.data
y_iris=iris.target

# !{sys.executable} -m pip install numpy sklearn


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Lorenz Vieta
# NOMBRE: Germán
#
# Segundo componente (si se trata de un grupo):
#
# APELLIDOS: Sáez Guerra
# NOMBRE: Pilar
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo
# que debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite, pero NO AL
# NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. Si tienen
# dificultades para realizar el ejercicio, consulten con el profesor. En caso
# de detectarse plagio, supondrá una calificación de cero en la asignatura,
# para todos los alumnos involucrados. Sin perjuicio de las medidas
# disciplinarias que se pudieran tomar. 
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO se permite usar Scikit Learn, EXCEPTO algunos métodos 
#   que se indican exprésamente en el enunciado. En particular no se permite
#   usar ningún clasificador de Scikit Learn.  
# * Supondremos que los conjuntos de datos se presentan en forma de arrays numpy, 
#   Se valorará el uso eficinte de numpy. 


# ====================================================
# PARTE I: IMPLEMENTACIÓN DEL CLASIFICADOR NAIVE BAYES
# ====================================================

# Se pide implementar el clasificador Naive Bayes, en su versión categórica
# con suavizado y log probabilidades (descrito en el tema 2, diapositivas 22 a
# 34). En concreto:


# ----------------------------------
# I.1) Implementación de Naive Bayes
# ----------------------------------

# Definir una clase NaiveBayes con la siguiente estructura:

class NaiveBayes():

    def __init__(self,k=1):        
        self.k = k
        self.X = None
        self.y = None
        self.uniqueClas = None
        self.p_c = []
        self.list_p_condics = []
        self.probabilidades_c = {}
        
    def entrena(self,X,y):
        
        self.X = X
        self.y = y
    
        #countsClas nos da el nº de veces se da un valor de clasificación, uniqueClas los dos posibles valores de 
        #clasificación ('si' y 'no')
        uniqueClas, countsClas = np.unique(y, return_counts=True)
        self.uniqueClas = uniqueClas
    
        #Función de probabilidad P(C=c)
        p_c = []
        for i in range(len(uniqueClas)):
            p_c.append(countsClas[i]/len(y))
            
        self.p_c = p_c
        
        #Listado de probabilidades condicionales
        list_p_condics = []
        self.list_p_condics = list_p_condics
        
        for i in list(uniqueClas): #Por cada valor de clasificación ('si' o 'no'), vamos a añadir un registro en list_p_condics
            p_condics =[] #Matriz para el valor de clasificación i
            
            for j in range(X.shape[1]): #Por cada atributo, vamos a calcular sus probabilidades condicionadas. 
                                        #j es el nº de atributo 
        
                valuesAtribj, numberOfValuesAtribj = np.unique(X[:,j], return_counts=True)
                p_condics_j = [] #Por cada atributo, tenemos la fila de las probabilidades condicionadas
                
                for k in range(len(valuesAtribj)): #Por cada valor que puede tomar el atributo,
                                                   #vamos a calcular su probabilidad condicionada
                    
                    positionsOfk = np.where(X[:,j] == valuesAtribj[k]) #Posiciones en las que el atributo j vale k
                    #Formamos el vector y con las posiciones en las que el clasificador vale i y el atributo j vale k
                    y_k = []
                    for l in positionsOfk: #Por cada valor de posición de k, nos quedamos con las posiciones de y, formamos y_k
                        y_k.append(y[l])
                        
                    positionsOfk_i = np.where(y_k[0] == i) #Posiciones del valor de clasificación i en y
                    y_ki = []
                    for m in positionsOfk_i:
                        y_ki.append(y_k[0][m])
                        
                    #Nº de ejemplos clasificados como i en el atributo j con valor k    
                    n_i_j_k = len(y_ki[0])
                    
                    p_condics_j.append((n_i_j_k + self.k)/(countsClas[np.where(uniqueClas == i)][0] + self.k*len(valuesAtribj)))
                    
                p_condics.append(p_condics_j)
            
            list_p_condics.append(p_condics)

    def clasifica_prob(self,ejemplo):
        if (self.list_p_condics == []):
            raise ClasificadorNoEntrenado("Debe entrenar antes el clasificador.")
        else:
                
            #Por cada atributo de 'ejemplo', buscamos sus dos probabilidades condicionadas en la matriz construida
            probabilidades_c = {}
            
            for c in list(self.uniqueClas):
                posic_c = np.where(self.uniqueClas == c)[0][0]
                
                probs_condics = []
                
                for a in range(len(ejemplo)): #Por cada valor de los atributos, sacamos su probabilidad condicionada de la matriz construida
                    matriz_p_c = self.list_p_condics[posic_c] #Matriz para el valor de clasificación actual
                    valuesAtriba, numberOfValuesAtriba = np.unique(self.X[:,a], return_counts=True)
                    fila = a #Corresponde al nº de atributo
                    columna = np.where(valuesAtriba == ejemplo[a])[0][0] #Corresponde al valor del atributo
                    probs_condics.append(matriz_p_c[fila][columna])
                
                sum_logs = np.sum([math.log(x,10) for x in probs_condics])
                resultado = math.log(self.p_c[posic_c],10) + sum_logs
                probabilidades_c[c] = (resultado)
                probs_condics = []
                
            self.probabilidades_c = probabilidades_c
            return probabilidades_c

    def clasifica(self,ejemplo):
        if (self.list_p_condics == []):
            raise ClasificadorNoEntrenado("Debe entrenar antes el clasificador.")
        else:
            maximo = {}
            find_max = max(self.clasifica_prob(ejemplo), key=self.clasifica_prob(ejemplo).get)
            maximo[find_max] = self.clasifica_prob(ejemplo)[find_max]
        
        return maximo


# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1) 
# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los
#   datos y los valores de clasificación respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan.  
# * Método clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribución de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase. 
# * Método clasifica: recibe un ejemplo (en forma de array de numpy) y
#   devuelve la clase que el modelo predice para ese ejemplo.   

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

  
# Ejemplo "jugar al tenis":


# >>> nb_tenis=NaiveBayes(k=0.5)
# >>> nb_tenis.entrena(X_tenis,y_tenis)
# >>> ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
# >>> nb_tenis.clasifica_prob(ej_tenis)
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
# >>> nb_tenis.clasifica(ej_tenis)
# 'no'

# ----------------------------------------------
# I.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y. 

# Ejemplo:

# >>> rendimiento(nb_tenis,X_tenis,y_tenis)
# 0.9285714285714286

def rendimiento_p1(clasificador,X,y):
    y_prediccion = []
    for x in X:
        y_prediccion.append([k for k, v in clasificador.clasifica(x).items()][0])
    total=len(y)
    v_ok=[]
    for i in range(len(y)):
        if y[i]==y_prediccion[i]:
            v_ok.append(i)
    return len(v_ok)/len(y)


# --------------------------
# I.3) Aplicando Naive Bayes
# --------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Concesión de prestamos
# - Críticas de películas en IMDB (ver NOTA con instrucciones para obtenerlo)

# En todos los casos, será necesario separar un conjunto de test para dar la
# valoración final de los clasificadores obtenidos. Si fuera necesario, se permite usar
# train_test_split de Scikit Learn, para separar el conjunto de test y/o
# validación. Ajustar también el valor del parámetro de suavizado k. 

# Mostrar el proceso realizado en cada caso, y los rendimientos obtenidos. 

from sklearn.model_selection import train_test_split

#Prueba con fichero: Votos de congresistas US

# run "votos.py"

# X_votos_train, X_votos_test, y_votos_train, y_votos_test = train_test_split(datos, clasif, test_size=0.25)
# nb_votos = NaiveBayes(k=0.5)
# nb_votos.entrena(X_votos_train,y_votos_train)
# rendimiento_votos=rendimiento(nb_votos,X_votos_test,y_votos_test)

# rendimiento_votos

#0.8990825688073395

#Prueba con fichero: Concesión de prestamos

# run "credito.py"

#Prueba con fichero: Votos de congresistas US

# X_credito_train, X_credito_test, y_credito_train, y_credito_test = train_test_split(X_credito, y_credito, test_size=0.25)
# nb_credito = NaiveBayes(k=0.7)
# nb_credito.entrena(X_credito_train,y_credito_train)
# rendimiento_credito=rendimiento(nb_credito,X_credito_test,y_credito_test)

# rendimiento_credito

#0.7361963190184049

def optimizar_nb(X,y):
    result = []
    band = False
    for i in range(1,20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
        nb = NaiveBayes(k=i/20)
        nb.entrena(X_train, y_train)
        valor = rendimiento_p1(nb, X_test, y_test)
   
       # Buscar maximo
        if band == True and result[-1] < valor:
            max = valor
   
       # primera vez
        if band == False:
            max = valor
            band = True
       # Guardo registros
        result.append(valor)

    for i in range(len(result)):
        #print("Rendimiento : "+str(result[i])+" con k = "+str((i+1)/10))
        if max == result[i]:
            print("El mejor rendimiento es de : "+str(max)+" con k = "+str((i+1)/10)) 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTA:
# INSTRUCCIONES PARA OBTENER EL CONJUNTO DE DATOS IMDB A USAR EN EL TRABAJO

# Este conjunto de datos ya se ha usado en un ejercicio del tema de modelos 
# probabilísticos. Los textos en bruto y comprimidos están en aclImdb.tar.gz, 
# que se ha de descomprimir previamente (NOTA: debido a la gran cantidad de archivos
# que aparecen al descomprimir, se aconseja pausar la sincronización si se está conectado
# a algún servicio en la nube).

# NO USAR TODO EL CONJUNTO: extraer, usando random.sample, 
# 2000 críticas en el conjunto de entrenamiento y 400 del conjunto de test. 
# Usar por ejemplo la siguiente secuencia de instrucciones, para extraer los textos:


# >>> import random as rd
# >>> from sklearn.datasets import load_files
# >>> reviews_train = load_files("data/aclImdb/train/")
# >>> muestra_entr=random.sample(list(zip(reviews_train.data,
#                                     reviews_train.target)),k=2000)
# >>> text_train=[d[0] for d in muestra_entr]
# >>> text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
# >>> yimdb_train=np.array([d[1] for d in muestra_entr])
# >>> reviews_test = load_files("data/aclImdb/test/")
# >>> muestra_test=random.sample(list(zip(reviews_test.data,
#                                         reviews_test.target)),k=400)
# >>> text_test=[d[0] for d in muestra_test]
# >>> text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
# >>> yimdb_test=np.array([d[1] for d in muestra_test])

# Ahora restaría vectorizar los textos. Puesto que la versión NaiveBayes que
# se ha pedido implementar es la categórica (es decir, no es la multinomial),
# a la hora de vectorizar los textos lo haremos simplemente indicando en cada
# componente del vector si el correspondiente término del vocabulario ocurre
# (1) o no ocurre (0). Para ello, usar CountVectorizer de Scikit Learn, con la
# opción binary=True. Para reducir el número de características (es decir,
# reducir el vocabulario), usar "stop words" y min_df=50. Se puede ver cómo
# hacer esto en el ejercicio del tema de modelos probabilísticos.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  















# =====================================================
# PARTE II: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# =====================================================

# En esta SEGUNDA parte se pide implementar en Python un clasificador binario
# lineal, basado en regresión logística. 



# ---------------------------------------------
# II.1) Implementación de un clasificador lineal
# ---------------------------------------------

# En esta sección se pide implementar un clasificador BINARIO basado en
# regresión logística, con algoritmo de entrenamiento de descenso por el
# gradiente mini-batch (para minimizar la entropía cruzada).

# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#     def __init__(self,clases=[0,1],normalizacion=False,
#                  rate=0.1,rate_decay=False,batch_tam=64)
#         .....
        
#     def entrena(self,X,y,n_epochs,reiniciar_pesos=False,pesos_iniciales=None):

#         .....        

#     def clasifica_prob(self,ejemplo):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......

        

# Explicamos a continuación cada uno de estos elementos:


# * El constructor tiene los siguientes argumentos de entrada:

#   + Una lista clases (de longitud 2) con los nombres de las clases del
#     problema de clasificación, tal y como aparecen en el conjunto de datos. 
#     Por ejemplo, en el caso de los datos de las votaciones, esta lista sería
#     ["republicano","democrata"]. La clase que aparezca en segundo lugar de
#     esta lista se toma como la clase positiva.  

#   + El parámetro normalizacion, que puede ser True o False (False por
#     defecto). Indica si los datos se tienen que normalizar, tanto para el
#     entrenamiento como para la clasificación de nuevas instancias. La
#     normalización es la estándar: a cada característica se le resta la media
#     de los valores de esa característica en el conjunto de entrenamiento, y
#     se divide por la desviación típica de dichos valores.

#  + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el número de epoch, y rate_0 es la cantidad introducida
#    en el parámetro rate anterior. Su valor por defecto es False. 

#  + batch_tam: indica el tamaño de los mini batches (por defecto 64) que se
#    usan para calcular cada actualización de pesos.



# * El método entrena tiene los siguientes parámetros de entrada:

#  + X e y son los datos del conjunto de entrenamiento y su clasificación
#    esperada, respectivamente. El primero es un array con los ejemplos, y el
#    segundo un array con las clasificaciones de esos ejemplos, en el mismo
#    orden.

#  + n_epochs: número de veces que se itera sobre todo el conjunto de
#    entrenamiento.

#  + reiniciar_pesos: si es True, cada vez que se llama a entrena, se
#    reinicia al comienzo del entrenamiento el vector de pesos de
#    manera aleatoria (típicamente, valores aleatorios entre -1 y 1).
#    Si es False, solo se inician los pesos la primera vez que se
#    llama a entrena. En posteriores veces, se parte del vector de
#    pesos calculado en el entrenamiento anterior, excepto que se diera
#    explícitamente el vector de pesos en el parámetro peso_iniciales.  

#  + pesos_iniciales: si no es None y el parámetro anterior reiniciar_pesos 
#    es False, es un array con los pesos iniciales. Este parámetro puede ser
#    útil para empezar con unos pesos que se habían obtenido y almacenado como
#    consecuencia de un entrenamiento anterior.



# * Los métodos clasifica y clasifica_prob se describen como en el caso del
#   clasificador NaiveBayes. Igualmente se debe devolver
#   ClasificadorNoEntrenado si llama a los métodos de clasificación antes de
#   entrenar. 

# Se recomienda definir la función sigmoide usando la función expit de
# scipy.special, para evitar "warnings" por "overflow":

# from scipy.special import expit    
#
# def sigmoide(x):
#    return expit(x)


# ----------------------------------------------------------------


# Ejemplo de uso, con los datos del cáncer de mama, que se puede cargar desde
# Scikit Learn:

# >>> from sklearn.datasets import load_breast_cancer
# >>> cancer=load_breast_cancer()

# >>> X_cancer,y_cancer=cancer.data,cancer.target


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)

# >>> lr_cancer.entrena(Xe_cancer,ye_cancer,10000)

# >>> rendimiento(lr_cancer,Xe_cancer,ye_cancer)
# 0.9906103286384976
# >>> rendimiento(lr_cancer,Xt_cancer,yt_cancer)
# 0.972027972027972

# -----------------------------------------------------------------
from scipy.special import expit    


def suma_paralelo(a1, a2):
    """
        Recibe dos arrays y devuelve un array con las sumas de sus componentes sumadas
    """
    a3 = [a1[i]+a2[i] for i in range(len(a1))]
    return a3


def sigmoide(x):
    return expit(x)


def normaliza(X):
    """Normaliza los datos"""
    medias = np.mean(X, axis=0)
    desvs = np.std(X, axis=0)
    X_norm = (X - medias) / desvs
    return X_norm


class RegresionLogisticaMiniBatch():

    def __init__(self, clases=[0, 1], normalizacion=False,
                rate=0.1, rate_decay=False, batch_tam=64, n_epochs=200,
                 pesos_iniciales=None):
        
        mapa_reverse = {0: clases[0], 1: clases[1]}
        self.clases = clases
        self.mapa_reverse = mapa_reverse
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.n_epochs = n_epochs
        self.pesos_iniciales = pesos_iniciales
        self.pesos = list()
        # si pesos está vacía, el clasificador no está entrenado



    def entrena(self, X, y):
        
        pesos = []
        if self.pesos_iniciales is not None:  # tomamos los pesos directamente de la clase
            pesos = list(self.pesos_iniciales)
        else:  # iniciamos los pesos de forma aleatoria
            pesos = [random.random() for i in range(X.shape[1])]
        n_epochs = self.n_epochs
        y_2 = y.reshape(len(y), 1) #transformamos y para poder luego unirlo a X

        if self.normalizacion:
            X = normaliza(X)
                    
        # merge de los array para trabajar mejor con ellos
        big_chunk = np.concatenate((X, y_2), axis=1)
        # inicialización de parámetros
        batch_tam = self.batch_tam
        tasa_l = self.rate
        tasa_l0 = self.rate
        for i in range(n_epochs):
            chunks = np.array_split(big_chunk, batch_tam)
            # dividimos los datos en subconjuntos
            for block in chunks: #iteramos por cada minibatch
                # tomamos un subgrupo de datos
                # para cada subconjunto actualizamos
                pesos_previos = [0.0 for _ in block[0][:-1]]
                for array in block: #iteramos por cada ejemplo de nuestro minibatch para actualizar los pesos
                    sum_a = array[-1]-sigmoide(np.dot(pesos, array[:-1]))
                    sum_t = np.dot(sum_a, array[:-1])
                    pesos_previos = suma_paralelo(pesos_previos, sum_t)
                # una vez hecho todo el sumatorio de los elementos del subgrupo,
                # actualizamos los pesos reales multiplicando por la tasa de
                # aprendizaje y sumando
                
                act_b = np.dot(tasa_l, pesos_previos)
                pesos = suma_paralelo(pesos, act_b)
            if self.rate_decay:
                tasa_l = tasa_l0*(1/(1+i))
        self.pesos = pesos

    def clasifica_prob(self, ejemplo):
        if not self.pesos:
            raise ClasificadorNoEntrenado("Debe entrenar antes el clasificador.")
        else:
            result = sigmoide(np.dot(self.pesos, ejemplo))
            probs = dict()
            reverse= self.mapa_reverse
            probs[reverse.get(0)]=1-result
            probs[reverse.get(1)]=result
            return probs

    def clasifica(self, ejemplo):
        if not self.pesos:
            raise ClasificadorNoEntrenado("Debe entrenar antes el clasificador.")
        else:
            result = sigmoide(np.dot(self.pesos, ejemplo))
            return self.mapa_reverse.get(round(result))



def rendimiento_p2(clasificador, X, y):
    """
    Devuelve el porcentaje de ejemplos bien clasificados
    :param clasificador: clasificador a emplear
    :param X: conjunto de ejemplos X
    :param y: clasificacion esperada
    """
    pred = []  # tablero que contenga nuestra prediccion para todos los ejemplos en X con el clasificador dado en argumento
    for i in range(len(X)):
        pred.append(clasificador.clasifica(X[i]))
    return len(np.where(pred == y)[0]) / len(X)


# from sklearn.datasets import load_breast_cancer

# cancer=load_breast_cancer()

# X_cancer,y_cancer=cancer.data, cancer.target

# Xe_cancer, Xp_cancer, ye_cancer, yp_cancer = particion_entr_prueba(carga_datos.X_cancer,carga_datos.y_cancer)

# lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)

# lr_cancer.entrena(Xe_cancer,ye_cancer,10000)      

# print("Test Regresion logistica sobre los datos del cancer.")
# print("Rendimiento:")
# rendimiento(lr_cancer, normaliza(Xe_cancer), ye_cancer)


# -----------------------------------
# II.2) Aplicando Regresión Logística 
# -----------------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Como antes, será necesario separar conjuntos de validación y test para dar
# la valoración final de los clasificadores obtenidos Se permite usar
# train_test_split de Scikit Learn para esto. Ajustar los parámetros de tamaño
# de batch, tasa de aprendizaje y rate_decay. En alguno de los conjuntos de
# datos puede ser necesaria normalización.

# Mostrar el proceso realizado en cada caso, y los rendimientos obtenidos. 





















# ===================================
# PARTE III: CLASIFICACIÓN MULTICLASE
# ===================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica
# de "One vs Rest" (OvR)

# ------------------------------------
# III.1) Implementación de One vs Rest
# ------------------------------------


#  En concreto, se pide implementar una clase python RL_OvR con la siguiente
#  estructura, y que implemente un clasificador OvR usando como base el
#  clasificador binario del apartado anterior.


# class RL_OvR():

#     def __init__(self,clases,rate=0.1,rate_decay=False,batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs):

#        .......

#     def clasifica(self,ejemplo):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, excepto que ahora "clases" puede ser una lista con más de dos
#  elementos. 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> from sklearn.datasets import load_iris
# >>> iris=load_iris()
# >>> X_iris=iris.data
# >>> y_iris=iris.target
# >>> Xe_iris,Xt_iris,ye_iris,yt_iris=train_test_split(X_iris,y_iris)

# >>> rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20)

# >>> rl_iris.entrena(Xe_iris,ye_iris,n_epochs=1000)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris,Xt_iris,yt_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

class RL_OvR():

     def __init__(self, clases, rate=0.1, rate_decay=False, batch_tam=64, n_epochs=200):
         self.clases = clases
         self.rate = rate
         self.rate_decay = rate_decay
         self.batch_tam = batch_tam
         self.n_epochs = n_epochs
         self.pesos = list()
         self.reg = list()
         # si pesos está vacía, el clasificador no está entrenado

     def entrena(self, X, y):
        for i in range(len(self.clases)):
            self.reg.append(RegresionLogisticaMiniBatch(clases=[0, 1], normalizacion=False, rate=self.rate, rate_decay=self.rate_decay, batch_tam=self.batch_tam, n_epochs=self.n_epochs,pesos_iniciales=None))
            y_new = np.array([1 if j == self.clases[i] else 0 for j in y])
            self.reg[i].entrena(X, y_new)

     def clasifica(self, ejemplo):
         prob = []
         for i in range(len(self.clases)):
            prob.append(self.reg[i].clasifica_prob(ejemplo)[1])
         return self.clases[np.argmax(prob)]


# ------------------------------------------------------------
# III.2) Clasificación de imágenes de dígitos escritos a mano
# ------------------------------------------------------------


#  Aplicar la implementación del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. Si el
#  tiempo de cómputo en el entrenamiento no permite terminar en un tiempo
#  razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 

def convierte_0_1(c):
    if c==" ":
        return 0
    else:
        return 1

# Esta función lee un fichero dado con imágenes de 28x29 pixeles que representan números y devuelve
# un vector que representa los dígitos del fichero.
def leer_digitos(fichero):

    f = open(fichero)
    count = 0
    datos = []
    imagen = []
    for linea in f:
        transform_linea = [convierte_0_1(c) for c in linea]
        imagen.append(transform_linea)
        count += 1
        if count == 28:
            count = 0
            datos.append(imagen)
            imagen = []
    f.close()
    return np.array(datos)


def leer_label(fichero):
    f = open(fichero)
    labels = []
    for l in f:
        labels.append(int(l))
    f.close()
    return np.array(labels)


# Ejecucion con python clasificadores.py -t True
# Ejecucion con python clasificadores.py -o True -t False
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topic", help=" Para imprimir los resultados", default=True)
    parser.add_argument("-o", "--optimize", help=" Para imprimir optimizacion y resultados segun modelos", default=False)
    args = parser.parse_args()
    
    
    if args.topic == "True":
        # Ejemplo para imprimir resultado de Punto I.3 

        # Tenis
        print("----------------------------------") 
        print("Test NaiveBayes con datos de Jugar al tenis")
        nb_tenis = NaiveBayes(k=0.5)
        nb_tenis.entrena(X_tenis,y_tenis)
        #ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
        #nb_tenis.clasifica_prob(ej_tenis)
        #nb_tenis.clasifica(ej_tenis)
        print("Rendimiento: ",rendimiento_p1(nb_tenis,X_tenis,y_tenis))
        # Rendimiento:  

        # Votos
        print("----------------------------------") 
        print("Test NaiveBayes con datos de Votos")
        X_votos_train, X_votos_test, y_votos_train, y_votos_test = train_test_split(X_votos, y_votos, test_size=0.25, random_state=10)
        nb_votos = NaiveBayes(k=0.5)
        nb_votos.entrena(X_votos_train,y_votos_train)
        print("Rendimiento: ",rendimiento_p1(nb_votos,X_votos_test,y_votos_test))
        # Rendimiento:  0.908256880733945

        # Creditos
        print("------------------------------------") 
        print("Test NaiveBayes con datos de Credito")
        X_credito=np.array([d[:-1] for d in datos_con_la_clase])
        y_credito=np.array([d[-1] for d in datos_con_la_clase])
        X_credito_train, X_credito_test, y_credito_train, y_credito_test = train_test_split(X_credito, y_credito, test_size=0.25, random_state=10)
        nb_credito = NaiveBayes(k=0.7)
        nb_credito.entrena(X_credito_train,y_credito_train)
        print("Rendimiento: ", rendimiento_p1(nb_credito,X_credito_test,y_credito_test))
        # Rendimiento:  0.6257668711656442

        # Peliculas
        
        # Ejemplo para imprimir resultado de Punto II.1
        print("----------------------------------------------------") 
        print("Test Regresion logistica sobre los datos del cancer.")
        cancer=load_breast_cancer()
        X_cancer,y_cancer=cancer.data, cancer.target
        # Xe_cancer, Xp_cancer, ye_cancer, yp_cancer = particion_entr_prueba(X_cancer,y_cancer)
        X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, test_size=0.25, random_state=10)
        lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
        lr_cancer.entrena(X_cancer_train, y_cancer_train)
        print("Rendimiento: ",rendimiento_p2(lr_cancer, normaliza(X_cancer_test), y_cancer_test))
        # Rendimiento:  0.951048951048951

        # Ejemplo para imprimir resultado de Punto II.1
        print("--------------------------------------------------") 
        print("Test Regresion logistica sobre los datos de Votos.")
        X_votos_train, X_votos_test, y_votos_train, y_votos_test = train_test_split(X_votos, y_votos, test_size=0.25, random_state=10)
        lr_votos = RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
        lr_votos.entrena(X_votos_train, y_votos_train)
        print("Rendimiento: ",rendimiento_p2(lr_votos, normaliza(X_votos_test), y_votos_test))
        # Rendimiento:  0.9541284403669725

        # Ejemplo para imprimir resultado de Punto III.1
        print("------------------------------------") 
        print("Test One vs Rest con los datos iris.")
        X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.25, random_state=10)
        rl_iris = RL_OvR([0, 1, 2], rate=0.001, batch_tam=20, n_epochs=1000)
        rl_iris.entrena(X_iris_train, y_iris_train)
        print("Rendimiento:", rendimiento_p2(rl_iris, X_iris_test, y_iris_test), "\n")
        # Rendimiento: 0.9473684210526315 

        # Clasificación de imágenes de dígitos escritos a mano de Punto III.2
        print("---------------------------------------------------------------------------------------") 
        print("Test One vs Rest con los datos de clasificación de imágenes de dígitos escritos a mano.")
        # Train tiene 5000 ejemplos, test 1000 y validación 1000. 
        # Hemos tomado de train sólo 500 ejemplos para reducir el tiempo de ejecución.
        #X_digitos_train = leer_digitos("digitdata/trainingimages").reshape(5000, 28*29)[0:500]
        #y_digitos_train = leer_label("digitdata/traininglabels")[0:500]
        #X_digitos_test = leer_digitos("digitdata/testimages").reshape(1000, 28*29)
        #y_digitos_test = leer_label("digitdata/testlabels")
        #X_digitos_val = leer_digitos("digitdata/validationimages").reshape(1000, 28*29)
        #y_digitos_val = leer_label("digitdata/validationlabels")
        #ovr_digitos = RL_OvR(np.arange(10), rate=0.4, batch_tam=64, n_epochs=100)
        #ovr_digitos.entrena(X_digitos_train, y_digitos_train)
        #print("Rendimiento:", rendimiento_p2(ovr_digitos, X_digitos_test, y_digitos_test), "\n")
        # Rendimiento: 0.763


    if args.optimize == "True":
        # Codigo mezclado de posible funcion de busqueda de maximo con 1 parametro.
        # Habria que generalziar para N parametros, identificar y ajustar
        result = []
        band = False
        for i in range(1,20):
            # Prueba con skiti
            # nb_votos = CategoricalNB(alpha=i/10)
            # nb_votos.fit(X_votos_train,y_votos_train)
            X_votos_train, X_votos_test, y_votos_train, y_votos_test = train_test_split(X_votos, y_votos, test_size=0.25, random_state=10)
            nb_votos = NaiveBayes(k=i/20)
            nb_votos.entrena(X_votos_train, y_votos_train)
            valor = rendimiento_p1(nb_votos, X_votos_test, y_votos_test)
        
            # Buscar maximo
            if band == True and result[-1] < valor:
                max = valor
        
            # primera vez
            if band == False:
                max = valor
                band = True
            # Guardo registros
            result.append(valor)

        for i in range(len(result)):
            print("Rendimiento : "+str(result[i])+" con k = "+str((i+1)/10))
            if max == result[i]:
                print("El mejor rendimiento es de : "+str(max)+" con k = "+str((i+1)/10))

        #print("-----------------------------------------------") 
        #print("Test NaiveBayes con datos de Votos: optimizar k")
        #optimizar_nb(X_votos,y_votos)

        #print("-------------------------------------------------") 
        #print("Test NaiveBayes con datos de Credito: optimizar k")
        #optimizar_nb(X_credito,y_credito)

if __name__ == '__main__':
    main()