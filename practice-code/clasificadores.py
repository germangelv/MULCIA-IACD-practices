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
        self.flag_entrenado = False
        
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
            self.flag_entrenado = True

    def clasifica_prob(self,ejemplo):
        if (self.flag_entrenado == False):
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
        if (self.flag_entrenado == False):
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

def rendimiento(clasificador,X,y):
    y_prediccion = []
    for x in X:
        y_prediccion.append([k for k, v in clasificador.clasifica(x).items()][0])
    total=len(y)
    v_ok=[]
    for i in range(len(y)):
        if y[i]==y_prediccion[i]:
            v_ok.append(i)
    return len(v_ok)/len(y)


# --------------------------`¡'0`
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

# Creamos una función para buscar el parámetro que optimiza el rendimiento de Naive Bayes.

def optimizar_nb(X_train,X_test,y_train,y_test):
    result = []
    band = False
    for i in range(1,10):
        nb = NaiveBayes(k=i/10)
        nb.entrena(X_train, y_train)
        valor = rendimiento(nb, X_test, y_test)

       # Buscar maximo
        if band == True and result[-1][0] < valor:
            max = valor
   
       # primera vez
        if band == False:
            max = valor
            band = True

        # Guardo registros
        result.append([valor,i/10])

    for i in range(len(result)):
        if max == result[i][0]:
            print("El mejor rendimiento es de : "+str(max)+" con k = "+str(result[i][1]))

# -> Ejecución con python para Ejercicio 2: clasificadores.py -e 3

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


# Creamos una función para importar los datos de IMDB y vectorizarlos.
  
def cine():
    import random as rd
    from sklearn.datasets import load_files
    from sklearn.feature_extraction.text import CountVectorizer
    
    print('Load Info')
    reviews_train = load_files("aclImdb/train/")
    muestra_entr=random.sample(list(zip(reviews_train.data,
                                    reviews_train.target)),k=2000)
    text_train=[d[0] for d in muestra_entr]
    text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
    y_imdb_train=np.array([d[1] for d in muestra_entr])
    reviews_test = load_files("aclImdb/test/")
    muestra_test=random.sample(list(zip(reviews_test.data,
                                        reviews_test.target)),k=400)
    text_test=[d[0] for d in muestra_test]
    text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
    y_imdb_test=np.array([d[1] for d in muestra_test])

    print('Vectorizer')
    vectorizer = CountVectorizer(min_df=100, stop_words="english", binary=True).fit(text_train)
    
    X_train_imdb = vectorizer.transform(text_train).toarray()
    X_test_imdb =vectorizer.transform(text_test).toarray()
    




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

def sigmoide(x):
    return expit(x)

def normalizar(X):
    medias = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - medias) / std
    return X_norm

class RegresionLogisticaMiniBatch():

    def __init__(self, clases=[0, 1], normalizacion=False,
                rate=0.1, rate_decay=False, batch_tam=64):
        
        dic_clases = {}
        dic_clases[0] = clases[0]
        dic_clases[1] = clases[1]
        self.dic_clases = dic_clases
        self.clases = clases
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.flag_entrenado = False


    def entrena(self, X, y,n_epochs,reiniciar_pesos=False,pesos_iniciales=None):

        self.n_epochs = n_epochs
        self.reiniciar_pesos = reiniciar_pesos
        self.pesos_iniciales = pesos_iniciales
        self.pesos = list()
        
        
        if reiniciar_pesos == True:
            pesos = []
            if self.pesos_iniciales is not None:  # Si nos dan los pesos iniciales, los usamos para comenzar
                pesos = list(self.pesos_iniciales)
            else:  # Sino, iniciamos los pesos de forma aleatoria usando random, con valores entre -1 y 1
                pesos = [random.uniform(-1,1) for i in range(X.shape[1])]
        else: # Si reiniciar_pesos es False, sólo los reiniciamos cuando es la primera vez que entrenamos
            if self.flag_entrenado == False:
                pesos = []
                if self.pesos_iniciales is not None:  # Si nos dan los pesos iniciales, los usamos para comenzar
                    pesos = list(self.pesos_iniciales)
                else:  # Sino, iniciamos los pesos de forma aleatoria usando random
                    pesos = [random.uniform(-1,1) for i in range(X.shape[1])]
        
        y_2 = y.reshape(len(y), 1) # Transformamos y para poder luego unirlo a X

        if self.normalizacion: # Si nos piden normalizar los datos, aplicamos la función de normalización
            X = normalizar(X)
                    
        # Unimos X e y en un array para dividir en cada epoch ambos conjuntos
        conjunto_total = np.concatenate((X, y_2), axis=1)
        rate_0 = self.rate
        rate_n = self.rate
        # Por cada epoch vamos a dividir el cjto en mini-batches y actualizar los pesos en cada uno de ellos
        for n in range(n_epochs):
            total_minibatches = np.array_split(conjunto_total, self.batch_tam) #En cada epoch, dividimos el cjto total
            #en minibatches del tamaño dado como parámetro batch_tam
            # dividimos los datos en subconjuntos
            for minibatch in total_minibatches: #Iteramos por cada minibatch
                suma_acumulada = [0 for x in minibatch[0][:-1]]
                for minib_element in minibatch:
                    # Separamos de nuevo en X e y
                    minib_X = minib_element[:-1]
                    minib_y = minib_element[-1]
                    diferencia_y_hip = minib_y - sigmoide(np.dot(pesos, minib_X))
                    sumatorio = np.dot(diferencia_y_hip, minib_X)
                    suma_acumulada = np.add(suma_acumulada, sumatorio) # Sumamos el vector resultante con lo que 
                    #ya llevábamos de las anteriores iteraciones
                # Multiplicamos por la tasa de aprendizaje y sumamos los pesos anteriores
                actualizacion_pesos = np.dot(rate_n, suma_acumulada)
                pesos = np.add(pesos, actualizacion_pesos)
            if self.rate_decay:
                rate_n = rate_0*(1/(1+n))
        self.pesos = pesos
        self.flag_entrenado = True

    def clasifica_prob(self, ejemplo):
        if self.flag_entrenado == False:
            raise ClasificadorNoEntrenado("Debe entrenar antes el clasificador.")
        else:
            result = sigmoide(np.dot(self.pesos, ejemplo))
            probs = {}
            dic_clases = self.dic_clases
            probs[dic_clases.get(0)] = (1-result)
            probs[dic_clases.get(1)] = (result)
            return probs

    def clasifica(self, ejemplo):
        if self.flag_entrenado == False:
            raise ClasificadorNoEntrenado("Debe entrenar antes el clasificador.")
        else:
            maximo = {}
            find_max = max(self.clasifica_prob(ejemplo), key=self.clasifica_prob(ejemplo).get)
            maximo[find_max] = self.clasifica_prob(ejemplo)[find_max]
            
        return maximo


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


# -> Ejecución con python para Ejercicio 2: clasificadores.py -e 2



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

     def __init__(self, clases, rate=0.1, rate_decay=False, batch_tam=64):
         self.clases = clases
         self.rate = rate
         self.rate_decay = rate_decay
         self.batch_tam = batch_tam
         self.pesos = list()
         self.clasificadores_bin = list()
         self.flag_entrenado = False

     def entrena(self, X, y, n_epochs):
        self.n_epochs = n_epochs

        for i in range(len(self.clases)):
            self.clasificadores_bin.append(RegresionLogisticaMiniBatch(clases=[0, 1], normalizacion=False, rate=self.rate, rate_decay=self.rate_decay, batch_tam=self.batch_tam))
            # Transformamos el vector y marcando como 1 los valores que corresponden a la clase actual, y 0 el resto
            y_clase = np.array([1 if j == self.clases[i] else 0 for j in y])
            self.clasificadores_bin[i].entrena(X, y_clase, n_epochs=self.n_epochs, reiniciar_pesos=True, pesos_iniciales=None)
            self.flag_entrenado = True

     def clasifica(self, ejemplo):
        if self.flag_entrenado == False:
            raise ClasificadorNoEntrenado("Debe entrenar antes el clasificador.")
        else:
         probs = {}
         maximo = {}
         for i in range(len(self.clases)):
            probs[i] = self.clasificadores_bin[i].clasifica_prob(ejemplo)[1]
        find_max = max(probs, key=probs.get)
        maximo[find_max] = probs[find_max]
        return maximo


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
def cargaImágenes(fichero,ancho,alto):
    with open(fichero) as f:
        lista_imagenes=[]
        ejemplo=[]
        cont_lin=0
        for lin in f:
            ejemplo.extend(list(map(convierte_0_1,lin[:ancho])))
            cont_lin+=1
            if cont_lin == alto:
                lista_imagenes.append(ejemplo)  
                ejemplo=[]
                cont_lin=0
    return np.array(lista_imagenes)


def cargaClases(fichero):
    with open(fichero) as f:
        return np.array([int(c) for c in f])



# ------------------------------------------------------------
# Ejecución de los resultados para los diferentes modelos
# ------------------------------------------------------------


# Ejecucion con python para Ejercicio 1: clasificadores.py -e 1
# Ejecucion con python para Ejercicio 2: clasificadores.py -e 2
# Ejecucion con python para Ejercicio 3: clasificadores.py -e 3
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--ejercicio", help=" Indicar ejercicio (1,2 o 3)", required=True)
    args = parser.parse_args()
    

    if args.ejercicio == "1":       
        
        # Ejemplo para imprimir resultado de Punto I.3 

        # Tenis
        print("----------------------------------------------------") 
        print("Test NaiveBayes con datos de Jugar al tenis: ejemplo", "\n")
        nb_tenis = NaiveBayes(k=0.5)
        nb_tenis.entrena(X_tenis,y_tenis)
        #ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
        #nb_tenis.clasifica_prob(ej_tenis)
        #nb_tenis.clasifica(ej_tenis)
        print("Rendimiento con k = 0.5: ",rendimiento(nb_tenis,X_tenis,y_tenis))
        # Rendimiento:  0.9285714285714286

        # Votos
        print("-------------------------------------------") 
        print("Test NaiveBayes con datos de Votos: ejemplo", "\n")
        X_votos_train, X_votos_test, y_votos_train, y_votos_test = train_test_split(X_votos, y_votos, test_size=0.2, random_state=10)
        X_votos_train, X_votos_val, y_votos_train, y_votos_val = train_test_split(X_votos_train, y_votos_train, test_size=0.25, random_state=10)

        nb_votos = NaiveBayes(k=0.5)
        nb_votos.entrena(X_votos_train,y_votos_train)
        print("Rendimiento con k = 0.5: ",rendimiento(nb_votos,X_votos_test,y_votos_test))

        print("Buscamos los hiperparámetros que dan el mejor rendimiento.", "\n")
        optimizar_nb(X_votos_train,X_votos_test,y_votos_train,y_votos_test)
        print("Vemos el rendimiento para el primer modelo con el conjunto de validación: ",rendimiento(nb_votos,X_votos_val,y_votos_val))

        # Creditos
        print("---------------------------------------------") 
        print("Test NaiveBayes con datos de Credito: ejemplo", "\n")
        X_credito=np.array([d[:-1] for d in datos_con_la_clase])
        y_credito=np.array([d[-1] for d in datos_con_la_clase])
        X_credito_train, X_credito_test, y_credito_train, y_credito_test = train_test_split(X_credito, y_credito, test_size=0.2, random_state=10)
        X_credito_train, X_credito_val, y_credito_train, y_credito_val = train_test_split(X_credito_train, y_credito_train, test_size=0.25, random_state=10)

        nb_credito = NaiveBayes(k=0.7)
        nb_credito.entrena(X_credito_train,y_credito_train)
        print("Rendimiento con k = 0.7: ", rendimiento(nb_credito,X_credito_test,y_credito_test))

        print("Buscamos los hiperparámetros que dan el mejor rendimiento.")
        optimizar_nb(X_credito_train,X_credito_test,y_credito_train,y_credito_test)
        print("Vemos el rendimiento para el primer modelo con el conjunto de validación: ",rendimiento(nb_credito,X_credito_val,y_credito_val))


        # Peliculas
        print("------------------------------------------") 
        print("Test NaiveBayes con datos de IMDB: ejemplo", "\n")
        #cine()
        #nb_imdb = NaiveBayes(k=1)
        #nb_imdb.entrena(X_train_imdb, y_imdb_train)
        #print("Rendimiento con k = 1: ",rendimiento(nb_imdb, X_test_imdb, y_imdb_test))
        # Rendimiento con Naive Bayes: 0.7625

    elif args.ejercicio == "2":

        # Ejemplo para imprimir resultado de Punto II.1
        print("------------------------------------------------------------") 
        print("Test Regresion logistica sobre los datos del cancer: ejemplo", "\n")
        cancer=load_breast_cancer()
        X_cancer,y_cancer=cancer.data, cancer.target
        # Xe_cancer, Xp_cancer, ye_cancer, yp_cancer = particion_entr_prueba(X_cancer,y_cancer)
        X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=10)
        X_cancer_train, X_cancer_val, y_cancer_train, y_cancer_val = train_test_split(X_cancer_train, y_cancer_train, test_size=0.25, random_state=10)


        lr_cancer1=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
        lr_cancer1.entrena(X_cancer_train, y_cancer_train,100)
        print("Rendimiento con rate=0.1, rate_decay=True, normalizacion=True y n_epochs=100: ",
        rendimiento(lr_cancer1, normalizar(X_cancer_test), y_cancer_test))

        lr_cancer2=RegresionLogisticaMiniBatch(rate=0.01,rate_decay=True,normalizacion=True)
        lr_cancer2.entrena(X_cancer_train, y_cancer_train,100)
        print("Rendimiento con rate=0.01, rate_decay=True, normalizacion=True y n_epochs=100: ",
        rendimiento(lr_cancer2, normalizar(X_cancer_test), y_cancer_test))

        lr_cancer3=RegresionLogisticaMiniBatch(rate=0.01,rate_decay=True,normalizacion=True)
        lr_cancer3.entrena(X_cancer_train, y_cancer_train,200)
        print("Rendimiento con rate=0.01, rate_decay=True, normalizacion=True y n_epochs=200: ",
        rendimiento(lr_cancer3, normalizar(X_cancer_test), y_cancer_test))

        lr_cancer4=RegresionLogisticaMiniBatch(rate=0.01,rate_decay=False,normalizacion=True)
        lr_cancer4.entrena(X_cancer_train, y_cancer_train,100)
        print("Rendimiento con rate=0.01, rate_decay=False, normalizacion=True y n_epochs=100: ",
        rendimiento(lr_cancer4, normalizar(X_cancer_test), y_cancer_test), "\n")

        print("Vemos el rendimiento para el primer modelo con el conjunto de validación: ",
                rendimiento(lr_cancer1, normalizar(X_cancer_val), y_cancer_val))

        # Rendimiento:  

        # Ejemplo para imprimir resultado de Punto II.1
        print("----------------------------------------------------------") 
        print("Test Regresion logistica sobre los datos de Votos: ejemplo", "\n")
        X_votos_train, X_votos_test, y_votos_train, y_votos_test = train_test_split(X_votos, y_votos, test_size=0.2, random_state=10)
        X_votos_train, X_votos_val, y_votos_train, y_votos_val = train_test_split(X_votos_train, y_votos_train, test_size=0.25, random_state=10)

        lr_votos1 = RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
        lr_votos1.entrena(X_votos_train, y_votos_train,100)
        print("Rendimiento con rate=0.1, rate_decay=True, normalizacion=True y n_epochs=100: ",
        rendimiento(lr_votos1, normalizar(X_votos_test), y_votos_test))
  
        lr_votos2 = RegresionLogisticaMiniBatch(rate=0.3,rate_decay=True,normalizacion=True)
        lr_votos2.entrena(X_votos_train, y_votos_train,100)  
        print("Rendimiento con rate=0.3, rate_decay=True, normalizacion=True y n_epochs=100: ",
        rendimiento(lr_votos2, normalizar(X_votos_test), y_votos_test))      
        
        lr_votos2 = RegresionLogisticaMiniBatch(rate=0.01,rate_decay=True,normalizacion=True)
        lr_votos2.entrena(X_votos_train, y_votos_train,100)  
        print("Rendimiento con rate=0.01, rate_decay=True, normalizacion=True y n_epochs=100: ",
        rendimiento(lr_votos2, normalizar(X_votos_test), y_votos_test))      
        
        lr_votos3 = RegresionLogisticaMiniBatch(rate=0.3,rate_decay=False,normalizacion=True)
        lr_votos3.entrena(X_votos_train, y_votos_train,100)  
        print("Rendimiento con rate=0.3, rate_decay=False, normalizacion=True y n_epochs=100: "
        ,rendimiento(lr_votos3, normalizar(X_votos_test), y_votos_test), "\n")

        print("Vemos el rendimiento para el primer modelo con el conjunto de validación: ",
                rendimiento(lr_votos1, normalizar(X_votos_val), y_votos_val))
        # Rendimiento:  

        print("---------------------------------------------------") 
        print("Test Regresion logistica con datos de IMDB: ejemplo", "\n")
        #cine()
        #lr_imdb = RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
        #lr_imdb.entrena(X_train_imdb, y_imdb_train)
        #print("Rendimiento: ",rendimiento(lr_imdb, normaliza(X_test_imdb), y_imdb_test))
        # Rendimiento con Regresión logística: 0.735

    elif args.ejercicio == "3":

        # Ejemplo para imprimir resultado de Punto III.1
        print("--------------------------------------------") 
        print("Test One vs Rest con los datos iris: ejemplo", "\n")
        X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.25, random_state=10)
        rl_iris = RL_OvR([0, 1, 2], rate=0.001, batch_tam=20)
        rl_iris.entrena(X_iris_train, y_iris_train, n_epochs=500)
        print("Rendimiento con rate=0.001, batch_tam=20, n_epochs=500:", rendimiento(rl_iris, X_iris_test, y_iris_test), "\n")
        # Rendimiento:  0.9210526315789473

        # Clasificación de imágenes de dígitos escritos a mano de Punto III.2
        print("--------------------------------------------------------------------------------------") 
        print("Test One vs Rest con los datos de clasificación de imágenes de dígitos escritos a mano", "\n")
        # Train tiene 5000 ejemplos, test 1000 y validación 1000. 
        # Hemos tomado de train sólo 600 ejemplos para reducir el tiempo de ejecución.
        X_digitos_train = cargaImágenes("digitdata/trainingimages",28,28)[0:600]
        y_digitos_train = cargaClases("digitdata/traininglabels")[0:600]
        X_digitos_test = cargaImágenes("digitdata/testimages",28,28)
        y_digitos_test = cargaClases("digitdata/testlabels")
        X_digitos_val = cargaImágenes("digitdata/validationimages",28,28)
        y_digitos_val = cargaClases("digitdata/validationlabels")
        # Probamos con distintos valores de los hiperparámetros:
        ovr_digitos = RL_OvR(np.arange(10), rate=0.8, batch_tam=64)
        ovr_digitos.entrena(X_digitos_train, y_digitos_train, n_epochs=100)
        print("Rendimiento con rate=0.8, batch_tam=64 y n_epochs=100:", rendimiento(ovr_digitos, X_digitos_test, y_digitos_test))
        ovr_digitos = RL_OvR(np.arange(10), rate=0.4, batch_tam=64)
        ovr_digitos.entrena(X_digitos_train, y_digitos_train, n_epochs=400)
        print("Rendimiento con rate=0.4, batch_tam=64 y n_epochs=400:", rendimiento(ovr_digitos, X_digitos_test, y_digitos_test))
        ovr_digitos = RL_OvR(np.arange(10), rate=0.2, batch_tam=64)
        ovr_digitos.entrena(X_digitos_train, y_digitos_train, n_epochs=200)
        print("Rendimiento con rate=0.2, batch_tam=64 y n_epochs=200:", rendimiento(ovr_digitos, X_digitos_test, y_digitos_test), "\n")
        print("Vemos el rendimiento para el último modelo con el conjunto de validación: ",
                rendimiento(ovr_digitos, X_digitos_val, y_digitos_val))
        # Rendimiento: 0.783


if __name__ == '__main__':
    main()