# ejercicios-python-iacd.py
# Inteligencia Artidicial para la Ciencia de Datos
# =================================================

# -----------
# EJERCICIO 1
# -----------

# Definir una función suma(l) que recibiendo como entrada una lista l de
# números, devuelva la suma de sus elementos.

# Por ejemplo:

# >>> suma([2,4.6,3.1,2.8,5,8,9,23])
# 57.5









# -----------
# EJERCICIO 2
# -----------

# Definir una función n_elementos_pos(l) que recibiendo como entrada una
# lista l de números enteros, devuelva el número de elementos positivos de la
# lista 

# Por ejemplo:

# >>> n_elementos_pos([-2,2,1,-3,2,5,-6,4,5,2,-8])
# 7






    



# -----------
# EJERCICIO 3
# -----------

# Definir una función máximo(l) que recibiendo como entrada una lista l de
# números, devuelva el mayor de sus elementos

# Por ejemplo:

# >>> máximo([23,2,45,6,78,2,4,9,55])
# 78








# -----------
# EJERCICIO 4
# -----------

# Definir una función suma_saltando(l,i,n) que recibiendo como entrada una lista
# números, una posición i de esa lista, y un número natural n, devuelve la
# suma de los elementos de la lista, empezanod en el i-ésimo y saltando de n
# en n. 

# Por ejemplo:

# >>> suma_saltando([2,4,3,7,8,1,2,9,4,3,2],4,3)
# 19
# >>> suma_saltando([2,4,3,7,8,1,2,9,4,3,2],2,2)
# 19
# >>> suma_saltando([2,4,3,7,8,1,2,9,4,3,2],3,2)
# 20





# -----------
# EJERCICIO 5
# -----------

# Definir una función pos_máximo(l) que recibiendo como entrada una lista de
# números, devuelve la posición del mayor elemento de la lista.

# Por ejemplo:

# >>> pos_máximo([23,2,45,6,78,2,4,9,55])
# 4






# -----------
# EJERCICIO 6
# -----------

# Definir una función media(l) que recibiendo una lista numérica como entrada,
# devuelve la media aritmética de sus elementos 

# Por ejemplo:

# >>> media([1,2,5,2,3,6,7])
# 3.7142857142857144












# -----------
# EJERCICIO 7
# -----------

# Definir una función varianza(l) que recibiendo una lista numérica como
# entrada, devuelve la varianza de ese conjunto de números

# Por ejemplo:

# >>> varianza([1,2,5,2,3,6,7])
# 4.489795918367346









# -----------
# EJERCICIO 8
# -----------

# Definir una función mediana(l) que recibiendo una lista numérica como
# entrada, devuelve la mediana de ese conjunto de números. Nota: puede ser de
# utilidad usar la función predefinida sorted(l), que ordena listas. 

# Por ejemplo:

# >>> mediana([3,1,4,2,7,8,5,3,5])
# 4
# >>> mediana([9,1,4,3,3,2,2,4,5,3,11,6])
# 3.5









# -----------
# EJERCICIO 9
# -----------

# Definir una función producto_acumulado(l),que recibiendo como entrada una
# lista l de números, devuelve la lista de la misma longitud, y en la que el
# elemento de la posición i es el resultado del producto de los elementos de
# la lista de entrada que es tán en una posición menor o igual que i. 

# Por ejemplo:


# >>> producto_acumulado([2,3,1,4,7])
# [2, 6, 6, 24, 49, 168]





# ------------
# EJERCICIO 10
# ------------
#

# Definir una función factoriza_primos(n), que recibiendo como entrada un número 
# natural n, cuya factorizaciçón en números primos es n=p1^e1*p2^e2*...*p_m^em,
# devueve la lista [(p1,e1),(p2,e2),...,(p_m,em)] 

# Ejemplos:

# >>> factoriza_primos(171)
# >>> [(3, 2), (19, 1)]
# >>> factoriza_primos(272250)
# [(2, 1), (3, 2), (5, 3), (11, 2)]
# >>> factoriza_primos(358695540883235472)
# [(2, 4), (3, 1), (7, 1), (83, 2), (173, 5)]

# NOTA: Hacerlo sin usar una lista predefinida de números primos.
# SUGERENCIA: se puede hacer mediante dos bucles "while" anidados. 
# El más interno calcula el exponente de cada posible divisor del número, 
# dividiendo con ese divisor mientras sea divisible. 
# El bucle externo contendría al bucle interno
# y además iría incrementando en uno el valor de ese posible divisor.    








# ------------
# EJERCICIO 11
# ------------

# Usando como técnica principal la definición de secuencias por comprensión,
# definir las siguientes funciones:

# a) Dada una lista de números naturales, la suma de los cuadrados de los
#    números pares de la lista.

# Ejemplo:
# >>> suma_cuadrados([9,4,2,6,8,1])
# 120






# b) Dada una lista de números l=[a(1),...,a(n)], calcular el sumatorio de i=1
#    hasta n de i*a(i).

# Ejemplo:

# >>> suma_fórmula([2,4,6,8,10])
# 110

def suma_fórmula(l):
    return sum((i+1)*e for i,e in enumerate(l))

# =============

# c) Dados dos listas numéricas de la misma longitud, representado dos puntos
#    n-dimensionales, calcular la distancia euclídea entre ellos. 

# Ejemplo:

# >>> distancia([3,1,2],[1,2,1])
# 2.449489742783178
from math import sqrt

def distancia(l1,l2):
    return sqrt(sum((x-y)**2 
                for x,y in zip(l1,l2)))

    
    
# ===========

# d) Dada un par de listas (de la misma longitud) y una función de dos
#    argumentos, devolver la lista de los resultados de aplicar la función a
#    cada par de elementos que ocupan la misma posición en las listas de
#    entrada.


# Ejemplo:

# >>> map2_mio((lambda x,y: x+y) ,[1,2,3,4],[5,2,7,9])
# [6, 4, 10, 13]


def map2_mio(f,l1,l2):
    return [f(i,j) for i,j in zip(l1,l2)]




# e) Dada una lista de números, contar el número de elementos que sean múltiplos
#    de tres y distintos de cero. 

# Ejemplo:

# >>> m3_no_nulos([4,0,6,7,0,9,18])
# 3

def m3_no_nulos(l):
    return sum(x%3==0 and x!=0 for x in l)







# f) Dadas dos listas de la misma longitud, contar el número de posiciones en
#    las que coinciden los elementos de ambas listas.  

# Ejemplo:

# >>> cuenta_coincidentes([4,2,6,8,9,3],[3,2,1,8,9,6])
# 3








# ------------
# EJERCICIO 12
# ------------


# Definir la siguiente función usando comprensión. Dadas dos listas de la
# misma longitud, devolver un diccionario que tiene como claves las posiciones
# en las que coinciden los elementos de ambas listas, y como valor de esas
# claves, el elemento coincidente.

# Ejemplos:

# >>> dic_posiciones_coincidentes([4,2,6,8,9,3],[3,2,1,8,9,6])
# {1: 2, 3: 8, 4: 9}
# >>> dic_posiciones_coincidentes([2,8,1,2,1,3],[1,8,1,2,1,6])
# {1: 8, 2: 1, 3: 2, 4: 1}


def dic_posiciones_coincidentes(l,m):
    return {i:x for i,(x,y) in enumerate(zip(l,m))
                        if x==y}



# ------------
# EJERCICIO 13
# ------------
#
# Supongamos que recibimos un diccionario cuyas claves son cadenas de
# caracteres de longitud uno y los valores asociados son números enteros 
# entre 0 y 50. 
# Definir una función histograma_horizontal(d), que recibiendo un diccionario 
# de ese tipo, escribe un histograma de barras horizontales asociado, 
# como se ilustra en el siguiente ejemplo:  

# >>> d1={"a":5,"b":10,"c":12,"d":11,"e":15,"f":20,
#         "g":15,"h":9,"i":7,"j":2}
# >>> histograma_horizontal(d1)
# a: *****
# b: **********
# c: ************
# d: ***********
# e: ***************
# f: ********************
# g: ***************
# h: *********
# i: *******
# j: **
#
# Nota: imprimir las barras, de arriba a abajo, en el orden que determina la
#         función "sorted" sobre las claves 
# ---------------------------------------------------------------------------











# ------------
# EJERCICIO 14
# ------------
# Con la misma entrada que el ejercicio anterior, definir una función
# histograma_vertical(d) que imprime el mismo histograma pero con las barras
# en vertical. 

# Ejemplo:

# >>> d2={"a":5,"b":7,"c":9,"d":12,"e":15,"f":20,
#         "g":15,"h":9,"i":7,"j":2}
# >>> histograma_vertical(d2)
#           *         
#           *         
#           *         
#           *         
#           *         
#         * * *       
#         * * *       
#         * * *       
#       * * * *       
#       * * * *       
#       * * * *       
#     * * * * * *     
#     * * * * * *     
#   * * * * * * * *   
#   * * * * * * * *   
# * * * * * * * * *   
# * * * * * * * * *   
# * * * * * * * * *   
# * * * * * * * * * * 
# * * * * * * * * * * 
# a b c d e f g h i j

# Nota: imprimir las barras, de izquierda a derecha, en el orden que determina la
#         función "sorted" sobre las claves 
# ---------------------------------------------------------------------------











# ------------
# EJERCICIO 15
# ------------
#
# 
# Supongamos que tenemos almacenada, usando diccionario, la información sobre
# el grupo de los alumnos de un curso. Para ello, usamos como clave el nombre
# de los alumnos de un grupo y como valor el grupo que tienen asignado. 

# 1) Definir una función alumnos_grupo(d) que a partir de un diccionario
# de ese tipo, devuelve otro diccionario cuyas claves son los nombres de los
# grupos y cuyo valor asignado a cada clave es la lista los alumnos que
# forman parte del grupo.

# Ejemplos:

# >>> alum={"Juan":"G1", "Rosa":"G2"  , "Joaquín":"G1"   ,"Carmen":"G2"  , 
#           "Isabel":"G1" , "Rocío":"G3" , "Bernardo":"G3", "Jesús":"G2"}
# >>> grupos=alumnos_grupo(alum)
# >>> grupos
# {'G3': ['Rocío', 'Bernardo'], 'G2': ['Jesús', 'Carmen', 'Rosa'], 
#  'G1': ['Isabel', 'Juan', 'Joaquín']}


# 2) Definir una función nuevo_alumno(dict_n,dict_g,nombre,grupo) tal que
# supuesto que dict_n y dict_g son dos variables conteniendo respectivamente
# el grupo de cada alumno y los alumnos de cada grupo, introduce un nuevo
# alumno con su nombre y grupo, modificando adecuadamente tanto dict_n como
# dict_g. Si el alumno ya está en los diccionarios, modificar el dato
# existente (en ese caso, si además el grupo que se quiere asignar no coincide
# que el que ya tiene se mostrará un mensaje de advertencia). Si se asigna un
# grupo que no está dado de alta, la correspondiente entrada se debe añadir al
# diccionario de grupos.

# Ejemplos:

# >>> nuevo_alumno(alum,grupos,"Bernardo","G3")
# Nog actualizado. El alumno Bernardo ya está dado de alta en el grupo G3
# >>> alum
# {'Isabel': 'G1', 'Jesús': 'G2', 'Rocío': 'G3', 'Juan': 'G1', 'Carmen': 'G2', 
#  'Rosa': 'G2', 'Joaquín': 'G1', 'Bernardo': 'G3'}
# >>> nuevo_alumno(alum,grupos,"Bernardo","G1")
# El alumno Bernardo ya está dado de alta. Se cambia al grupo G1
# >>> alum
# {'Isabel': 'G1', 'Jesús': 'G2', 'Rocío': 'G3', 'Juan': 'G1', 'Carmen': 'G2', 
#  'Rosa': 'G2', 'Joaquín': 'G1', 'Bernardo': 'G1'}
# >>> grupos
# {'G3': ['Rocío'], 'G2': ['Jesús', 'Carmen', 'Rosa'], 
#  'G1': ['Isabel', 'Juan', 'Joaquín', 'Bernardo']}
# >>> nuevo_alumno(alum,grupos,"Ana","G3")
# Nuevo alumno Ana. Incluido en el grupo G3
# >>> nuevo_alumno(alum,grupos,"Juan","G4")
# El alumno Juan ya está dado de alta. Se cambia al grupo G4
# >>> alum
# {'Isabel': 'G1', 'Jesús': 'G2', 'Rocío': 'G3', 'Ana': 'G3', 'Juan': 'G4', 'Carmen': 'G2', 
#  'Rosa': 'G2', 'Joaquín': 'G1', 'Bernardo': 'G1'}
# >>> grupos
# {'G4': ['Juan'], 'G3': ['Rocío', 'Ana'], 'G2': ['Jesús', 'Carmen', 'Rosa'], 
#  'G1': ['Isabel', 'Joaquín', 'Bernardo']}
# --------------------------------------------------------------------------

























# ------------
# EJERCICIO 16
# ------------


# (J. Zelle) Supongamos que queremos simular la trayectoria de un proyectil
# que se dispara en un punto dado a una determinada altura inicial. El disparo
# se realiza hacia adelante con una velocidad inicial y con un determinado
# ángulo. Inicialmente el proyectial avanzará subiendo pero por la fuerza de
# la gravedad en un momento dado empezará a bajar hasta que aterrice. Por
# simplificar, supondremos que no existe rozamiento ni resistencia del viento.

# Diseñar una clase Proyectil que sirva representar el estado del proyectil en
# un instante de tiempo dado. Para ello, necsitamos al menos los siguientes
# atributos de datos:
# - Distancia recorrida (en horizontal)
# - Altura
# - Velocidad horizontal
# - Velocidad vertical

# Además, incluir los siguientes tres métodos:
# - actualiza(t): actualiza la posición y la velocidad del proyectil tras t
#   segundos
# - obtén_posx(): devuelve la distancia horizontal recorrida 
# - obtén_posy(): devuelve la distancia vertical recorrida 

# Una vez definida la clase Proyectil, usarla para definir una función 
#    aterriza(altura, velocidad, ángulo, intervalo)
# que imprimirá por pantalla las distintas posiciones por las que pasa un
# proyectil que se ha disparado con una velocidad, un ángulo (en grados) 
# y una áltura inicial dada. Se mostrará la posición del proyectil 
# en cada intervalo de tiempo, hasta que aterriza.
# Además se mostrará la altura máxima que ha alcanzado, cuántos intervalos de
# tiempo ha tardado en aterrizar y el alcance que ha tenido 

# Indicaciones:

# - Si el proyectil tiene una velocidad inicial v y se lanza con un ángulo
#   theta, las componentes horizontal y vertical de la velocidad inicial son
#   v*cos(theta) y v*sen(theta), respectivamente.
# - La componente horizontal de la velocidad, en ausencia de rozamiento 
#   y viento, podemos suponer que permanece constante.
# - La componente vertical de la velocidad cambia de la siguiente manera
#   tras un instante t: si vy0 es la velocidad vertical al inicio del
#   intervalo, entonces al final del intervalo tiene una velocidad 
#   vy1=vy0-9.8*t, debido a la gravedad de la tierra.
# - En ese caso, si el proyectil se encuentra a una altura h0, tras un
#   intervalo t de tiempo se encontrará a una altura h1=h0 - vm*t, donde vm es la
#   media entre las anteriores vy0 y vy1. 

# Ejemplo:

# >>> aterriza(30,45,20,0.01)
# Proyectil en posición(0.0,30.0)
# Proyectil en posición(0.4,30.2)
# Proyectil en posición(0.8,30.3)
# Proyectil en posición(1.3,30.5)
# Proyectil en posición(1.7,30.6)
# Proyectil en posición(2.1,30.8)
# Proyectil en posición(2.5,30.9)
#           ·······
# ·······SALIDA OMITIDA ·······
#           ·······
# Proyectil en posición(187.3,2.0)
# Proyectil en posición(187.8,1.7)
# Proyectil en posición(188.2,1.5)
# Proyectil en posición(188.6,1.2)
# Proyectil en posición(189.0,0.9)
# Proyectil en posición(189.4,0.6)
# Proyectil en posición(189.9,0.3)
# Proyectil en posición(190.3,0.0)

# Tras 451 intervalos de 0.01 segundos (4.51 segundos) el proyectil ha aterrizado.
# Ha recorrido una distancia de 190.7 metros
# Ha alcanzado una altura máxima de 42.1 metros
# -----------------------------------------------------------------------------






# ------------
# EJERCICIO 17
# ------------

# Apartado 17.1
# -------------

# Supongamos que queremos gestionar los alumnos de una titulación, con las
# asignaturas en las que están matriculados, y las notas que tienen. Para
# ello, se pide implementar una clase Alumno, con las siguintes
# características: 

# - El constructor de la clase recibe como argumentos el nombre del alumno y
#   una lista de las asignaturas que matricula inicialmente (sin nota). Por
#   simplificar, supondremos que el nombre es un string con un nombre de pila
#   y dos apellidos, y que la asignatura viene dada por sus siglas. Se supone
#   que ni el nombre de pila ni los apellidos son compuestos.

# - El nombre debe ser un atributo de datos de la clase. Además, incluir cualquier 
#   atributo que pudiera ser necesario para mantener la información sobre las 
#   asignaturas en las que está matriculado el alumno, y la nota, si la tuviera 
#   (si aún no tiene nota de una asignatura, asignar el valor "-")

# - Los métodos de la clase son los siguientes:
#    * Método __repr__, que devuelve simplemente el nombre del alumno
#    * Método pon_nota, que recibe una asignatura y una nota, y anota
#      al alumno la nota de esa asignatura. Si el alumno no está matriculado
#      en esa asignatura, el método debe devolver la excepción 
#      AsignaturaNoMatriculada, que se define más abajo.
#    * Método consulta_nota, que recibe una asignatura y devuelve la nota que
#      ese alumno tiene en la asigntura. Si el alumno no está matriculado
#      en esa asignatura, el método debe devolver la excepción 
#      AsignaturaNoMatriculada, que se define así:
#         class AsignaturaNoMatriculada(Exception):
#             pass
#    * Método añade_asignatura que recibe una asignatura, y añade esa
#      asignatura al alumno. Si la asignatura ya la tiene el alumno, no hacer
#      nada.
#    * Método asignaturas_matriculadas, que devuelve la lista de asignaturas
#      matriculadas del alumno
#    * Método media_expediente, que recibiendo el plan de estudios del alumno,
#      devuelve la nota media del alumno (ponderada por número de créditos de
#      cada asignatura). El plan de estudios es un diccionario cuyas claves
#      son todas las asignaturas, y el valor asociado a cada clave es el
#      número de creditos de la asignatura (ver ejemplo más abajo). Si una
#      asignatura no está matriculada o evaluada, se considera que está
#      puntuada con un cero.


# Ejemplos:

# >>> alumno1=Alumno("Antonio Ruiz Santos", ["DGPDS1","DGPDS2","IPPPD","FEST","AEM","APCD"])

# >>> alumno1.nombre
# 'Antonio Ruiz Santos'

# >>> alumno1 # Aquí se llamaría al método __repr__
# Antonio Ruiz Santos

# >>> alumno1.consulta_nota("IPPPD")
# '-'

# >>> alumno1.pon_nota("IPPPD",8.9)

# >>> alumno1.consulta_nota("IPPPD")
# 8.9

# >>> alumno1.consulta_nota("ML1")
# Traceback (most recent call last):

#   File "<ipython-input-41-33cce032017f>", line 1, in <module>
#     alumno1.consulta_nota("ML1")

#   File ".......", line 26, in consulta_nota
#     raise AsignaturaNoMatriculada("Asignatura no matriculada para este alumno")

# AsignaturaNoMatriculada: Asignatura no matriculada para este alumno

# >>> alumno1.añade_asignatura("ML1")

# >>> alumno1.consulta_nota("ML1")
# '-'

# >>> alumno1.pon_nota("ML1",6.3)

# >>> alumno1.consulta_nota("ML1")
# 6.3

# >>> alumno1.asignaturas_matriculadas()
# ['APCD', 'DGPDS1', 'ML1', 'IPPPD', 'DGPDS2', 'AEM', 'FEST']

plan_de_estudios_MDS={"DGPDS1":3,"DGPDS2":6,"IPPPD":4,"FEST":4,"AEM":6,"APCD":4,
                   "APBD":5,"ML1":5,"ML2":5,"TMO":4,"ICSR":3,"MDTE":3,"DSBI":3,
                   "PLNCD1":2,"PLNCD2":2,"VD":2,"VI":2,"TFM":6} 

# >>> alumno1.media_expediente(plan_de_estudios_MDS)
# 0.9724637681159419

# ------------------------------------------------------------















# Apartado 17.2
# -------------


# Supongamos que tenemos un archivo de texto en los que cada línea corresponde
# a un alumno con sus asignaturas y notas, con el siguiente formato:

# NOMBRE APELLIDO1 APELLIDO2 A1 N1 A2 N2 .... An Nn

# Por ejemplo, podríamos tener un archivo alumno_notas.txt con las siguientes
# líneas:

# Juan Pérez Quirós DGPDS1 7.4 DGPDS2 8.4 IPPPD 9.1 FEST 7.5 AEM 6.2 APCD 8.2 APBD 5.3 ML1 8.8 ML2 7.5 TMO 8.7 ICSR 6.1 MDTE 7.3 DSBI 10.0 PLNCD1 5.0 PLNCD2 6.2 VD 6.4 VI 7.1 TFM 8.5
# María González Peña DGPDS1 5.4 DGPDS2 9.3 IPPPD 8.7 FEST 7.6 APCD 9.2 APBD 6.6 ML1 .8 ML2 7.7 TMO 5.2 MDTE 5.3 DSBI 8.2 PLNCD1 6.0 PLNCD2 9.2 VD 6.4 VI 7.1 
# Pedro Moncada Escobar DGPDS1 6.4 IPPPD 9.5 FEST 7.8 AEM 5.2 APCD 7.2 APBD 5.8 ML1 8.8 TMO 7.2 ICSR 8.8 DSBI 5.0 PLNCD1 7.0 VD 8.4 VI 6.1 
# Salvador Gutiérrez Sánchez DGPDS1 7.7 DGPDS2 8.0 IPPPD 7.3 FEST 7.9 AEM 8.2 APCD 8.6 APBD 5.3 TMO 5.2 ICSR 8.1 MDTE 5.3 PLNCD1 5.3 PLNCD2 7.5 VD 8.4
# Rocío Cotán Sánchez DGPDS2 8.2 FEST 7.1 APCD 6.2 ML1 5.8 ML2 7.9 TMO 5.2 ICSR 9.1 MDTE 6.3 DSBI 6.6 PLNCD1 5.6 PLNCD2 6.5 VI 6.1 TFM 9.5
# Gabriel Mejías Cifuentes DGPDS1 6.9 DGPDS2 7.3 IPPPD 9.0 FEST 6.5 AEM 6.5 APBD 5.7 ML1 7.8 ICSR 8.1 MDTE 5.3 PLNCD1 5.1 PLNCD2 8.0 
# Josefa Cabrera León DGPDS1 7.4 DGPDS2 8.4 IPPPD 9.1 FEST 7.5 

# Por simplificar, ni los nombres de pila ni los apellidos serán compuestos.

# Se pide definir una función lee_notas(archivo), que recibiendo el nombre del
# archivo, devuelva una lista de objetos de la clase Alumno, cada uno
# conteniendo toda la información de la correspondiente línea del archivo de
# texto. 

# Ejemplo:

# >>> lista_alumnos=lee_notas("alumno_notas.txt")

# >>> lista_alumnos
# [Juan Pérez Quirós,
#  María González Peña,
#  Pedro Moncada Escobar,
#  Salvador Gutiérrez Sánchez,
#  Rocío Cotán Sánchez,
#  Gabriel Mejías Cifuentes,
#  Josefa Cabrera León]

# >>> lista_alumnos[2].nombre
# 'Pedro Moncada Escobar'

# >>> lista_alumnos[2].consulta_nota("APCD")
# 7.2

# >>> lista_alumnos[2].consulta_nota("DSBI")
# 5.0

# >>> lista_alumnos[2].consulta_nota("TFM")
# Traceback (most recent call last):

#   File "<ipython-input-56-b068fc897dbd>", line 1, in <module>
#     lista_alumnos[2].consulta_nota("TFM")

#   File "......", line 26, in consulta_nota
#     raise AsignaturaNoMatriculada("Asignatura no matriculada para este alumno")

# AsignaturaNoMatriculada: Asignatura no matriculada para este alumno


# >>> lista_alumnos[3].asignaturas_matriculadas() 
# ['DGPDS1',
#  'PLNCD2',
#  'PLNCD1',
#  'ICSR',
#  'IPPPD',
#  'DGPDS2',
#  'AEM',
#  'VD',
#  'TMO',
#  'APCD',
#  'APBD',
#  'MDTE',
#  'FEST']

# ------------------------------------------------------------











# Apartado 17.3
# -------------

# Definir una función mejor_expediente(lista_alumnos, plan_de_estudiso), que
# recibiendo como entrada:
#  - Una lista lista_de_alumnos de objetos de la clase Alumno
#  - Un diccionario plan_de_estudios, que asigna a cada asignatura del plan de
#    estudios su número de créditos. 
# devuelve el objeto de la clase Alumno (o lista de objetos, si hay más de uno), con la mejor
# nota media

# Ejemplo:

# >>> mejor_expediente(lista_alumnos,plan_de_estudios_MDS)
# Juan Pérez Quirós

# ------------------------------------------------------------









# ------------
# EJERCICIO 18
# ------------

# Realizar los siguientes ejercicicios con arrays de numpy

# 18.1
# ----
# Generar una matrix aleatoria 3x4 y normalizar sus componentes (escalándolos)
# de manera que el menor sea 0 y el mayor 1.



# 18.2
# ----

# Construir una matriz 3x4 cuyas tres filas son los elementos del 0 al 3




# 18.3
# ----

# En una matriz 20x20, obtener la matriz que se obtiene al extraer las
# filas de índice par y las columnas de índice impar 




# 18.4
# ----

# Dada una matriz A de tamaño 20x10 y otra B de tamaño 20x1 con valores entre
# 0 y 100, devolver una matriz conteniendo las filas que corresponden a las
# posiciones de B que son mayores de 50 














