#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2022-23
# PARTE I del trabajo práctico: Implementación de regresión logística
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Ramón Soria
# NOMBRE: Guillermo
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS: Molina Torres
# NOMBRE: Daniel
# ----------------------------------------------------------------------------

# ANOTACIÓN: No hemos tenido lugar de preparar un una impresión elegante de
# los test para cada ejercicio, pero puede filtrar las líneas a descomentar
# mediante las notaciones siguiente:
# -- TEST EJ --

# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La discusión 
# y el intercambio de información de carácter general con los compañeros se permite, 
# pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. En particular no se 
# permiten implementaciones obtenidas con HERRAMIENTAS DE GENERACIÓN AUTOMÁTICA DE CÓDIGO. 
# Si tienen dificultades para realizar el ejercicio, consulten con el profesor. 
# En caso de detectarse plagio (previamente con aplicaciones anti-plagio o durante 
# la defensa, si no se demuestra la autoría mediante explicaciones convincentes), 
# supondrá una CALIFICACIÓN DE CERO en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 

# * SE RECOMIENDA y SE VALORA especialmente usar numpy. Las implementaciones 
#   saldrán mucho más cortas y eficientes, y se puntuarÁn mejor.   

import numpy as np

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulo sklearn). Todos los datos se
# cargan en arrays de numpy:

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------

import carga_datos as cd
import numpy as np

def particion_entr_prueba(X, y, test=0.20):
    
    # Comenzamos inicializando los conjuntos de entrenamiento y prueba (vacíos)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    # Obtenemos los valores únicos de clasificación aplicando el método unique
    clases = np.unique(y)

    for c in clases:
        # Para cada clase, obtenemos sus índices correspondientes
        indice_clase = np.where(y == c)[0]
   
        # Calculamos el tamaño del conjunto de prueba
        tam_test = int(len(indice_clase) * test)

        # Aplicamos el método shuffle para aleatorizar los índices
        np.random.shuffle(indice_clase)
        indice_test = indice_clase[:tam_test]
        
        # Usamos los índices restantes para el entrenamiento
        indice_entr = np.setdiff1d(indice_clase, indice_test)
        
        # Agregamos los datos a sus respectivos conjuntos
        X_train.append(X[indice_entr])
        y_train.append(y[indice_entr])
        X_test.append(X[indice_test])
        y_test.append(y[indice_test])
    
    # Reconvertimos los conjuntos a un array unidimensional
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    
    return X_train, X_test, y_train, y_test


# -- TEST EJ 1 --
Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(cd.X_votos,cd.y_votos,test=1/3)
# print(np.unique(cd.y_votos,return_counts=True))
# #  (array([0, 1]), array([168, 267]))
# print(np.unique(ye_votos,return_counts=True))
# #  (array([0, 1]), array([112, 178]))
# print(np.unique(yp_votos,return_counts=True))
# #  (array([0, 1]), array([56, 89]))
# print("\n")

Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(cd.X_cancer,cd.y_cancer,test=0.2)
# print(np.unique(cd.y_cancer,return_counts=True))
# #(array([0, 1]), array([212, 357]))
# print(np.unique(yev_cancer,return_counts=True))
# # (array([0, 1]), array([170, 286]))
# print(np.unique(yp_cancer,return_counts=True))
# # (array([0, 1]), array([42, 71])) 
# print("\n")

Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
# print(np.unique(ye_cancer,return_counts=True))
# # (array([0, 1]), array([170, 286]))
# print(np.unique(yv_cancer,return_counts=True))
# # (array([0, 1]), array([170, 286]))
# print("\n")

Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(cd.X_credito,cd.y_credito,test=0.4)
# print(np.unique(cd.y_credito,return_counts=True))
# # (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
# #  array([202, 228, 220]))
# print(np.unique(ye_credito,return_counts=True))
# # (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
# #  array([121, 137, 132]))
# print(np.unique(yp_credito,return_counts=True))
# # (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
# #  array([81, 91, 88]))

## ---------- 

# ===========================
# EJERCICIO 2: NORMALIZADORES
# ===========================

# En esta sección vamos a definir dos maneras de normalizar los datos. De manera 
# similar a como está diseñado en scikit-learn, definiremos un normalizador mediante
# una clase con un metodo "ajusta" (fit) y otro método "normaliza" (transform).


# ---------------------------
# 2.1) Normalizador standard
# ---------------------------

# Definir la siguiente clase que implemente la normalización "standard", es 
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1. 

# En particular, definir la clase: 

class NormalizadorStandard():

    def __init__(self):
        #Inicializamos variable media y desviación típica con None
        self.media = None
        self.desviacion=None

    def ajusta(self,X):
        #Calculamos  la media y desviación típica de X de cada columna de la matriz X
        self.media = np.mean(X, axis=0)
        self.desviacion = np.std(X, axis=0)

    def normaliza(self,X):
        #Verificamos si la media y la desviación típica no están ajustadas
        if self.media is None or self.desviacion is None:
            raise NormalizadorNoAjustado("Normalizador no ajustado")
        #Calculamos X_normalizado aplicando su ecuación correspondiente
        X_normalizado = (X - self.media) / self.desviacion
        return X_normalizado


# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método 
# normaliza devuelve el correspondiente conjunto de datos normalizados. 

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorNoAjustado(Exception): pass

normStd_cancer = NormalizadorStandard()
normStd_cancer.ajusta(Xe_cancer)

# -- TEST EJ 2.1 --
Xe_cancer_n = normStd_cancer.normaliza(Xe_cancer)
# print(round(np.mean(Xe_cancer_n)))
# print(round(np.std(Xe_cancer_n)))
Xv_cancer_n = normStd_cancer.normaliza(Xv_cancer)
Xp_cancer_n = normStd_cancer.normaliza(Xp_cancer)

# print("Xe_cancer_n: ", Xe_cancer_n, "\n",
#                 "Xv_cancer_n: ", Xv_cancer_n, "\n",
#                 "Xp_cancer_n: ", Xp_cancer_n, "\n")



# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser 
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n, 
# ni con Xp_cancer_n. 



# ------ 


# ------------------------
# 2.2) Normalizador MinMax
# ------------------------

# Hay otro tipo de normalizador, que consiste en asegurarse de que todas las
# características se desplazan y se escalan de manera que cada valor queda entre 0 y 1. 
# Es lo que se conoce como escalado MinMax

# Se pide definir la clase NormalizadorMinMax, de manera similar al normalizador 
# del apartado anterior, pero ahora implementando el escalado MinMax.

# Ejemplo:

# >>> normminmax_cancer=NormalizadorMinMax()
# >>> normminmax_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_m=normminmax_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_m=normminmax_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_m=normminmax_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, los máximos y mínimos de las columnas de Xe_cancer_m
#  deben ser 1 y 0, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_m,
# ni con Xp_cancer_m. 


# ------ 

class NormalizadorMinMax():

    def __init__(self):
        #Inicializamos variable mínimo y máximo con None
        self.minimo = None
        self.maximo = None

    def ajusta(self,X):
        #Calculamos el mínimo y el maximo de X de cada columna de la matriz X (usando numpy)
        self.minimo = np.min(X, axis=0)
        self.maximo = np.max(X, axis=0)

    def normaliza(self,X):
        #Verificamos siel mínimo y el máximo no estan ajustadas
        if self.minimo is None or self.maximo is None:
            raise NormalizadorNoAjustado("Normalizador no ajustado")
        #Calcular X_normalizado aplicando su ecuación correspondiente
        X_normalizado = (X - self.minimo) / (self.maximo - self.minimo)
        return X_normalizado

normMM_cancer=NormalizadorMinMax()
normMM_cancer.ajusta(Xe_cancer)

# -- TEST EJ 2.2 --
Xe_cancer_m=normMM_cancer.normaliza(Xe_cancer)
Xv_cancer_m=normMM_cancer.normaliza(Xv_cancer)
Xp_cancer_m=normMM_cancer.normaliza(Xp_cancer)
# print(Xe_cancer_m)
# print(Xv_cancer_m)
# print(Xp_cancer_m)




# ===========================================
# EJERCICIO 3: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# En este ejercicio se propone la implementación de un clasificador lineal 
# binario basado regresión logística (mini-batch), con algoritmo de entrenamiento 
# de descenso por el gradiente mini-batch (para minimizar la entropía cruzada).


# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
#                 batch_tam=64):

#         .....
        
#     def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
#                     early_stopping=False,paciencia=3):

#         .....        

#     def clasifica_prob(self,ejemplos):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......



# * El constructor tiene los siguientes argumentos de entrada:



#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula: 
#        rate_n= (rate_0)*(1/(1+n)) 
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False. 
#  
#   + batch_tam: tamaño de minibatch


# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente. Las dos clases del problema 
#        son las que aparecen en el array y, y se deben almacenar en un atributo 
#        self.clases en una lista. La clase que se considera positiva es la que 
#        aparece en segundo lugar en esa lista.
#     
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se 
#       usarán en el caso de activar el parámetro early_stopping. Si son None (valor 
#       por defecto), se supone que en el caso de que early_stopping se active, se 
#       consideraría que Xv e yv son resp. X e y.

#     + n_epochs es el número máximo de epochs en el entrenamiento. 

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el 
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada 
#       del modelo respecto del conjunto de entrenamiento, y su rendimiento 
#       (proporción de aciertos). Igualmente para el conjunto de validación, si lo
#       hubiera. Esta opción puede ser útil para comprobar 
#       si el entrenamiento  efectivamente está haciendo descender la entropía
#       cruzada del modelo (recordemos que el objetivo del entrenamiento es 
#       encontrar los pesos que minimizan la entropía cruzada), y está haciendo 
#       subir el rendimiento.
# 
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor entropía conseguida hasta el momento
#       en el conjunto de validación 
#       NOTA: esto se suele hacer con mecanismo de  "callback" para recuperar el mejor modelo, 
#             pero por simplificar implementaremos esta versión más sencilla.  
#        



# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase positiva.       
    

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

        
  

# RECOMENDACIONES: 


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer 
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande. 

# + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.     

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

#   from scipy.special import expit    
#
#   def sigmoide(x):
#      return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)

# >>> lr_cancer.clasifica(Xp_cancer_n[24:27])
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26 

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])



# Para calcular el rendimiento de un clasificador sobre un conjunto de ejemplos, usar la 
# siguiente función:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]

# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:
    
# >>> rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
# 0.9824561403508771

# >>> rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)
# 0.9734513274336283




# Ejemplo con salida_epoch y early_stopping:

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento EC: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    EC: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento EC: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    EC: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento EC: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento EC: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento EC: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento EC: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento EC: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la entropía cruzada obtenida en el epoch 3 
# sobre el conjunto de validación, ésta no se ha mejorado. 

# -----------------------------------------------------------------

import numpy as np
from scipy.special import expit

class ClasificadorNoEntrenado(Exception):
    pass

class RegresionLogisticaMiniBatch():
    def __init__(self, rate=0.1, rate_decay=False, n_epochs=100, batch_tam=64):
        #Inicializamos el constructor y sus variables necesarias
        self.rate = rate  
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.pesos = None # Inicializamos pesos y clases en el método init
        self.clases = []        # ya que será útil su uso en distintas funciones
                                                    # y su valor debe almacenarse a modo de var estática 

    def sigmoide(self, x):
        return expit(x)

    def entropia_cruzada(self, y, y_pred):
        return np.sum(np.where(y == 1, -np.log(y_pred), -np.log(1 - y_pred)))

    def entrena(self, X, y, Xv=None, yv=None, n_epochs=100, salida_epoch=False, early_stopping=False, paciencia=3):

        #Obtenemos el conjunto clases únicas e inicializamos los pesos de manera aleatoria.
        self.clases = np.unique(y)
        # Creamos un dicc donde las claves son los nombres de las clases y los valores
        # son los índices correspondientes a cada clase en la lista
        dicc_clases = {nombre_clase: i for i, nombre_clase in enumerate(self.clases)}
        
        self.pesos = np.random.randn(X.shape[1])

        if Xv is None:
            Xv = X
            yv = y

        # Declaramos la mejor pérdida a infinito
        # De esta forma, si la pérdida actual es menor, aseguramos que se actualice
        mejor_perdida = float('inf')
        # Asímismo, contPaciencia llevará el conteo de epochs consecutivos sin mejora en la pérdida.
        contPaciencia = 0

        for epoch in range(n_epochs):
            if(self.rate_decay==True):
                rate_n = (self.rate)*(1/(1+epoch)) # self.rate decrecerá en cada iter
            else:
                rate_n = self.rate 

            indices = np.arange(len(X))
            np.random.shuffle(indices)

            # Recorremos X saltando de batch en batch
            for i in range(0, len(X), self.batch_tam):
                indices_batch = indices[i:(i+self.batch_tam)] # Indices del batch seleccionado
                X_batch = X[indices_batch] # Datos de X que hacen referencia a esos indices

                # Recogemos en y_batch los valores para cada clase del dicc
                y_batch = np.array([dicc_clases[nombre_clase] for nombre_clase in y[indices_batch]])

                # En y_pred tendremos la predicción de clase para el X_batch actual
                y_pred = self.sigmoide(np.dot(X_batch, self.pesos))
            
                # Calculamos el Error de predicción del modelo
                err_pred = y_pred - y_batch.astype(float)
                gradiente = np.dot(err_pred, X_batch)
                

                #Actualizamos los pesos
                self.pesos -= rate_n * gradiente

            if salida_epoch:
                # Si se cumple la condición, mostramos la EC del entrenamiento y validación de cada epoca 
                pred_entr = self.clasifica_prob(X)
                perdidaEntr = self.entropia_cruzada(y, pred_entr)
                rend_entr = rendimiento(self, X, y)

                y_pred_val = self.clasifica_prob(Xv)
                perdida = self.entropia_cruzada(yv, y_pred_val)
                rend_val = rendimiento(self, Xv, yv)

                print(f"Epoch {epoch + 1}:")
                print(f"  en entrenamiento EC: {perdidaEntr:.4f}, rendimiento: {rend_entr:.4f}")
                print(f"  en validación EC: {perdida:.4f}, rendimiento: {rend_val:.4f}")

                if early_stopping:
                    #Si se cumple esta condición, actualizamos el valor de perdida
                    # y establecemos la paciencia a 0.
                    if perdida < mejor_perdida:
                        mejor_perdida = perdida
                        contPaciencia = 0
                    else: 
                    # En caso contrario, aumentamos la paciencia
                        contPaciencia += 1
                    # Finalmente si la paciencia supera el limite establecido, se produce "PARADA TEMPRANA" (earlystopping)
                    if contPaciencia >= paciencia:
                        print("PARADA TEMPRANA")
                        break
                

    def clasifica_prob(self, ejemplos):
        # Si no estan declarados los pesos, salta una interrupción, indicando de que no está entrenado el modelo
        if self.pesos is None:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado")
        # Calculamos la probabilidad de pertenencia respecto a las clases
        y_pred = self.sigmoide(np.dot(ejemplos, self.pesos))
        return y_pred

    def clasifica(self, ejemplo):
        # Si no estan declarados los pesos, salta una interrupción, indicando de que no está entrenado el modelo
        if self.pesos is None:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado")
        # Calculamos la probabilidad de pertenencia respecto a las clases
        y_pred = self.sigmoide(np.dot(ejemplo, self.pesos))
        # Devolvemos un array que contiene las clases asignadas a cada predicción 
        y_pred_clase = np.where(y_pred >= 0.5, self.clases[1], self.clases[0])
        return y_pred_clase

# -- TEST EJ 3 --
lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
# lr_cancer.entrena(Xe_cancer_n, ye_cancer, Xv_cancer_n, yv_cancer, salida_epoch=True, early_stopping=True)
# print(lr_cancer.clasifica(Xp_cancer_n[24:27]))
# print(yp_cancer[24:27])
# print(lr_cancer.clasifica_prob(Xp_cancer_n[24:27]))
# print(rendimiento(lr_cancer,Xe_cancer_n,ye_cancer))
# print(rendimiento(lr_cancer,Xp_cancer_n,yp_cancer))
# ------------------------------------------------------------------------------


# =================================================
# EJERCICIO 4: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================



# Este jercicio puede servir para el ajuste de parámetros en los ejercicios posteriores, 
# pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1


# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador (como por ejemplo 
# la clase RegresionLogisticaMiniBatch). El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cancer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 Xe_cancer_n,ye_cancer,n=5)

# Partición: 1. Rendimiento:0.9863013698630136
# Partición: 2. Rendimiento:0.958904109589041
# Partición: 3. Rendimiento:0.9863013698630136
# Partición: 4. Rendimiento:0.9726027397260274
# Partición: 5. Rendimiento:0.9315068493150684
# >>> 0.9671232876712328




# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones DEBEN SER ALEATORIAS Y ESTRATIFICADAS. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# >>> lr16.entrena(Xe_cancer_n,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(lr16,Xp_cancer_n,yp_cancer)
# 0.9646017699115044

#------------------------------------------------------------------------------


import numpy as np

def rendimiento_validacion_cruzada(clase_clasificador, params, X, y, Xv=None, yv=None, n=5):
    if Xv is None or yv is None:
        Xv = X
        yv = y
    
    rendimientos = []
    
    # Inicializamos los indices aleatoriamente
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Dividimos en n partes los conjuntos de entrada
    particiones_X = np.array_split(X[indices], n)
    particiones_y = np.array_split(y[indices], n)
    
    for i in range(n):
        # Asignamos en X_prueba e y_prueba la i-ésima parte dividida anteriormente
        X_prueba = particiones_X[i]
        y_prueba = particiones_y[i]

        # Realizamos la exclusión de dicha parte y almacenamos el resultado en particiones_entrenamiento
        particiones_entrenamiento_X = particiones_X[:i] + particiones_X[i+1:]
        particiones_entrenamiento_y = particiones_y[:i] + particiones_y[i+1:]
        
        # Aplanamos el array a una única dimensión
        X_entrenamiento = np.concatenate(particiones_entrenamiento_X)
        y_entrenamiento = np.concatenate(particiones_entrenamiento_y)
        
        # Llamamos al clasificador con los parámetros de entrada propuestos
        # entre llaves y entrenamos el clasificador
        clasificador = clase_clasificador(**params)
        clasificador.entrena(X_entrenamiento, y_entrenamiento)
        
        # Calculamos el rendimiento para cada partición
        rend = rendimiento(clasificador, X_prueba, y_prueba)
        rendimientos.append(rend)
        
        print(f"Partición {i+1}. Rendimiento: {rend}")
    
    # Finalmente se calcula el rendimiento medio haciendo uso de numpy (np.mean)
    rendimiento_medio = np.mean(rendimientos)
    
    return rendimiento_medio


# -- TEST EJ 4 --
# rendimiento_medio = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,{"batch_tam":16,"rate":0.01,"rate_decay":True},Xe_cancer_n,ye_cancer,n=5)
# print(f"Rendimiento medio: {rendimiento_medio}")

# lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# lr16.entrena(Xe_cancer_n,ye_cancer)
# print(rendimiento(lr16,Xp_cancer_n,yp_cancer))




# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando la regeresión logística implementada en el ejercicio 2, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam) para mejorar el rendimiento 
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones). 
# Si se ha hecho el ejercicio 4, usar validación cruzada para el ajuste 
# (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba.     

# Mostrar también, para cada conjunto de datos, un ejemplo con salida_epoch, 
# en el que se vea cómo desciende la entropía cruzada y aumenta el 
# rendimiento durante un entrenamiento.     

# ----------------------------

def clasifBin_votos():
    lr_votos = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True)

    # Para ajustar los parámetros, vamos a ir actualizando mejor_rendimiento_cancer
    # y mejor_rendimiento_cancer según el rendimiento asociado a la combinatoria de
    # parámetros que estemos evaluando (ahora declararemos los valores de params)
    params_votos = {
        'rate': [0.1, 0.01, 0.001],
        'rate_decay': [True, False],
        'batch_tam': [16, 32, 64]
    }

    mejor_rendimiento_votos = 0
    mejores_parametros_votos = {}

    # Realizamos una búsqueda de los mejores parámetros de forma clásica
    # mediante bucles for y actualizando los mejores valores
    for rate in params_votos['rate']:
        for rate_decay in params_votos['rate_decay']:
            for batch_tam in params_votos['batch_tam']:
                params = {
                    'rate': rate,
                    'rate_decay': rate_decay,
                    'batch_tam': batch_tam
                }
                rendVC = rendimiento_validacion_cruzada(
                    RegresionLogisticaMiniBatch, params, Xe_votos_n, ye_votos, Xv_votos_n, yv_votos
                )
                print(f"Parámetros: {params}, Rendimiento medio: {rendVC}")

                if rendVC > mejor_rendimiento_votos:
                    mejor_rendimiento_votos = rendVC
                    mejores_parametros_votos = params

    # Aplicamos el método entrena de RegresionLogisticaMiniBatch con los mejores parámetros obtenidos
    lr_votos = RegresionLogisticaMiniBatch(**mejores_parametros_votos)
    lr_votos.entrena(Xe_votos_n, ye_votos)

    # Evaluamos el conjunto de prueba
    rendimiento_prueba_votos = rendimiento(lr_votos, Xe_votos_n, ye_votos)
    print(f"Rendimiento en conjunto de prueba: {rendimiento_prueba_votos}")

Xev_votos,Xp_votos,yev_votos,yp_votos=particion_entr_prueba(cd.X_votos,cd.y_votos,test=1/3)
Xe_votos,Xv_votos,ye_votos,yv_votos=particion_entr_prueba(Xev_votos,yev_votos,test=1/3)

normst_votos = NormalizadorStandard()
normst_votos.ajusta(Xe_votos)
Xe_votos_n = normst_votos.normaliza(Xe_votos)
Xv_votos_n = normst_votos.normaliza(Xv_votos)

def clasifBin_cancer():
    lr_cancer = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True)

    # Para ajustar los parámetros, vamos a ir actualizando mejor_rendimiento_cancer
    # y mejor_rendimiento_cancer según el rendimiento asociado a la combinatoria de
    # parámetros que estemos evaluando (ahora declararemos los valores de params)
    params_cancer = {
        'rate': [0.1, 0.01, 0.001],
        'rate_decay': [True, False],
        'batch_tam': [16, 32, 64]
    }

    mejor_rendimiento_cancer = 0
    mejores_parametros_cancer = {}

    # Realizamos una búsqueda de los mejores parámetros de forma clásica
    # mediante bucles for y actualizando los mejores valores
    for rate in params_cancer['rate']:
        for rate_decay in params_cancer['rate_decay']:
            for batch_tam in params_cancer['batch_tam']:
                params = {
                    'rate': rate,
                    'rate_decay': rate_decay,
                    'batch_tam': batch_tam
                }
                rendVC = rendimiento_validacion_cruzada(
                    RegresionLogisticaMiniBatch, params, Xe_cancer_n, ye_cancer, Xv_cancer_n, yv_cancer
                )
                print(f"Parámetros: {params}, Rendimiento medio: {rendVC}")

                if rendVC > mejor_rendimiento_cancer:
                    mejor_rendimiento_cancer = rendVC
                    mejores_parametros_cancer = params

    # Aplicamos el método entrena de RegresionLogisticaMiniBatch con los mejores parámetros obtenidos
    lr_cancer = RegresionLogisticaMiniBatch(**mejores_parametros_cancer)
    lr_cancer.entrena(Xe_cancer_n, ye_cancer)

    # Evaluamos el conjunto de prueba
    rendimiento_prueba_cancer = rendimiento(lr_cancer, Xe_cancer_n, ye_cancer)
    print(f"Rendimiento en conjunto de prueba: {rendimiento_prueba_cancer}")

Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(cd.X_cancer,cd.y_cancer,test=0.2)
Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

normst_cancer = NormalizadorStandard()
normst_cancer.ajusta(Xe_cancer)
Xe_cancer_n = normst_cancer.normaliza(Xe_cancer)
Xv_cancer_n = normst_cancer.normaliza(Xv_cancer)

def clasifBin_imdb():
    lr_imdb = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True)

    # Para ajustar los parámetros, vamos a ir actualizando mejor_rendimiento_cancer
    # y mejor_rendimiento_cancer según el rendimiento asociado a la combinatoria de
    # parámetros que estemos evaluando (ahora declararemos los valores de params)
    params_imdb = {
        'rate': [0.1, 0.01, 0.001],
        'rate_decay': [True, False],
        'batch_tam': [16, 32, 64]
    }

    mejor_rendimiento_imdb = 0
    mejores_parametros_imdb = {}

    # Realizamos una búsqueda de los mejores parámetros de forma clásica
    # mediante bucles for y actualizando los mejores valores
    for rate in params_imdb['rate']:
        for rate_decay in params_imdb['rate_decay']:
            for batch_tam in params_imdb['batch_tam']:
                params = {
                    'rate': rate,
                    'rate_decay': rate_decay,
                    'batch_tam': batch_tam
                }
                rendVC = rendimiento_validacion_cruzada(
                    RegresionLogisticaMiniBatch, params, Xe_imdb_n, ye_imdb, Xv_imdb_n, yv_imdb
                )
                print(f"Parámetros: {params}, Rendimiento medio: {rendVC}")

                if rendVC > mejor_rendimiento_imdb:
                    mejor_rendimiento_imdb = rendVC
                    mejores_parametros_imdb = params

    # Aplicamos el método entrena de RegresionLogisticaMiniBatch con los mejores parámetros obtenidos
    lr_imdb = RegresionLogisticaMiniBatch(**mejores_parametros_imdb)
    lr_imdb.entrena(Xe_imdb_n, ye_imdb)

    # Evaluamos el conjunto de prueba
    rendimiento_prueba_imdb = rendimiento(lr_imdb, Xe_imdb_n, ye_imdb)
    print(f"Rendimiento en conjunto de prueba: {rendimiento_prueba_imdb}")

Xe_imdb,Xv_imdb,ye_imdb,yv_imdb=particion_entr_prueba(cd.X_train_imdb,cd.y_train_imdb,test=0.2)

normst_imdb = NormalizadorStandard()
normst_imdb.ajusta(Xe_imdb)
Xe_imdb_n = normst_imdb.normaliza(Xe_imdb)
Xv_imdb_n = normst_imdb.normaliza(Xv_imdb)

# -- TEST EJ 5 --
# clasifBin_votos()

# clasifBin_cancer()

# clasifBin_imdb()

# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a 
#  realizar (donde k es el número de clases).
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.  

 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)

# >>> rl_iris_ovr.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

# >>> rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9
# --------------------------------------------------------------------

import numpy as np
from scipy.special import expit

class ClasificadorNoEntrenado(Exception):
    pass

class RL_OvR():
    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        # Creamos un dicc de clasificadores binarios
        self.clasificadores = {}

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        # Obtenemos las clases únicas
        self.clases = np.unique(y)

        # Iteramos dichas clases obtenidas en los datos de entrenamiento
        # y entrena un clasificador binario para cada una de ellas
        for c in self.clases:
            y_bin = np.where(y == c, 1, 0)
            clasificador = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay, batch_tam=self.batch_tam)

            # Entrenamos el clasificador para la predección de clases
            clasificador.entrena(X, y_bin, n_epochs=n_epochs, salida_epoch=salida_epoch)
            self.clasificadores[c] = clasificador

    def clasifica(self, ejemplos):
        if not self.clasificadores:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado.")

        # Asociamos a y_pred la probabilidades de pertenencia
        y_pred = []
        for _, clasificador in self.clasificadores.items():
            y_pred.append(clasificador.clasifica_prob(ejemplos))

        # Devolvemos la clase con mayor probabilidad
        y_pred = np.array(y_pred)
        y_class = np.argmax(y_pred, axis=0)
        return self.clases[y_class]


# -- TEST EJ 6 --
Xe_iris, Xp_iris, ye_iris, yp_iris = particion_entr_prueba(cd.X_iris, cd.y_iris)
rl_iris_ovr = RL_OvR(rate=0.001, batch_tam=8)
rl_iris_ovr.entrena(Xe_iris, ye_iris)

# print(rendimiento(rl_iris_ovr, Xe_iris, ye_iris))
# print(rendimiento(rl_iris_ovr, Xp_iris, yp_iris))

# --------------------------------







# =================================
# EJERCICIO 7: CODIFICACIÓN ONE-HOT
# =================================


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una 
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles 
# valores del atributo. El valor i-ésimo del atributo se convierte en k valores
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.  

# Por ejemplo, si un atributo tiene tres posibles valores "a", "b" y "c", ese atributo 
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)    

# Definir una función:    
    
#     codifica_one_hot(X) 

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X.Por simplificar supondremos 
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto 
# hay que codificarlos todos.

# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.     
  
# >>> Xc=np.array([["a",1,"c","x"],
#                  ["b",2,"c","y"],
#                  ["c",1,"d","x"],
#                  ["a",2,"d","z"],
#                  ["c",1,"e","y"],
#                  ["c",2,"f","y"]])
   
# >>> codifica_one_hot(Xc)
# 
# array([[1., 0., 0.,   1., 0.,   1., 0., 0., 0.,   1., 0., 0.],
#        [0., 1., 0.,   0., 1.,   1., 0., 0., 0.,   0., 1., 0.],
#        [0., 0., 1.,   1., 0.,   0., 1., 0., 0.,   1., 0., 0.],
#        [1., 0., 0.,   0., 1.,   0., 1., 0., 0.,   0., 0., 1.],
#        [0., 0., 1.,   1., 0.,   0., 0., 1., 0.,   0., 1., 0.],
#        [0., 0., 1.,   0., 1.,   0., 0., 0., 1.,   0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11


def codifica_one_hot(X):
    # En primer lugar, obtenemos el nº de columnas del conjunto X
    num_columnas = X.shape[1]
    
    # Creamos un array para almacenar los atributos codificados
    atributos_codificados = []
    
    for columna in range(num_columnas):
        # Obtenemos los valores únicos de la columna actual
        # haciendo uso del método unique de numpy
        valores_unicos = np.unique(X[:, columna])
        
        # Codificamos el atrib utilizando indices booleanos
        # y los añadimos al array anteriormente inicializado vacío
        codificado = (X[:, columna, None] == valores_unicos).astype(float)
        atributos_codificados.append(codificado)
    
    # Finalmente, aplanamos los atributos codificados (por filas) en un único array
    X_codificado = np.concatenate(atributos_codificados, axis=1)
    
    return X_codificado



# -- TEST EJ 7 --
Xc=np.array([["a",1,"c","x"],
             ["b",2,"c","y"],
             ["c",1,"d","x"],
             ["a",2,"d","z"],
             ["c",1,"e","y"],
             ["c",2,"f","y"]])

# print(codifica_one_hot(Xc))

# -------- 


# =====================================================
# EJERCICIO 8: APLICACIONES DEL CLASIFICADOR MULTICLASE
# =====================================================


# ---------------------------------------------------------
# 8.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR Y one-hot de los ejercicios anteriores,
# para obtener un clasificador que aconseje la concesión, 
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito. 

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado 
# exhaustivo)

# ----------------------

def clasif_mult(rate, rate_decay, batch_tam, imprimir):
    # Inicializamos el constructor del clasificador OvR y lo asignamos a una variable
    rl_credito_ovr = RL_OvR(rate=rate, rate_decay=rate_decay, batch_tam=batch_tam)


    # Codificamos los atributos en formato one-hot
    Xe_codificado = codifica_one_hot(Xe_credito)
    Xp_codificado = codifica_one_hot(Xp_credito)

    # Entrenar el modelo OvR con los datos de entrenamiento codificados
    rl_credito_ovr.entrena(Xe_codificado, ye_credito)


    # Realizar predicciones en el conjunto de entrenamiento y prueba
    # prediccion_entrenamiento = rl_credito_ovr.clasifica(Xe_codificado)
    prediccion_prueba = rl_credito_ovr.clasifica(Xp_codificado)

    if imprimir:
        for ejemplo, prediccion in zip(cd.X_credito, prediccion_prueba):
            print(f"Ejemplo: {ejemplo} => Predicción: {prediccion}")

    rend = rendimiento(rl_credito_ovr, Xe_codificado, ye_credito)

    return [rate, rate_decay, batch_tam, rend]


# Dividir el conjunto de datos en entrenamiento y prueba
Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(cd.X_credito, cd.y_credito)

import itertools
# Params
valores_rate = [0.1, 0.01, 0.001]
valores_rate_decay = [True, False]
valores_batch_tam = [16, 32, 64, 128]

# Creamos una lista de combinaciones de parametros haciendo uso del modulo itertools
combinaciones = list(itertools.product(valores_rate, valores_rate_decay, valores_batch_tam))

# Para cada combinacion de params
rendMax = 0
ls_param = [0, True, 0]
for combinacion in combinaciones:
    ls_comb = clasif_mult(combinacion[0], combinacion[1], combinacion[2], False)
    if rendMax < ls_comb[3]:
        rendMax = ls_comb[3]
        ls_param[0] = ls_comb[0]
        ls_param[1] = ls_comb[1]
        ls_param[2] = ls_comb[2]

# -- TEST EJ 8.1 -- 

clasif_mult(ls_param[0], ls_param[1], ls_param[2], True)
print("Rendimiento alcanzado", rendMax)


# ---------------------------------------------------------
# 8.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR anterior, para obtener un
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

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 


# --------------------------------------------------------------------------
import os
import numpy as np
import zipfile

import os
import numpy as np
import zipfile

def leer_datos():
    
    #Descomprimimos el fichero zip en caso de ser necesario
    if not os.path.exists("datos/digits"):
        with zipfile.ZipFile("datos/digitdata.zip", "r") as zip_ref:
            zip_ref.extractall("datos/digits")

    # Inicializamos las particiones de entrenamiento y test vacías
    X_train, y_train, X_test, y_test = [], [], [], []

    # Creamos un conversor de caracteres a 0 o 1
    conversor_caracter = lambda x: 0.0 if x == ' ' else 1.0

    # Leemos los 4 ficheros y aplicamos el conversor binario para
    # las imágenes y un casting a entero para los ficheros de etiquetas
    with open("datos/digits/testimages", encoding='utf-8') as f:
        X_test = [[conversor_caracter(c) for c in line] for line in f]

    with open("datos/digits/testlabels", encoding='utf-8') as f:
        y_test = [int(line) for line in f if len(line.strip()) > 0]

    with open("datos/digits/trainingimages", encoding='utf-8') as f:
        X_train = [[conversor_caracter(c) for c in line] for line in f]

    with open("datos/digits/traininglabels", encoding='utf-8') as f:
        y_train = [int(line) for line in f if len(line.strip()) > 0]

    # Aplicamos el método array_split para dividir los conjuntos 
    # de datos X_train y X_test en subconjuntos
    X_train_div = np.array(np.array_split(np.array(X_train), len(y_train)))
    X_test_div = np.array(np.array_split(np.array(X_test), len(y_test)))
    y_train_div = np.array(y_train)
    y_test_div = np.array(y_test)

    # Utilizamos np.reshape para cambiar la forma del array a (n_ejemplos, 784)
    return np.reshape(X_train_div, (X_train_div.shape[0], -1)), np.reshape(X_test_div, (X_test_div.shape[0], -1)), y_train_div, y_test_div

# -- TEST EJ 8.2 --
Xe_digitos,Xp_digitos,ye_digitos,yp_digitos = leer_datos()
rl_digitos=RL_OvR(rate=0.01, rate_decay=False, batch_tam=64)
rl_digitos.entrena(Xe_digitos,ye_digitos)
# print("Rendimiento en entrenamiento:", rendimiento(rl_digitos,Xe_digitos,ye_digitos))
# print("Rendimiento en prueba:", rendimiento(rl_digitos,Xp_digitos,yp_digitos))


# =========================================================================
# EJERCICIO OPCIONAL PARA SUBIR NOTA: 
#    CLASIFICACIÓN MULTICLASE CON REGRESIÓN LOGÍSTICA MULTINOMIAL
# =========================================================================


#  Se pide implementar un clasificador para regresión
#  multinomial logística con softmax (VERSIÓN MINIBATCH), descrito en las 
#  diapositivas 55 a 57 del tema de "Complementos de Aprendizaje Automático". 

# class RL_Multinomial():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica_prob(self,ejemplos):

#        ......
 

#     def clasifica(self,ejemplos):

#        ......
   

 
# Los parámetros tiene el mismo significado que en el ejercicio 7 de OvR. 

# En eset caso, tiene sentido definir un clasifica_prob, ya que la función
# softmax nos va a devolver una distribución de probabilidad de pertenecia 
# a las distintas clases. 


# NOTA 1: De nuevo, es muy importante para la eficiencia usar numpy para evitar
#         el uso de bucles for convencionales.  

# NOTA 2: Se recomienda usar la función softmax de scipy.special: 

    # from scipy.special import softmax   
#

    
# --------------------------------------------------------------------

# Ejemplo:

# >>> rl_iris_m=RL_Multinomial(rate=0.001,batch_tam=8)

# >>> rl_iris_m.entrena(Xe_iris,ye_iris,n_epochs=50)

# >>> rendimiento(rl_iris_m,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris_m,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

# --------------- 

import numpy as np
from scipy.special import softmax

class RL_Multinomial():
     #Inicializamos el constructor y sus variables necesarias
    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.weights = None
    #Codificamos en one-hot los valores de la clase
    def _one_hot_encode(self, y):
        n_samples = len(y)
        n_classes = np.max(y) + 1
    #Creamos una Matriz de ceros de n_samples y n_classes y establece en 1 los elementos correspondientes en y.
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    #Aplicamos función softmax a las filas de X
    def _softmax(self, X):
        return softmax(X, axis=1)
    #Inicializa los pesos del modelo
    def _initialize_weights(self, n_features, n_classes):
        #Calculamos el limite
        limit = 1 / np.sqrt(n_features)
        #Generamos valores aleatorios dentro de los limites para la matriz de pesos
        self.weights = np.random.uniform(-limit, limit, (n_features, n_classes))
    #Creamos lotes de tamaño batch_tam
    def _batch_generator(self, X, y):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_tam):
            end = min(start + self.batch_tam, n_samples)
            batch_idx = indices[start:end]
            yield X[batch_idx], y[batch_idx]

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        n_features = X.shape[1]
        n_classes = np.max(y) + 1
        y_encoded = self._one_hot_encode(y)
        self._initialize_weights(n_features, n_classes)

        for epoch in range(n_epochs):
            if self.rate_decay:
                self.rate /= (1 + epoch)

            for batch_X, batch_y in self._batch_generator(X, y_encoded):
                y_pred = self._softmax(batch_X.dot(self.weights))
                error = y_pred - batch_y
                gradient = batch_X.T.dot(error)
                self.weights -= self.rate * gradient

            if salida_epoch:
                y_pred = self.clasifica_prob(X)
                loss = self._cross_entropy_loss(y_encoded, y_pred)
                accuracy = self._accuracy(y, np.argmax(y_pred, axis=1))
                print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

    def clasifica_prob(self, ejemplos):
        return self._softmax(ejemplos.dot(self.weights))

    def clasifica(self, ejemplos):
        prob = self.clasifica_prob(ejemplos)
        return np.argmax(prob, axis=1)

    def _cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        return loss

    def _accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)



# rl_iris_m=RL_Multinomial(rate=0.001,batch_tam=8)
# rl_iris_m.entrena(Xe_iris,ye_iris,n_epochs=50)

# print(rendimiento(rl_iris_m,Xe_iris,ye_iris))
# print(rendimiento(rl_iris_m,Xp_iris,yp_iris))











