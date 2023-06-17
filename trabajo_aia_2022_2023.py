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
        self.media = None
        self.desviacion=None

    def ajusta(self,X):
        self.media = np.mean(X, axis=0)
        self.desviacion = np.std(X, axis=0)

    def normaliza(self,X):
        if self.media is None or self.desviacion is None:
            raise NormalizadorNoAjustado("Normalizador no ajustado")
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
        self.minimo = None
        self.maximo = None

    def ajusta(self,X):
        self.minimo = np.min(X, axis=0)
        self.maximo = np.max(X, axis=0)

    def normaliza(self,X):
        if self.minimo is None or self.maximo is None:
            raise NormalizadorNoAjustado("Normalizador no ajustado")
        X_normalizado = (X - self.minimo) / (self.maximo - self.minimo)
        return X_normalizado

normMM_cancer=NormalizadorMinMax()
normMM_cancer.ajusta(Xe_cancer)


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
        self.rate = rate  
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.pesos = None # Inicializamos pesos y clases en el método init
        self.clases = []  # ya que será útil su uso en distintas funciones
                          # y su valor debe almacenarse a modo de var estática 

    def sigmoide(self, x):
        return expit(x)

    def entropia_cruzada(self, y, y_pred):
        return np.sum(np.where(y == 1, -np.log(y_pred), -np.log(1 - y_pred)))
        # return np.where(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    def entrena(self, X, y, Xv=None, yv=None, n_epochs=100, salida_epoch=False, early_stopping=False, paciencia=3):

        #Obtenemos el conjunto clases únicas e inicializamos los pesos de manera aleatoria.
        self.clases = np.unique(y)
        self.pesos = np.random.rand(X.shape[1])

        if Xv is None:
            Xv = X
            yv = y

        # Declaramos la mejor pérdida a infinito
        # De esta forma, si la pérdida actual es menor, aseguramos que se actualice
        mejor_perdida = float('inf')
        contPaciencia = 0

        for epoch in range(n_epochs):
            if(self.rate_decay==True):
                rate_n = (self.rate)*(1/(1+epoch)) # self.rate decrecerá en cada it
            else:
                rate_n = self.rate 

            indices = np.arange(len(X))
            np.random.shuffle(indices)

            # Recorremos X saltando de batch en batch
            for i in range(0, len(X), self.batch_tam):
                indices_batch = indices[i:(i+self.batch_tam)] # Indices del batch seleccionado
                X_batch = X[indices_batch] # Datos de X que hacen referencia a esos indices
                y_batch = y[indices_batch]

                y_pred = self.sigmoide(np.dot(X_batch, self.pesos))
                gradiente = np.dot(X_batch.T, y_pred - y_batch) / len(X_batch)
                # gradiente = np.dot(X_batch.T, y_pred - y_batch)
                self.pesos -= rate_n * gradiente

            if salida_epoch:
                pred_entr = self.clasifica_prob(X)
                perdidaEntr = self.entropia_cruzada(y, pred_entr)
                # acc_train = np.mean(self.clasifica(X) == y)
                accuracy_entr = rendimiento(self, X, y)

                y_pred_val = self.clasifica_prob(Xv)
                perdida = self.entropia_cruzada(yv, y_pred_val)
                # acc_val = np.mean(self.clasifica(Xv) == yv)
                accuracy_val = rendimiento(self, Xv, yv)

                print(f"Epoch {epoch + 1}:")
                print(f"  en entrenamiento EC: {perdidaEntr:.4f}, rendimiento: {accuracy_entr:.4f}")
                print(f"  en validación EC: {perdida:.4f}, rendimiento: {accuracy_val:.4f}")

                if early_stopping:
                    if perdida < mejor_perdida:
                        mejor_perdida = perdida
                        contPaciencia = 0
                    else: 
                        contPaciencia += 1

                    if contPaciencia >= paciencia:
                        print("PARADA TEMPRANA")
                        break
                

    def clasifica_prob(self, ejemplos):
        if self.pesos is None:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado")

        y_pred = self.sigmoide(np.dot(ejemplos, self.pesos))
        return y_pred

    def clasifica(self, ejemplo):
        if self.pesos is None:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado")

        y_pred = self.sigmoide(np.dot(ejemplo, self.pesos))
        y_pred_class = np.where(y_pred >= 0.5, self.clases[1], self.clases[0])
        return y_pred_class

lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
# lr_cancer.entrena(Xe_cancer_n, ye_cancer, Xv_cancer_n, yv_cancer, salida_epoch=True, early_stopping=True)
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


# Importar la clase RegresionLogisticaMiniBatch (Ejercicio 3)
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

    # Para ajustar los parámetros, vamos a ir actualizando mejor_rendimiento_votos
    # según el rendimiento asociado a la combinatoria de parámetros que estemos 
    # evaluando (ahora declararemos los valores de params)
    mejor_rendimiento_votos = 0
    mejor_rendimiento_votos = {}

    # Asociamos los posibles valores para rate, rate_decay y batch_tam
    params_votos = {
        'rate': [0.1, 0.01, 0.001],
        'rate_decay': [True, False],
        'batch_tam': [32, 64, 128]
    }
    
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
                rendimiento = rendimiento_validacion_cruzada(
                    RegresionLogisticaMiniBatch, params, Xe_votos_n, ye_votos, Xv_votos_n, yv_votos
                )
                print(f"Parámetros: {params}, Rendimiento medio: {rendimiento}")

                if rendimiento > mejor_rendimiento_votos:
                    mejor_rendimiento_votos = rendimiento
                    mejores_parametros_votos = params

    # Aplicamos el método entrena de RegresionLogisticaMiniBatch con los mejores parámetros obtenidos
    lr_votos = RegresionLogisticaMiniBatch(**mejores_parametros_votos)
    lr_votos.entrena(Xe_cancer_n, ye_cancer)

    # Evaluamos el conjunto de prueba
    rendimiento_prueba_votos = rendimiento(Xe_votos_n, ye_votos)
    print(f"Rendimiento en conjunto de prueba: {rendimiento_prueba_votos}")

Xev_votos,Xp_votos,yev_votos,yp_votos=particion_entr_prueba(cd.X_votos,cd.y_votos,test=1/3)
Xe_votos,Xv_votos,ye_votos,yv_votos=particion_entr_prueba(Xev_votos,yev_votos,test=1/3)

normst_votos = NormalizadorStandard()
normst_votos.ajusta(Xe_votos)

Xe_votos_n = normst_votos.normaliza(Xe_votos)
Xv_votos_n = normst_votos.normaliza(Xv_votos)
Xp_votos_n = normst_votos.normaliza(Xp_votos)

# clasifBin_votos()

def clasifBin_cancer():
    lr_cancer = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True)

    # Para ajustar los parámetros, vamos a ir actualizando mejor_rendimiento_votos
    # y mejor_rendimiento_votos según el rendimiento asociado a la combinatoria de
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

# Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(cd.X_cancer,cd.y_cancer,test=0.2)
# Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# normst_cancer = NormalizadorStandard()
# normst_cancer.ajusta(Xe_cancer)
# Xe_cancer_n = normst_cancer.normaliza(Xe_cancer)
# Xv_cancer_n = normst_cancer.normaliza(Xv_cancer)
# Xp_cancer_n = normst_cancer.normaliza(Xp_cancer)

# clasifBin_cancer()




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
        self.classifiers = {}

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        # Obtenemos las clases únicas
        self.classes = np.unique(y)

        # Iteramos dichas clases obtenidas en los datos de entrenamiento
        # y entrena un clasificador binario para cada una de ellas
        for c in self.classes:
            y_binary = np.where(y == c, 1, 0)
            classifier = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay, batch_tam=self.batch_tam)
            
            # Entrenamos el clasificador para la predección de clases
            classifier.entrena(X, y_binary, n_epochs=n_epochs, salida_epoch=salida_epoch)
            self.classifiers[c] = classifier

    def clasifica(self, ejemplos):
        if not self.classifiers:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado.")

        # Calculamos la probabilidad de pertenencia para cada clase
        y_pred = []
        for _, classifier in self.classifiers.items():
            y_pred.append(classifier.clasifica_prob(ejemplos))

        # Devolvemos la clase con mayor probabilidad
        y_pred = np.array(y_pred)
        y_pred_class = np.argmax(y_pred, axis=0)
        return y_pred_class


Xe_iris, Xp_iris, ye_iris, yp_iris = particion_entr_prueba(cd.X_iris, cd.y_iris)
rl_iris_ovr = RL_OvR(rate=0.001, batch_tam=8)
rl_iris_ovr.entrena(Xe_iris, ye_iris)
print(rendimiento(rl_iris_ovr, Xe_iris, ye_iris))
print(rendimiento(rl_iris_ovr, Xp_iris, yp_iris))

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
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11


def codifica_one_hot(X):

    # Aplanamos los atributos y obtenemos los valores únicos
    # con el método unique de numpy
    val_unicos = np.unique(X.flatten()) 
    nValores = len(val_unicos)
    nMuestras, nCaract = X.shape

    # Creamos el array codificado
    X_codif = np.zeros((nMuestras, nCaract * nValores))

    for i in range(nMuestras):
        for j in range(nCaract):
            valor = X[i, j]
            #Obtenemos el índice del valor en val_unicos
            indice = np.where(val_unicos == valor)[0][0]  
            #Asignamos 1 para la posición correspondiente de del array
            X_codif[i, j * nValores + indice] = 1  

    return X_codif


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

# NO VAAAA
# # Codificar los datos de entrada utilizando one-hot encoding
# X_encoded = codifica_one_hot(cd.X_credito)

# # Crear el clasificador OvR
# clf = RL_OvR(rate=0.1, rate_decay=False, batch_tam=64)

# # Entrenar el clasificador
# clf.entrena(X_encoded, cd.y_credito, n_epochs=100, salida_epoch=False)

# # Datos de prueba
# X_prueba = np.array([[1, 0, 1, 0],
#                      [0, 1, 1, 1]])

# # Codificar los datos de prueba utilizando one-hot encoding
# X_prueba_encoded = codifica_one_hot(X_prueba)

# # Realizar la clasificación
# y_pred = clf.clasifica(X_prueba_encoded)

# # Imprimir los resultados
# for ejemplo, prediccion in zip(X_prueba, y_pred):
#     print(f"Ejemplo: {ejemplo} => Predicción: {prediccion}")











# ---------------------------------------------------------
# 7.2) Clasificación de imágenes de dígitos escritos a mano
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
# import os
# import numpy as np
# import zipfile

# def leer_datos():
#     if not os.path.exists("datos/digits"):
#         with zipfile.ZipFile("datos/digitdata.zip", "r") as zip_ref:
#             zip_ref.extractall("datos/digits")

#     with open("datos/digits/trainingimages", "r") as f:
#         lines = f.readlines()
#         #print(lines[:10])  # Imprimir las primeras 10 líneas de los datos de entrenamiento

#     X_train = np.loadtxt("datos/digits/trainingimages", delimiter=" ", dtype=str, usecols=range(29*29))
#     X_train[X_train == ''] = '0'
#     X_train = X_train.astype(int)
#     y_train = np.loadtxt("datos/digits/traininglabels", dtype=int)
#     X_test = np.loadtxt("datos/digits/testimages", delimiter=" ", dtype=int, usecols=range(29*29))
#     X_test[X_test == ''] = '0'
#     X_test = X_test.astype(int)
#     y_test = np.loadtxt("datos/digits/testlabels", dtype=int)

#     # Procesar imágenes
#     X_train = procesar_imagenes(X_train)
#     X_test = procesar_imagenes(X_test)

#     return X_train, X_test, y_train, y_test


# def procesar_imagenes(X):
#     X_processed = np.zeros((len(X), 28, 28), dtype=int)
#     for i, image in enumerate(X):
#         for j, row in enumerate(image):
#             X_processed[i, j] = [0 if c in [" "] else 1 for c in row]
#             # X_processed[i, j] = [0 if c in [" ", "+"] else 1 for c in row]
#     return X_processed


# def particion_entr_prueba(X, y):
#     indices = np.arange(len(X))
#     np.random.shuffle(indices)
#     train_indices = indices[:int(0.8 * len(X))]
#     test_indices = indices[int(0.8 * len(X)):]

#     X_train = X[train_indices]
#     y_train = y[train_indices]
#     X_test = X[test_indices]
#     y_test = y[test_indices]

#     return X_train, X_test, y_train, y_test


# X_train, X_test, y_train, y_test = leer_datos()

# X_train, X_val, y_train, y_val = particion_entr_prueba(X_train, y_train)

# rl_ovr = RL_OvR(rate=0.001, batch_tam=8)
# rl_ovr.entrena(X_train, y_train, n_epochs=100, salida_epoch=True)

# # Clasificar los ejemplos de prueba
# y_pred = rl_ovr.clasifica(X_test)

# # Evaluar el rendimiento
# accuracy = np.mean(y_pred == y_test)
# print("Rendimiento en prueba:", accuracy)




















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
















