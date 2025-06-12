"""
Modulo Toto
-------------------

Este modulo fue creado para organizar todas las clases necesarias para rendir CD3

Clases:
  - AnalisisDescriptivo: Clase para realizar analisis descriptivo sobre datos 1-Dimensionales. (Vectores)
  - GeneradoraDeDatos: Clase para generar y graficar datos con distintas distribuciones.
  - Regresion: Clase base para implementar modelos de regresion, a traves de la libreria statsmodels.
  - RegresionLineal: Clase para implementar modelos de regresion lineal.
  - RegresionLogistica: Clase para implementar modelos de regresion logistica.
  - Cualitativas: Clase para resolver ejercicios de cualitativas

(Codigo util si estas laburando en colab)
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks')
import mi_modulo (mi_modulo esta en Colab Notebooks)

(si cambiaste algo del modulo, y necesitas recargarlo)
import importlib
importlib.reload(mi_modulo)
"""

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t, f, chi2
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.anova import anova_lm
import random
from sklearn.metrics import auc
import pandas as pd

class Estimacion:
  """
  Clase para realizar analisis descriptivo sobre datos 1-Dimensionales. (Vectores)

  Atributos:
    datos (np.ndarray): Los datos a analizar.
    n (int): Numero de datos.
  """
  def __init__(self, datos):
    self.datos = datos
    self.n = len(datos)

  def genera_histograma(self,h:float) -> tuple:
    """
    Genera los datos necesarios para construir un histograma.

    Calcula los límites de los intervalos (bins) y las frecuencias absolutas
    normalizadas (densidad) para un ancho de bin dado.

    Args:
      h (float): El ancho de cada intervalo (bin) del histograma.

    Returns:
      tuple: Una tupla que contiene:
        - vector_con_extremos (np.array): Array con los límites de los intervalos.
        - fr_abs (np.array): Array con las frecuencias absolutas normalizadas (densidad).
    """
    n=len(self.datos)
    a=min(self.datos)
    b=max(self.datos)
    vector_con_extremos = np.arange(a,b+h,h)
    fr_abs=np.zeros(len(vector_con_extremos)-1)

    for inter_i in range(len(vector_con_extremos)-1):
      contador = 0
      for i in range(len(self.datos)):
        if vector_con_extremos[inter_i] <= self.datos[i] < vector_con_extremos[inter_i+1]:
          contador += 1
      fr_abs[inter_i]=(contador/n)/h

    return vector_con_extremos, fr_abs


  def evalua_histograma(self, h:float, x:np.ndarray) -> np.ndarray:
    """
    Evalúa la densidad del histograma en puntos específicos.

    Dado un ancho de bin 'h' y un conjunto de puntos 'x', este método
    determina el valor de la densidad del histograma en cada punto.

    Args:
      h (float): El ancho de cada intervalo (bin) del histograma.
      x (np.ndarray): Array de puntos donde se desea evaluar la densidad del histograma.

    Returns:
      np.ndarray: Array con los valores de densidad del histograma evaluados en 'x'.
    """
    inter, frec_abs = self.genera_histograma(h)

    resx=np.zeros(len(x))

    for i in range(len(x)):
      for ind_inter in range(len(inter)-1):
        if inter[ind_inter] <= x[i] < inter[ind_inter + 1]:
          resx[i] = frec_abs[ind_inter]

    return resx

  # Distintos kernels usados para estimar densidad

  def kernel_gaussiano(self,x):
    """Kernel gaussiano estándar"""
    valor_kernel_gaussiano= (1/np.sqrt(2*np.pi)) * np.exp((-1/2) * (x**2))
    return valor_kernel_gaussiano

  def kernel_uniforme(self,x):
    """Kernel uniforme"""
    if -1/2 <= x < 1/2:
      valor_kernel_uniforme = 1
    else:
      valor_kernel_uniforme = 0
    return valor_kernel_uniforme

  def kernel_cuadratico(self,x):
    """Kernel cuadrático o Epanechnikov"""
    i = 0
    if -1 <= x <= 1:
      i = 1
    return (3/4)*(1-x**2)*i

  def kernel_triangular(self,x):
    """Kernel triangular"""
    i1 = 0
    if -1 <= x <= 0:
      i1 = 1

    i2 = 0
    if 0 <= x <= 1:
      i2 = 1

    return (1+x)*i1 + (1-x)*i2

  def mi_densidad(self,h ,x, kernel_string) -> list:
    """
    Estima la densidad usando un estimador kernel.

    Args:
        h (float): Bandwidth o ancho de la ventana.
        x (np.ndarray): Puntos donde se evalúa la densidad.
        kernel_string (string): Función kernel a usar. Puede ser 'gaussiano', 'uniforme', 'cuadratico' o 'triangular'.

    Returns:
        list: Estimaciones de la densidad evaluadas en x.
    """
    if kernel_string == "gaussiano":
      kernel = self.kernel_gaussiano
    elif kernel_string == "uniforme":
      kernel = self.kernel_uniforme
    elif kernel_string == "cuadratico":
      kernel = self.kernel_cuadratico
    elif kernel_string == "triangular":
      kernel = self.kernel_triangular
    else:
      print("No escribiste un kernel elegible")
      return


    n = len(self.datos)
    fabsx=[]

    for i in range(len(x)):
      suma=[]
      for d in self.datos:
        y = (d-x[i])/h
        suma.append(kernel(y))
      fabsx.append(sum(suma) / (n*h))

    #Entonces asi calculamos una estimacion de la frecuencia absoluta.

    density = fabsx
    return density

  def calculo_de_media(self):
    """Calcula la media muestral"""
    media = sum(self.datos)/len(self.datos)

    return media

  def calculo_de_desvio_estandar(self):
    """Calcula el desvío estándar muestral (con n-1 en el denominador)"""
    n=len(self.datos)
    mu = self.calculo_de_media()
    y=0
    for i in self.datos:
      x=(i-mu)**2
      y=y+x
    desvio=np.sqrt(y/(n-1))
    return desvio

  def miqqplot(self):
    """
    Crea un QQ Plot personalizado para verificar normalidad.

    Se usan los cuantiles teóricos de la N(0,1) contra los cuantiles muestrales
    estandarizados.
    """
    self.datos = np.sort(self.datos)
    media = self.calculo_de_media()
    sd  = self.calculo_de_desvio_estandar()

    # Estandarizamos los datos
    data_s = [(x-media)/sd for x in self.datos]
    porcentajes = [i/self.n for i in range(self.n)]
    # Cuantiles
    qs = [np.quantile(data_s, i) for i in porcentajes]
    qteo = [norm.ppf(i) for i in porcentajes]
    # Gráfico
    plt.title("Mi QQ Plot")
    plt.grid()
    plt.scatter(qteo, qs, color='blue', marker='o')
    plt.xlabel('Cuantiles teóricos')
    plt.ylabel('Cuantiles muestrales')
    plt.plot(qteo,qteo , linestyle='-', color='red')
    plt.show()
    return


class GeneradoraDeDatos:
  """
  Clase para generar datos aleatorios según distintas distribuciones estadísticas.

  Atributos:
    n (int): Cantidad de datos a generar.
  """
  def __init__(self, n):
    # Inicializa la clase con la cantidad de datos a generar
    self.n = n

  def unif(self, a, b, grafico = False) -> np.ndarray:
    """
    Genera una muestra de datos con distribución uniforme entre a y b.

    Args:
      a (float): Límite inferior.
      b (float): Límite superior.
      grafico (bool): Si se desea graficar la distribución.

    Returns:
      np.ndarray: Muestra generada de tamaño n con distribución U(a, b).
    """
    n = self.n
    y = np.random.uniform(a, b, n)
    x = np.linspace(a - 5, b + 5, 100)

    # Definición de la función de densidad (no se utiliza en el retorno)
    def f(x):
      res = 0
      if a <= x <= b:
        res = 1 / (b - a)

    # Gráfico
    if grafico:
      plt.plot(x, f(x), label="Grafica distribucion uniforme")
      plt.grid()
      plt.legend()
      plt.show()
    return y

  def normal(self, media, desvio, grafico=False) -> np.ndarray:
    """
    Genera una muestra de datos con distribución normal N(media, desvio).

    Args:
      media (float): Media de la distribución.
      desvio (float): Desvío estándar de la distribución.
      grafico (bool): Si se desea graficar la distribución.

    Returns:
      np.ndarray: Muestra generada de tamaño n.
    """
    n = self.n
    y = np.random.normal(media, desvio, n)

    if grafico:
      x = np.linspace(min(y), max(y), 1000)
      plt.plot(x, norm.pdf(x, media, desvio), label="Grafica distribucion normal")
      plt.grid()
      plt.legend()
      plt.show()
    return y

  def t_student(self, df, grafico=False) -> np.ndarray:
    """
    Genera una muestra de datos con distribución t de Student.

    Args:
      df (int): Grados de libertad de la distribución t.
      grafico (bool): Si se desea graficar la distribución.

    Returns:
      np.ndarray: Muestra generada de tamaño n.
    """
    y = np.random.standard_t(df, self.n)

    if grafico:
      x = np.linspace(min(y), max(y), 1000)
      plt.plot(x, t.pdf(x, df), label="Grafica distribucion t-student")
      plt.grid()
      plt.legend()
      plt.show()
    return y

  def exponencial(self, media, grafico=False) -> np.ndarray:
    """
    Genera una muestra de datos con distribución exponencial de parámetro lambda = 1/media.

    Args:
      media (float): Media de la distribución (lambda = 1/media).
      grafico (bool): Si se desea graficar la distribución.

    Returns:
      np.ndarray: Muestra generada de tamaño n.
    """
    y = np.random.exponential(media, self.n)

    if grafico:
      x = np.linspace(min(y), max(y), 1000)
      f = lambda x: (1 / media) * np.exp(-x / media)
      plt.plot(x, f(x), label="Grafica distribucion exponencial")
      plt.grid()
      plt.legend()
      plt.show()
    return y


class RegresionLineal:
  def __init__(self, y, X):
    """
    Inicializa el modelo con los datos.

    Parámetros:
    - y: vector columna de respuesta (variable dependiente) en el caso de que y sea cuantitativa.
    - X: es la matriz de diseño ya hecha
    (Ayudamemoria de comandos utiles para la creacion de la misma)
    X = np.stack((x1,x2,x3) , axis=1)  stack si los x son 1D, porque stack eleva la dimensionalidad
    X = np.vstack((x1,x2,x3))  vstack si los x ya son 2D, porque vstack crea una matriz 2D solo podes filas
    np.concatenate((A, B), axis=1)  # Para apilar columnas (horizontalmente)
    np.concatenate((A, B), axis=0)  # Para apilar filas (verticalmente)
    X = sm.add_constant(X)
    """
    self.x = X
    self.y = y
    self.n = len(y)
    self.resultado = None

  def mostrar_estadisticas(self,h,kernel):
    """
    Muestra estadísticas básicas de la variable respuesta y grafica su densidad suavizada.

    Parámetros:
    - h: ancho de banda para la estimación de densidad
    - kernel: tipo de kernel (string: 'gaussiano', 'uniforme', "cuadratico", "triangular")
    """
    print(f"""
    media:{np.mean(self.y)}\n
    varianza:{np.var(self.y,ddof=1)}\n
    desviacion estandar:{np.std(self.y,ddof=1)}\n
    minimo:{np.min(self.y)}\n
    maximo:{np.max(self.y)}\n
    cuartiles:{np.quantile(self.y,[0.25,0.5,0.75])}\n
    """)

    x = np.linspace(min(self.y),max(self.y),500)
    plt.plot(x,Estimacion(self.y).mi_densidad(h,x,kernel))
    plt.title("Histograma con kernel", kernel)
    plt.grid()
    plt.show()
    return

  def ajustar_modelo(self,resumen=True):
    """
    Ajusta el modelo lineal usando Mínimos Cuadrados Ordinarios (OLS) con statsmodels.
    Guarda el resultado en self.resultado y opcionalmente imprime el resumen.
    """
    modelo = sm.OLS(self.y, self.x)
    resultado = modelo.fit()

    #Mostremos los resultados
    if resumen:
      print(resultado.summary())
    self.resultado = resultado
    return resultado

  def Recta_evaluada(self,t):
    """
    Evalúa el modelo en un nuevo punto t (lista de predictores).

    Parámetros:
    - t: array de valores de predictores. Ejemplo t = [1,2,3,4,5]

    Retorna:
    - valor predicho por el modelo
    """
    parametros = np.array(self.ajustar_modelo(False).params)  # arreglo 1D de los parametros
    t = np.atleast_1d(t)                                      # convierte escalar en array si hace falta
    t = np.array(t)                                           # convierte en arreglo
    valoraux = np.concatenate(([1],t))
    #self.resultado.predict(valoraux)
    return sum(parametros * valoraux)

  def residuos_modelo(self):
    """
    Calcula los residuos del modelo: diferencia entre valores observados y ajustados.

    Retorna:
    - lista de residuos
    """
    r = []
    X = np.array(self.x)
    parametros = self.ajustar_modelo(False).params
    for i in range(len(self.x)):
        r.append( self.y[i] - sum(parametros * X[i,:]) )

    #Forma alternativa:
    #r = self.ajustar_modelo(False).resid

    return r


  def Varianza_residuos(self):
    """
    Estima la varianza de los residuos usando el estimador clásico.

    Retorna:
    - varianza estimada de los residuos
    """
    r = self.residuos_modelo()
    sigma2alt = (1/(self.n-2)) * sum((r-np.mean(r))**2)
    return sigma2alt

  def Analisis_de_Normalidad_del_error(self):
    """
    Realiza el análisis gráfico y estadístico de la normalidad y homocedasticidad de los residuos.

    Incluye:
    - Gráfico de residuos vs valores ajustados
    - QQ plot de residuos
    - Test de Shapiro-Wilk para normalidad
    - Test de Breusch-Pagan para homocedasticidad
    """

    #residuos vs valores ajustados
    x = np.array(self.x)[:,1:]
    r = self.residuos_modelo()
    y_recta = [self.Recta_evaluada(i) for i in x]
    plt.scatter(y_recta,r)
    plt.title("Valores Ajustados vs Residuos")
    plt.xlabel("Valores ajustados")
    plt.ylabel("Residuos")
    plt.grid()
    plt.show()
    #QQplot
    Estimacion(r).miqqplot()

    #Test de shapiro
    """
    Test de normalidad. En este test se evalúan las hipótesis
    H0:  la distribucion es normal
    H1 : la distribución no es normal
    """
    stat, p_valor1 = shapiro(r)
    print("Valor p para el test hipotesis de la normalidad de los residuos:", round(p_valor1,4))

    #Test de Breusch/Pagan
    """
    Test de homogeneidad de la varianza (homocedasticidad). En este test se evalúan las hipótesis
    H0:  la varianza de la respuesta no depende de las covariables
    H1 : la varianza de la respuesta depende de las covariables
    (H0 implica a de cierta forma que la varianza de los residuos es constante)
    """
    bp_test = het_breuschpagan(r, x)
    bp_value = bp_test[1]
    print("Valor p Homocedasticidad:", bp_value)
    return

  def Int_conf_betas(self,confianza):
    """
    Muestra los intervalos de confianza para los coeficientes del modelo.

    Parámetros:
    - confianza: nivel de confianza (por ejemplo, 0.95)
    """
    alfa = 1 - confianza
    print(self.ajustar_modelo(False).conf_int(alpha = alfa))
    return

  def ANOVA(self, modelo_comparar, F = True):
    """
    Compara el modelo actual (reducido) con un modelo completo mediante análisis ANOVA.

    Parámetros:
    - modelo_comparar: matriz de diseño del modelo completo
    - F: si es True, también calcula y muestra el estadístico F y su p-valor

    Retorna:
    - estadístico F observado si F=True, de lo contrario None
    """
    #Test usando la funcion anova_lm
    modelo_reducido = self.ajustar_modelo(False)
    modelo_completo = sm.OLS(self.y, modelo_comparar).fit()
    anova_resultado = anova_lm(modelo_reducido, modelo_completo)
    print(anova_resultado)

    """
    modelo_completo es la matriz de diseño  del modelo completo
    Comparacion de modelo reducido (en el que estamos) con modelo_completo
    H0: Modelo reducido
    H1: Modelo completo

    indica si hay diferencias significativas en el poder explicativo entre el modelo reducido y el modelo completo.
    """

    #Forma "manual" de calcular el p-valor
    if F:
      RSS_m=sum(modelo_reducido.resid**2)
      RSS_M=sum(modelo_completo.resid**2)
      n=len(self.y)

      if self.x.ndim == 1:
        gl_m=n-1
      else:
        gl_m=n-self.x.shape[1]

      gl_M=n-modelo_comparar.shape[1]
      numerador=(RSS_m-RSS_M)/(gl_m-gl_M)
      denominador=RSS_M/gl_M
      F_obs=numerador/denominador
      p_value = 1-f.cdf(F_obs, gl_m-gl_M, gl_M)
      print("P valor de anova", p_value)
      return F_obs
    return


class RegresionLinealSimple(RegresionLineal):
    """Caso particular de RegresionLineal donde X es un vector."""

    def __init__(self, x, y):
      # Inicializa el modelo simple llamando al constructor de la clase base
      super().__init__(x, y)
      return

    def Grafica_de_dispersion(self):
      # Extrae la única columna de predictores (ignorando el intercepto)
      x = np.array(self.x)[:,1]
      plt.scatter(x, self.y, marker="o", facecolors="none", edgecolors="blue")
      plt.xlabel("X")
      plt.ylabel("Y")
      plt.grid()
      plt.show()
      return

    def Grafica_recta_ajustada(self, imagen=True):
      """
      Grafica la recta ajustada sobre el gráfico de dispersión.

      Parámetros:
      - imagen: si es True, muestra la gráfica.

      Retorna:
      - tupla (b0, b1) con los coeficientes del modelo ajustado
      """
      b0, b1 = self.ajustar_modelo(False).params
      x = np.array(self.x)[:,1]
      if imagen:
          rango_x = np.linspace(min(x), max(x), 100)
          rango_y = [self.Recta_evaluada(i) for i in rango_x]
          plt.plot(rango_x,rango_y)

          plt.scatter(x, self.y, marker="o", facecolors="none", edgecolors="blue")
          plt.xlabel("X")
          plt.ylabel("Y")
          plt.grid()
          plt.show()
      return b0 , b1

    def Intervalo_de_confianza_x0(self,confianza,x0):
      """
      Calcula el intervalo de confianza para la media condicional E[Y|X=x0].

      Parámetros:
      - confianza: nivel de confianza deseado (por ejemplo, 0.95)
      - x0: valor de la variable predictora

      Retorna:
      - lista con los extremos inferior y superior del intervalo
      """
      def SE_conf(x,y,x_0):
        # x: variable predictora
        # y: variable respuesta
        # x_0: valor particular de la variable X donde interesa para el cual interesa estimar la espoeranza de Y

        beta1_est = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
        beta0_est =np.mean(y)-beta1_est*np.mean(x)

        ## estimación de sigma^2
        y_hat=beta0_est+beta1_est*x
        sigma2_est=np.sum((y-y_hat)**2)/(len(x)-2)

        # varianza estimada de beta1_est
        var_beta1_est=sigma2_est/(sum((x-np.mean(x))**2))

        # varianza estimada de beta0_est
        var_beta0_est=sigma2_est*np.sum(x**2)/(len(x)*sum((x-np.mean(x))**2))

        # covarianza de beta0_est y beta1_est
        cov_01=-np.mean(x)*sigma2_est/sum((x-np.mean(x))**2)
        SE2_est=var_beta0_est+(x_0**2)*var_beta1_est+2*x_0*cov_01

        return np.sqrt(SE2_est)
      x = np.array(self.x)[:,1]
      a = 1 - confianza
      SEy0 = SE_conf(x,self.y,x0)
      y0 = self.Recta_evaluada(x0)
      t_critico1, t_critico2 = t.ppf(a/2, len(x) - 2) , t.ppf(1 - a/2, len(x) - 2)

      IC = [float(y0 + t_critico1 * SEy0), float(y0 + t_critico2 * SEy0)]
      print(f"IC para y0 = {IC}")
      return IC

    def Intervalo_predictor_x0(self,confianza,x0):
      """
      Calcula el intervalo predictor para una nueva observación Y dada X=x0.

      Parámetros:
      - confianza: nivel de confianza deseado
      - x0: valor de la variable X donde se desea predecir

      Retorna:
      - lista con los extremos del intervalo predictor
      """
      def SE_conf(x,y,x_0):
        # x: variable predictora
        # y: variable respuesta
        # x_0: valor particular de la variable X donde interesa para el cual interesa estimar la espoeranza de Y

        beta1_est = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
        beta0_est =np.mean(y)-beta1_est*np.mean(x)

        ## estimación de sigma^2
        y_hat=beta0_est+beta1_est*x
        sigma2_est=np.sum((y-y_hat)**2)/(len(x)-2)

        # varianza estimada de beta1_est
        var_beta1_est=sigma2_est/(sum((x-np.mean(x))**2))

        # varianza estimada de beta0_est
        var_beta0_est=sigma2_est*np.sum(x**2)/(len(x)*sum((x-np.mean(x))**2))

        # covarianza de beta0_est y beta1_est
        cov_01=-np.mean(x)*sigma2_est/sum((x-np.mean(x))**2)
        SE2_est=var_beta0_est+(x_0**2)*var_beta1_est+2*x_0*cov_01

        return np.sqrt(SE2_est)

        def SE_pred(x,y,x_0):
          # x: variable predictora
          # y: variable respuesta
          # x_0: valor particular de la variable X donde interesa para el cual interesa estimar la espoeranza de Y

          beta1_est = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
          beta0_est =np.mean(y)-beta1_est*np.mean(x)

          ## estimación de sigma^2
          y_hat=beta0_est+beta1_est*x
          sigma2_est=np.sum((y-y_hat)**2)/(len(x)-2)

          SE2_est_pred=(SE_conf(x,y,x_0))**2+sigma2_est

          return np.sqrt(SE2_est_pred)

        x = np.array(self.x)[:,1]
        a = 1 - confianza
        SEy0 = SE_pred(x,self.y,x0)
        y0 = self.Recta_evaluada(x0)
        t_critico1, t_critico2 = t.ppf(a/2, len(x) - 2) , t.ppf(1 - a/2, len(x) - 2)

        IC = [float(y0 + t_critico1 * SEy0), float(y0 + t_critico2 * SEy0)]
        print(f"IP para y0 = {IC}")
        return IC

    # # Nuevo punto de predicción
    # x_new = x0

    # # Crear la matriz de diseño con el nuevo punto de predicción
    # X_new = sm.add_constant(np.array([[x_new]])) # chatgpt dice que es sm.add_constant(np.array([[1, x_new]]))

    #x1_new, x2_new, x3_new = 0.5, 0.2, 0.8
    #X_new = sm.add_constant(np.array([[x1_new, x2_new, x3_new]]))


    # # Obtener el intervalo de predicción para el nuevo punto
    # prediccion = resultado.get_prediction(X_new)
    # print(prediccion.conf_int(alpha =0.05))                             Confianza
    # print(prediccion.conf_int(obs=True , alpha =0.05))



class RegresionLinealMultiple(RegresionLineal):
    """
    Caso de RegresionLineal donde X es una matriz, en donde las columnas
    representan las distintas variables predictoras.
    """
    def __init__(self, x, y):
      # Inicializa el modelo múltiple llamando al constructor de la clase base
      super().__init__(x, y)
      self.tabla = None
      return

    def predecir_media_de_y(self, new_x, a = 0.05):
      """
      Predice el valor de la media poblacional de Y para un nuevo vector de entrada.

      Parámetros:
      - new_x: lista con los valores de las variables predictoras [x1, x2, ..., xn]
      - a: nivel de significancia (confianza = 1 - a)

      Retorna:
      - intervalo de confianza para la media de Y en ese punto
      """

      resultado = self.ajustar_modelo(False)
      X_new = np.concatenate(( np.array([[1]]), np.array([new_x]) ), axis = 1)
      prediccion = resultado.get_prediction(X_new)
      print(prediccion.conf_int(alpha = a))
      return

    def predecir_y(self, new_x, a = 0.05):
      """
      Predice una nueva observación de Y (no la media) para un nuevo vector de entrada.

      Parámetros:
      - new_x: lista con los valores de las variables predictoras
      - a: nivel de significancia (confianza = 1 - a)

      Retorna:
      - intervalo predictor para una nueva observación de Y
      """
      resultado = self.ajustar_modelo(False)

      X_new = np.concatenate(( np.array([[1]]), np.array([new_x]) ), axis = 1)
      prediccion = resultado.get_prediction(X_new)
      print(prediccion.conf_int(obs=True, alpha = a))
      return

class RegresionLogistica:
  """
  Clase para regresión logística.
  """
  def __init__(self, y, X):
    """
    Inicializa el modelo con los datos.

    Parámetros:
    - y: vector columna de respuesta (variable dependiente) en el caso de que y sea cualitativa
    (Las variables de y deben ser 0 o 1 por lo tanto esto es util)
    y_ajustada = np.where(y == "categoria",1,0)
    - X: es la matriz de diseño ya hecha
    (Ayudamemoria de comandos utiles para la creacion de la misma)
    X = np.stack((x1,x2,x3) , axis=1)  stack si los x son 1D, porque stack eleva la dimensionalidad
    X = np.vstack((x1,x2,x3))  vstack si los x ya son 2D, porque vstack crea una matriz 2D solo podes filas
    np.concatenate((A, B), axis=1)  # Para apilar columnas (horizontalmente)
    np.concatenate((A, B), axis=0)  # Para apilar filas (verticalmente)
    X = sm.add_constant(X)


    Si nos interesa entrenar el modelo
    n=len(y)
    n_train=int(n*0.8)
    n_test=n-n_train
    random.seed(10)
    cuales = random.sample(range(n), n_train)

    datos_test = datos.drop(cuales)   #datos con el 20 porciento de los datos  (por si queres sacar una columna) datos.drop(columns=['GRAVEDAD']) 
    datos_train = datos.iloc[cuales]  #el 80 porciento de los datos

    PRECAUCION: En caso de querer entrenar el modelo se debera separar los datos antes, y dar en la clase el X_train y y_train
    """

    self.x = X
    self.y = y
    self.n = len(y)
    self.resultado = None

  def ajustar_modelo(self,resumen=True):
    """
    Ajusta el modelo Logistico con Logit de statsmodels.
    Guarda el resultado en self.resultado y opcionalmente imprime el resumen.
    """
    modelo = sm.Logit(self.y, self.x)
    resultado = modelo.fit(disp=0)

    #Mostremos los resultados
    if resumen:
      print(resultado.summary())
    self.resultado = resultado
    return resultado

  def Recta_evaluada(self,t):
    """
    Evalúa el modelo en un nuevo punto t (lista de predictores).

    Parámetros:
    - t: array de valores de predictores. Ejemplo t = [1,2,3,4,5]

    Retorna:
    - valor predicho por el modelo
    """
    t = np.atleast_1d(t)                                      # convierte escalar en array si hace falta
    t = np.array(t)                                           # convierte en arreglo
    valoraux = np.concatenate(([1],t))
    return self.ajustar_modelo(False).predict(valoraux)

  def Int_conf_betas(self,confianza):
    """
    Muestra los intervalos de confianza para los coeficientes del modelo.

    Parámetros:
    - confianza: nivel de confianza (por ejemplo, 0.95)
    """
    alfa = 1 - confianza
    print(self.ajustar_modelo(False).conf_int(alpha = alfa))
    return

  def Tabla(self,y_test, X_test,p,graf=True):
    """
    Crea la tabla de confusion a partir de la muestra train que le diste en un inicio y la muestra test. Ademas de printear el sensibilidad, especificidad y la medida de mal clasificados.

    Parámetros:
    - y_test: vector columna de respuesta (variable dependiente) del 20% de los datos restantes
    - X_test: matriz de diseño del 20% de los datos restantes
    - p (float): umbral de decision. 0<p<=1

    Retorna:
    - tupla (sensibilidad, especificidad): sensibilidad y especificidad del modelo con punto de corte p.
    """
    y_pred = np.where(self.ajustar_modelo(False).predict(X_test) >= p,1,0) #Fijate que self.ajustar_modelo(False) da como resultado el modelo para el 80% datos que separaste desde un inicio
    a11 = np.sum((y_pred == 1) & (y_test == 1))
    a12 = np.sum((y_pred == 1) & (y_test == 0))
    a21 = np.sum((y_pred == 0) & (y_test == 1))
    a22 = np.sum((y_pred == 0) & (y_test == 0))

    tablaaux = {"ytest = 1": [a11, a21], "ytest = 0": [a12, a22], "total": [a11+a12, a21+a22]}
    tabla =pd.DataFrame(tablaaux, index=["ypred = 1","ypred = 0"])

    self.tabla = tabla

    sensibilidad = a11/(a11+a21)
    especificidad = a22/(a12+a22)

    MMC = (self.tabla.iloc[1,0] + self.tabla.iloc[0,1])/ (self.tabla.iloc[0,2]+self.tabla.iloc[1,2])

    if graf:
      print(tabla)
      print("Medida de mal clasificados", MMC)
      print("Sensibilidad", sensibilidad)
      print("Especificidad", especificidad)

    return sensibilidad, especificidad


  def Graficas_p_valor(self,y_test, X_test):
    """
    Graficas generadas a partir de la grilla de 100 p-valores, ademas de calcular el indice de Yedin o punto de corte optimo, a la que printea una apreciacion de la curva ROC

    Parámetros:
    - y_test: vector columna de respuesta (variable dependiente) del 20% de los datos restantes
    - X_test: matriz de diseño del 20% de los datos restantes

    Retorna:
    - p_values[np.argmax(res)] (float): el punto de corte optimo para la maxima especifidad y sensibilidad.
    """
    # Generar valores de p
    p_values = np.linspace(0, 1, 100)

    # Inicializar listas para almacenar sensibilidad y especificidad
    sensibilidad_list = []
    especificidad_list = []

    for p in p_values:
      sen , esp = self.Tabla(y_test, X_test,p,False)
      sensibilidad_list.append(sen)
      especificidad_list.append(esp)

    # Graficar sensibilidad y especificidad en función de p
    plt.plot(p_values, sensibilidad_list, label='Sensibilidad')
    plt.plot(p_values, especificidad_list, label='Especificidad')
    plt.xlabel('p')
    plt.ylabel('Valor')
    plt.title('Sensibilidad y Especificidad en función de p')
    plt.grid(True)
    plt.show()

    # Graficar la curva ROC
    especificidadaux = 1-np.array(especificidad_list)
    plt.plot(especificidadaux, sensibilidad_list)
    plt.xlabel('1-especificidad')
    plt.ylabel("sensibilidad")
    plt.title('Curva ROC')
    plt.grid()
    plt.show()

    #Calcular el punto de corte
    sensibilidad_list = np.array(sensibilidad_list)
    especificidad_list = np.array(especificidad_list)
    res = sensibilidad_list + especificidad_list - 1
    print("El punto de corte que maximiza la sensibilidad y especificidad es", p_values[np.argmax(res)])

    #Inferencia a traves de la curva de ROC
    roc_auc = auc(1-especificidad_list, sensibilidad_list)
    if 0.90 < roc_auc <= 1:
      print("AUC:", roc_auc, "el clasificador es EXCELENTE")
    elif 0.80 < roc_auc <= 0.90:
      print("AUC:", roc_auc, "el clasificador es BUENO")
    elif 0.70 < roc_auc <= 0.80:
      print("AUC:", roc_auc, "el clasificador es REGULAR")
    elif 0.60 < roc_auc <= 0.70:
      print("AUC:", roc_auc, "el clasificador es POBRE")
    else:
      print("AUC:", roc_auc, "el clasificador es FALLIDO")

    return p_values[np.argmax(res)]

class Cualitativas():
  """
  Clase para la resolucion de ejercicios de clasificacion cualitativas.
  """
  def __init__(self, o, n, p0):
    """
    Inicializa el modelo con los datos.

    Parámetros:
    - n (int): cantidad de veces que sucede el experimento
    - p0 (array_like): probabilidad estimada de cada evento
    - o (array_like): resultados del experimento (frecuencia muestral)

    Observaciones: La sum(o) = n y len(p) = len(o)
    """
    self.n = n
    self.p0 = np.array(p0)
    self.esperados = self.n * self.p0  #(frecuencia esperada)
    self.o = np.array(o)
    return
    
  def test_chi_cuadrado(self, alfa=0.05):
    """
    Hacemos el test de chi cuadrado, para saber si la frecuencia muestral coincide con la frecuencia esperada.

    Test: H0: p0 = p.
          H1: p0 != p.
    donde p0 es la probabilidad de cada evento (parametro).

    Parámetros:
    - alfa (float): nivel de significancia (confianza = 1 - alfa)

    retorna: None
    """

    chi_observado = sum( [(self.o[i] - self.esperados[i])**2 / self.esperados[i] for i in range(len(self.esperados))] )
    print("Nuestro Chi_observado es ", chi_observado)
    print(f"Percentil de chi cuadrada que tiene {1-alfa}% de area a la izquierda. {chi2.ppf(1 - alfa,len(self.p0) - 1)}")
    print(f"El p-valor de X^2_obs es {1 - chi2.cdf(chi_observado,len(self.p0) - 1)}")
    return
