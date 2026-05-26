#Importamos pandas para poder leer y trabajar con archivos csv
import pandas as pd 
#Importamos train_test_split para dividir nuestro dataset en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
#Importamos Pipeline para unir varios pasos del modelo en una sola estructura
from sklearn.pipeline import Pipeline
#Importamos TfidfVectorizer para convertir texto en numeros
from sklearn.feature_extraction.text import TfidfVectorizer
#Importamos LogisticRegression como clasificador inicial
from sklearn.linear_model import LogisticRegression
#Importamos metricas para evaluar que tan bien esta clasificando el modelo
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Importamos joblib para guardar el modelo entrenado en un archivo
import joblib
import os

#Guardamos en una variable la ruta donde se encuentra nuestro dataset
rutaDataset = "data/mensajesClientes.csv"

#Guardamos en una variable la ruta donde se guardara el modelo entrenado
rutaModelo = "models/modeloClasificador.joblib"

#Guardamos en una variable el reporte de metricas que se mostrará al finalizar el entrenamiento
rutaReporteMetricas = "reports/reporteMetricas.txt"

#Guardamos en una variable la matriz de confusión que se mostrará al finalizar el entrenamiento
rutaMatrizConfusion = "reports/matrizConfusion.txt"

#Definimos una función para cargar el dataset y realizar limpieza básica de los datos
def cargaDataSet(ruta):
        #Leemos el dataset utilizando pandas
        datosMensajes = pd.read_csv(ruta)

        #Definimos las columnas esperadas en el dataset
        columnasEsperadas = {'mensaje', 'categoria'}

        #Verificamos que el dataset contenga las columnas esperadas
        if not columnasEsperadas.issubset(datosMensajes.columns):
            raise ValueError(f"Error: El dataset debe contener las columnas: {columnasEsperadas}")
            
        #Eliminamos filas con valores faltantes en las columnas 'mensaje' o 'categoria'
        datosMensajes = datosMensajes.dropna(subset = ['mensaje', 'categoria'])

        #Convertimos las columnas 'mensaje' y 'categoria' a tipo string y eliminamos espacios 
        datosMensajes['mensaje'] = datosMensajes['mensaje'].astype(str).str.strip()
        datosMensajes['categoria'] = datosMensajes['categoria'].astype(str).str.strip()

        #Eliminamos filas con mensajes vacíos después de limpiar espacios
        datosMensajes = datosMensajes[datosMensajes['mensaje'] != '']

        #Mostramos un mensaje para confirmar que el dataset se cargó correctamente
        print(f"Dataset cargado correctamente desde: {ruta}")

        #Devolvemos el dataset limpio y listo para usar
        return datosMensajes

#Definimos una función para crear el modelo de clasificación de mensajes
def crearModelo():
    #Creamnos el Pipeline del modelo
    modelo = Pipeline([
        #Primer paso: Convertir los mensajes de texto en números utilizando TF-IDF
        ('tfidf', TfidfVectorizer(
            lowercase = True, #Convertir todo el texto a minúsculas para evitar duplicados
            strip_accents = 'unicode', #Eliminar acentos para mejorar la consistencia
            ngram_range = (1, 2), #Considerar tanto palabras individuales como pares de palabras para capturar más contexto
            max_features = 1000, #Limitar el número de características para evitar sobreajuste
        )),
        #Segundo paso: Clasificar los mensajes utiizando regresión logística
        ('clasificador', LogisticRegression(
            max_iter = 1000, #Aumentar el número de iteraciones para asegurar la convergencia del modelo
        ))
    ])

    #Devolvemos el modelo creado
    return modelo

#Defininimos una duunción para los reportes de evaluación del modelo y guardarlos en archivos
def guardarReportes(yPrueba, predicciones, etiquetas, exactitud):
    #Creamos la carpeta para los reportes si no existe
    os.makedirs(os.path.dirname(rutaReporteMetricas), exist_ok=True)
    os.makedirs(os.path.dirname(rutaMatrizConfusion), exist_ok=True)

    #Generamos el reporte de clasificación
    reporteClasificacion = classification_report(yPrueba, predicciones, target_names = etiquetas)

    #Generamos la matriz de confusión
    matrizConfusion = confusion_matrix(yPrueba, predicciones, labels = etiquetas)   

    #Guardamos el reporte de métricas en un archivo de texto
    with open(rutaReporteMetricas, 'w', encoding='utf-8') as archivoReporte:
        archivoReporte.write("REPORTE DEL MODELO\n")
        archivoReporte.write(f"Exactitud del modelo: {exactitud:.2f}\n\n")
        archivoReporte.write("Reporte de clasificación:\n")
        archivoReporte.write(reporteClasificacion)

    #Convertimos la matriz de confusión en una tabla
    tablaMatrizConfusion = pd.DataFrame(
        matrizConfusion, #Datos de la matriz de confusión 
        index = etiquetas, #Etiquetas para las filas de la matriz (categorías reales)
        columns = etiquetas #Etiquetas para las columnas de la matriz (categorías predichas)
        )
    
    #Guardamos la matriz de confusión en formato CSV
    tablaMatrizConfusion.to_csv(rutaMatrizConfusion, encoding = 'utf-8')

#Definimos la función principal para entrenar el modelo
def entrenarModelo():
    #Cargamos el dataset limpio
    datosMensajes = cargaDataSet(rutaDataset)

    #Mostramos información básica del dataset para verificar que se ha cargado correctamente
    print(datosMensajes.head())
    print(datosMensajes.shape)
    print(datosMensajes.columns)

    #Contamos cuantas veces aparece cada categoría en el dataset
    conteoCategorias = datosMensajes['categoria'].value_counts()
    #Mostramos el conteo de mensajes por categoría
    print(f"\nCantidad de mensajes por categorías: \n{conteoCategorias}")

    #Separamos los mensajes que serán los datos de entrada del modelo
    mensajes = datosMensajes['mensaje']
    #Separamos las categorías que serán las respuestas correctas del modelo
    categorias = datosMensajes['categoria']

    #Dividimos el dataset en conjuntos de entrenamiento y prueba
    xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(
        mensajes, #Datos de entrada para el modelo 
        categorias, #Respuestas correctas para el modelo
        test_size = 0.2, #El 20% para pruebas y el 80% para entrenamiento
        random_state = 42, #Semilla para garantizar que la división sea escalable
        stratify = categorias #Estratificamos la división para mantener la proporción de categorías en ambos conjuntos
        )

    #Mostramos cuantos mensajes quedaron para entrenamiento y cuantos para prueba 
    print(f"\nCantidad de mensajes para entrenamiento: \n{len(xEntrenamiento)}")
    print(f"\nCantidad de mensajes para prueba: \n{len(xPrueba)}")

    #Creamos el modelo de clasificación de mensajes
    modelo = crearModelo()

    #Entrenamos el modelo usando los mensajes y categorías de entrenamiento
    modelo.fit(xEntrenamiento, yEntrenamiento)
    #Mostramos un mensaje para confirmar que el entrenamiento terminó
    print("\nModelo entrenado correctamente.")

    #Usamos el modelo entrenado para predecir las categorías de los mensajes de prueba
    predicciones = modelo.predict(xPrueba)

    #Calculamos la exactitud del modelo
    exactitud = accuracy_score(yPrueba, predicciones)

    #Mostramos el % de exactitud del modelo
    print(f"\nExactitud del modelo: {exactitud:.4f}")

    #Obtener las etiquetas únicas ordenadas de las categorías para los reportes
    etiquetasCategorias = sorted(categorias.unique())

    #Guardamos los reportes del modelo
    guardarReportes(yPrueba, predicciones, etiquetasCategorias, exactitud)

    #Creamos la carpeta para guardar el modelo si no existe
    os.makedirs(os.path.dirname(rutaModelo), exist_ok=True)

    #Guardamos el modelo entrenado para usarlo despues sin volver a entrenar
    joblib.dump(modelo, rutaModelo)

    #Confirmamos que el modelo se guardó correctamente
    print(f"\nModelo guardado correctamente en: {rutaModelo}")

    #Confirmamos que los reportes se guardaron correctamente
    print(f"Reporte de métricas guardado en: {rutaReporteMetricas}")
    print(f"Matriz de confusión guardada en: {rutaMatrizConfusion}")

#Ejecutamos el entrenamiento del modelo solo si este archivo se ejecuta directamente, no si se importa como módulo
if __name__ == "__main__":    
    entrenarModelo()
