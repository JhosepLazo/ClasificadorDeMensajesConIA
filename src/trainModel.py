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


#Guardamos en una variable la ruta donde se encuentra nuestro dataset
rutaDataset = "data/mensajes_clientes.csv"

#Leemos el dataset utilizando pandas y lo guardamos en una tabla
datosMensajes = pd.read_csv(rutaDataset)

#Mostramos las primeras filas del dataset para verificar que se ha leído correctamente
print(datosMensajes.head())

#Mostramos cuantas filas y columnas tiene el dataset
print(datosMensajes.shape)

#Mostramos los nombres de las columnas del dataset
print(datosMensajes.columns)

#Contamos cuantas veces aparece cada categoría en el dataset
conteoCategorias = datosMensajes['categoria'].value_counts()
print(f"\nCantidad de mensajes por categorías: \n{conteoCategorias}")

#Separamos los mensajes que serán lso datos de entrada del modelo
mensajes = datosMensajes['mensaje']

#Separamos las categorías que serán las respuestas correctas del modelo
categorias = datosMensajes['categoria']

#mostramos los mensajes y sus categorías correspondientes para verificar la seaparación
print("\nMensajes y sus Categorías:")
for i in range(10):
    print(f"Mensaje: {mensajes[i]} - Categoría: {categorias[i]}")

#Dividimos el dataset en conjuntos de entrenamiento y prueba utilizando train_test_split
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(
    mensajes, #Datos de entrada para el modelo 
    categorias, #Respuestas correctas para el modelo   
    test_size = 0.2, #El 20% para pruebas y el 80% para entrenamiento    
    random_state = 42, #Semilla para garantizar que la división sea escalable
    stratify = categorias #Estratificación para asegurar que la proporción de categorías se mantenga en ambos conjuntos
)

#Mostramos cuantos mensajes quedaron para entrenamiento y cuantos para prueba 
print(f"\nCantidad de mensajes para entrenamiento: \n{len(xEntrenamiento)}")
print(f"\nCantidad de mensajes para prueba: \n{len(xPrueba)}")

#Creamos un pipeline del modelo que incluye TfidfVectorizer para convertir texto en números y LogisticRegression como clasificador  
modelo = Pipeline([
    #Primer paso: Convertir los mensajes de texto en números utilizando TF-IDF
    ('tfidf', TfidfVectorizer(
        lowercase = True, #Convertir todo el texto a minúsculas para evitar duplicados
        strip_accents = 'unicode', #Eliminar acentos para mejorar la consistencia
        ngram_range = (1, 2), #Considerar tanto palabras individuales como pares de palabras para capturar más contexto
        max_features = 1000 #Limitar el número de características para evitar sobreajuste        
    )),

    #Segundo paso: Clasificar los mensajes utilizando regresión logística
    ('clasificador', LogisticRegression(
        max_iter = 1000, #Aumentar el número de iteraciones para asegurar la convergencia del modelo
    ))
])

#Entrenamos el modelo usando los mensajes y categorías de entrenamiento
modelo.fit(xEntrenamiento, yEntrenamiento)

#Mostramos un mensaje para confirmar que el entrenamiento terminó
print("\nModelo entrenado correctamente.")

#Usamos el modelo entrenado para predecir las categorías de los mensajes de prueba
predicciones = modelo.predict(xPrueba)

#Mostramos las primeras predicciones para revisar que el modelo ya responde 
print("\nPredicciones para los primeros mensajes de prueba:")
for i in range(10):
    print(f"Mensaje: {xPrueba.iloc[i]} - Categoría Real: {yPrueba.iloc[i]} - Categoría Predicha: {predicciones[i]}\n")

#Calculamos la exactitud del modelo 
exactitud = accuracy_score(yPrueba, predicciones)

#Mostramos el % de exactitud del modelo
print(f"\nExactitud del modelo: {exactitud:.2f}")

#Mostramos métricas detalladas por categoría
print("\nReporte de clasificación por categoría:")
print(classification_report(yPrueba, predicciones))

#Mostramos la matriz de confusión para visualizar aciertos y errores del modelo
print("\nMatriz de confusión:")
print(confusion_matrix(yPrueba, predicciones))



















