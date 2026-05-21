#Importamos pandas para poder leer y trabajar con archivos csv
import pandas as pd 

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
print("\nCantidad de mensajes por categorías:")
print(conteoCategorias)

#Separamos los mensajes que serán lso datos de entrada del modelo
mensajes = datosMensajes['mensaje']

#Separamos las categorías que serán las respuestas correctas del modelo
categorias = datosMensajes['categoria']

#mostramos los mensajes y sus categorías correspondientes para verificar la seaparación
print("\nMensajes y sus Categorías:")
for i in range(10):
    print(f"Mensaje: {mensajes[i]} - Categoría: {categorias[i]}")


