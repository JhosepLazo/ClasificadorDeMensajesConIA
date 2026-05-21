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




