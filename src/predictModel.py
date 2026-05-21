#Importamos joblib para cargar el modelo entrenado
import joblib

#Cargamos el modelo entrenado desde el archivo
rutaModelo = "modeloClasificador.pkl"

#Cargamos el modelo entrenado desde el archivo
modelo = joblib.load(rutaModelo)

#Creamos un mensaje de prueba para clasificar
mensajeprueba = "Quiero saber si tienen zapatos de talla 42"

#Usamos el modelo para predecir la categoría del mensaje
categoriaPredicha = modelo.predict([mensajeprueba])

#Mostramos el mensaje y la categoría predicha
print(f"Mensaje: {mensajeprueba} - Categoría Predicha: {categoriaPredicha[0]}")