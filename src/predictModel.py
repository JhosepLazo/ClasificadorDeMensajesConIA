#Importamos joblib para cargar el modelo entrenado
import joblib

#Cargamos el modelo entrenado desde el archivo
rutaModelo = "modeloClasificador.pkl"

#Cargamos el modelo entrenado desde el archivo
modelo = joblib.load(rutaModelo)

#Pedimos al usuario que ingrese un mensaje 
mensajePrueba = input("Ingrese el mensaje del cliente: ")

#Usamos el modelo para predecir la categoría del mensaje
categoriaPredicha = modelo.predict([mensajePrueba])

#Mostramos el mensaje y la categoría predicha
print(f"Mensaje: {mensajePrueba} - Categoría Predicha: {categoriaPredicha[0]}")