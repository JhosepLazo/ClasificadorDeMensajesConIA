#Importamos Path para manejar rutas de archivos de forma segura
from pathlib import Path
#Importamos joblib para cargar el modelo entrenado
import joblib

#Cargamos el modelo entrenado desde el archivo
rutaModelo = Path("models/modeloClasificador.joblib")

#Definimos una función para cargar el modelo entrenado desde el archivo
def cargarModelo():
    #Validamos que el archivo del modelo exista antes de intentar cargarlo
    if not rutaModelo.exists():
        raise FileNotFoundError("No se encontró el modelo entrenado. Primero ejecuta: python src/train_model.py")
    
    #Cargamos el modelo entrenado desde el archivo
    modelo = joblib.load(rutaModelo)

    #Retornamos el modelo cargado para usarlo en la predicción
    return modelo

#Definimos una función para predecir la categoría de un mensaje usando el modelo cargado
def predecirMensaje(modelo, mensajeCliente):
    #Usamos el modelo para predecir la categoría del mensaje
    categoriaPredicha = modelo.predict([mensajeCliente])[0]

    #Obtenemos las probabilidades de cada categoría para el mensaje
    probabilidades = modelo.predict_proba([mensajeCliente])[0]

    #Obtenemos la confianza más alta del modelo
    confianza = max(probabilidades)

    #Retornamos la categoría predicha y la confianza del modelo
    return categoriaPredicha, confianza

#Definimos una función para mostrar la predicción de manera amigable al usuario
def mostrarPrediccion(mensajeCliente, categoriaPredicha, confianza):
    #Diccionario para mostrar nombres más amigables de las categorías
    nombresCategorias = {
        "stock": "Consulta sobre stock",
        "reclamo": "Reclamo o queja",
        "cambio_devolucion": "Cambio o devolución",
        "consulta_general": "Consulta general",
        "pedido_pendiente": "Pedido pendiente",
    }

    #Obtenemos el nombre amigable de la categoría usando el diccionario
    categoriaPredicha = nombresCategorias.get(categoriaPredicha, "Categoría desconocida")

    #Mostramos el mensaje analizado
    print(f"\nMensaje del cliente: {mensajeCliente}")
    #Mostramos la categoría predicha
    print(f"\nCategoría predicha: {categoriaPredicha}")
    #Mostramos la confianza del modelo 
    print(f"\nConfianza del modelo: {confianza:.2f}")

    #Mostramos el mensaje, la categoría predicha y la confianza del modelo
    print(f"Mensaje: {mensajeCliente} - Categoría Predicha: {categoriaPredicha} (Confianza: {confianza:.2f})")

def ejecutarPrediccion():
    #Cargamos el modelo entrenado
    modelo = cargarModelo()

    #Creamos un ciclo para clasificar varios mensajes sin cerrar el programa
    while True: 
        #Pedimos al usuario que ingrese un mensaje
        mensajeCliente = input("\nIngrese el mensaje del cliente: ")

        #Limpiamos espacios al inicio y al final del mensaje ingresado
        mensajeCliente = mensajeCliente.strip()

        #Si el usuario no escribe nada, pedimos un mensaje valido
        if mensajeCliente == "":
            print("Debes escribir un mensaje para clasificar.")
            continue

        #Si el usuario ingresa "salir", terminamos el ciclo
        if mensajeCliente.lower() == "salir":
            print("Programa finalizado.")
            break   

        #Usamos el modelo para predecir la categoría del mensaje
        categoriaPredicha , confianza= predecirMensaje(modelo, mensajeCliente)

        #Mostramos el mensaje y la categoría predicha
        mostrarPrediccion(mensajeCliente, categoriaPredicha, confianza)

#Ejecutamos la función principal para iniciar el programa de predicción
if __name__ == "__main__":
    ejecutarPrediccion()