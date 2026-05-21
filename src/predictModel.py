#Importamos joblib para cargar el modelo entrenado
import joblib

#Cargamos el modelo entrenado desde el archivo
rutaModelo = "modeloClasificador.pkl"

#Cargamos el modelo entrenado desde el archivo
modelo = joblib.load(rutaModelo)

#Creamos un ciclo para clasificar varios mensajes sin cerrar el programa
while True: 
    #Pedimos al usuario que ingrese un mensaje
    mensajePrueba = input("\nIngrese el mensaje del cliente: ")

    #Limpiamos espacios al inicio y al final del mensaje ingresado
    mensajePrueba = mensajePrueba.strip()

    #Si el usuario no escribe nada, pedimos un mensaje valido
    if mensajePrueba == "":
        print("Debes escribir un mensaje para clasificar.")
        continue

    #Si el usuario ingresa "salir", terminamos el ciclo
    if mensajePrueba.lower() == "salir":
        print("Programa finalizado.")
        break   

    #Usamos el modelo para predecir la categoría del mensaje
    categoriaPredicha = modelo.predict([mensajePrueba])

    #Mostramos el mensaje y la categoría predicha
    print(f"Mensaje: {mensajePrueba} - Categoría Predicha: {categoriaPredicha[0]}")
