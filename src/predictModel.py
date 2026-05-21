#Importamos joblib para cargar el modelo entrenado
import joblib

#Cargamos el modelo entrenado desde el archivo
rutaModelo = "modeloClasificador.pkl"

#Diccionario para mostrar nombres más amigables de las categorías
nombresCategorias = {
    "stock": "Consulta sobre stock",
    "reclamo": "Reclamo o queja",
    "cambio_devolucion": "Cambio o devolución",
    "consulta_general": "Consulta general",
    "pedido_pendiente": "Pedido pendiente",
}

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

    #Obtenemos el codigo de la categoría predicha 
    codigoCategoria = categoriaPredicha[0]

    #Obtenemos el nombre amigable de la categoría usando el diccionario
    nombreCategoria = nombresCategorias.get(codigoCategoria, "Categoría desconocida")

    #Mostramos el mensaje y la categoría predicha
    print(f"Mensaje: {mensajePrueba} - Categoría Predicha: {nombreCategoria}")
