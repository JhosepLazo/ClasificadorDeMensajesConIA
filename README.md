# Clasificador Inteligente de Mensajes para Tienda

Este proyecto se va a construir paso a paso con enfoque de aprendizaje.

La idea es crear un sistema que reciba mensajes de clientes y los clasifique en categorias como stock, reclamo, cambio/devolucion, pedido pendiente o consulta general.

## Objetivo de esta etapa
Por ahora el proyecto queda en una base minima para que puedas escribir el codigo tu mismo.

No hay modelo entrenado todavia.
No hay demo web todavia.
No hay scripts terminados todavia.

Vamos a avanzar por partes y entendiendo cada linea.

## Estructura actual
```text
ClasificadorDeMensajesConIA/
├── data/
│   └── mensajes_clientes.csv
├── src/
├── .gitignore
└── README.md
```

## Que contiene cada parte
- `data/mensajes_clientes.csv`: dataset inicial con mensajes ficticios clasificados.
- `src/`: carpeta donde escribiremos los scripts de Python.
- `.gitignore`: archivo para evitar subir carpetas temporales como `.venv` o `__pycache__`.
- `README.md`: explicacion del proyecto.

## Categorias iniciales
| Codigo | Significado |
| --- | --- |
| `stock` | Consulta de stock |
| `reclamo` | Reclamo |
| `cambio_devolucion` | Cambio o devolucion |
| `pedido_pendiente` | Pedido pendiente |
| `consulta_general` | Consulta general |

## Proximo paso
El primer archivo que escribiremos sera:

```text
src/train_model.py
```

Pero antes de programarlo, conviene entender estas preguntas:

1. Como leer un archivo CSV con Python.
2. Como separar mensajes y categorias.
3. Que significa entrenar un modelo.
4. Que hace TF-IDF.
5. Que hace un clasificador.

## Regla de trabajo
Tu escribes el codigo.
Yo te guio, reviso, explico y corrijo cuando lo necesites.
