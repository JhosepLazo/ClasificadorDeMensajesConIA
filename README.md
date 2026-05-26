# Clasificador Inteligente de Mensajes para Tienda

## Descripción

Este proyecto implementa un clasificador inteligente de mensajes para una tienda.  
El sistema recibe mensajes escritos por clientes y los clasifica automáticamente en una categoría comercial.

## Categorías del modelo

| Categoría | Descripción |
|---|---|
| stock | Consulta de disponibilidad de productos |
| reclamo | Reclamos por problemas con pedidos o atención |
| cambio_devolucion | Solicitudes de cambio o devolución |
| pedido_pendiente | Consultas sobre pedidos pendientes o envíos |
| consulta_general | Preguntas generales sobre la tienda |

## Tecnologías utilizadas

- Python
- pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- joblib

## Estructura del proyecto

```text
ClasificadorDeMensajesConIA/
├── data/
│   └── mensajes_clientes.csv
├── models/
│   └── modelo_clasificador.joblib
├── reports/
│   ├── metricas_modelo.txt
│   └── matriz_confusion.csv
├── src/
│   ├── train_model.py
│   └── predict_message.py
├── requirements.txt
├── .gitignore
└── README.md