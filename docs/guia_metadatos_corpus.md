# Guia de metadatos del corpus

## Ramas actuales

Por lo que existe hoy en GitHub, el corpus tiene dos ramas practicas:

| Rama | Significado | Uso |
| --- | --- | --- |
| `deep_learning` | Temas propios del programa de Deep Learning | Rama principal para la presentacion del 22 de mayo |
| `machine_learning_estadistica` | Libros de Statistical Learning, ML clasico y fundamentos estadisticos | Rama secundaria/de apoyo; no debe mezclarse como si fuera Deep Learning |

No usaria el nombre `estadistica` solo, porque los archivos que subio Mafe no son estadistica pura: son principalmente **Machine Learning con base estadistica**.

## Archivo principal

Los metadatos estan en:

```text
docs/metadatos_temas_corpus.csv
```

## Campos

| Campo | Para que sirve |
| --- | --- |
| `id` | Identificador corto y estable de la fuente o capitulo |
| `branch` | Rama academica: `deep_learning` o `machine_learning_estadistica` |
| `topic` | Bloque grande del programa |
| `subtopic` | Tema especifico dentro del bloque |
| `chapter_hint` | Capitulo, paper o seccion que conviene usar |
| `source_type` | Tipo de fuente: `pdf_repo`, `paper`, `capitulo_online`, etc. |
| `source_path_or_title` | Ruta del repo o titulo formal de la fuente |
| `url` | Enlace de referencia |
| `scope` | Si es fuente general, tema especifico, mixto o fuera del nucleo DL |
| `use_in_retrieval` | `yes`, `no` o `conditional` |
| `priority` | Alta, media o baja |
| `notes` | Observacion breve para evitar cruces o mal uso |

## Uso en ingesta

Cuando se procese un documento, cada chunk deberia conservar al menos:

```python
metadata = {
    "branch": "deep_learning",
    "topic": "cnn_vision",
    "subtopic": "resnet_conexiones_residuales",
    "chapter": "Deep Residual Learning for Image Recognition",
    "source_type": "paper",
    "source": "resnet.pdf",
    "page": page_number,
}
```

## Uso en retrieval

Si el usuario pregunta algo de Deep Learning, el agente debe priorizar:

```python
filter = {"branch": "deep_learning"}
```

Si la pregunta es mas especifica, por ejemplo ResNet:

```python
filter = {
    "branch": "deep_learning",
    "topic": "cnn_vision"
}
```

Las fuentes con `use_in_retrieval = "no"` no deberian entrar en busquedas normales de Deep Learning. Las de `conditional` solo se usan cuando el tema lo justifique.

## Uso en presentacion

La idea se puede explicar asi:

> Organizamos el corpus con metadatos academicos. Cada chunk sabe a que rama, tema y subtema pertenece. Esto evita que el agente mezcle libros de Machine Learning clasico con preguntas de Deep Learning y permite mostrar una tabla clara de cobertura.

## Decision importante

No hace falta borrar lo que subio Mafe. El cambio es agregar una capa de metadatos para controlar el alcance.
