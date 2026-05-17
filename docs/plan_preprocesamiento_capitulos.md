# Plan de preprocesamiento por capitulos

## Problema

El repo actual tiene PDFs completos dentro de `corpus/`. Eso sirve para un prototipo rapido, pero tiene tres riesgos:

1. Un libro grande puede mezclar muchos temas en la misma busqueda.
2. El agente puede recuperar fragmentos generales cuando el usuario pregunta algo especifico.
3. Es dificil explicar hasta donde llega el agente si no hay una tabla de cobertura por tema.

## Decision

Mantener los PDFs actuales y agregar una capa de organizacion mediante `manifest_corpus_deep_learning.csv`.

No es obligatorio mover archivos fisicamente. El manifest indica que fuente cubre cada tema, y despues el pipeline puede usar esos metadatos para mejorar recuperacion.

## Estrategia Para Libros Grandes

### Caso: `Dive_into_deep.pdf`

No tratarlo como un unico documento de Deep Learning. Tratarlo como una fuente general con capitulos especificos.

Estrategia:

1. Conservar el PDF completo como respaldo.
2. Identificar capitulos utiles para cada bloque academico.
3. Durante la ingesta, asignar metadatos:

```python
metadata = {
    "branch": "deep_learning",
    "topic": "secuenciales_rnn_lstm_gru",
    "chapter": "Modern Recurrent Neural Networks",
    "source": "Dive_into_deep.pdf",
    "page": page_number,
}
```

4. Si no se conocen paginas exactas, empezar con tema por archivo y luego refinar por rangos.
5. Para demo, probar retrieval con preguntas de cada bloque.

## Que Hacer Si Un Capitulo Depende De Otro

Si un capitulo especifico depende de conceptos anteriores, no se debe meter todo el libro sin control.

Ejemplo:

- Pregunta: "Como funciona LSTM?"
- Fuente principal: capitulo de LSTM/GRU.
- Fuente de apoyo: capitulo corto de RNN basica o embeddings si la respuesta lo necesita.

Esto se puede manejar con:

- overlap entre chunks;
- metadatos `topic` y `chapter`;
- incluir capitulos previos solo cuando sean necesarios;
- tabla de cobertura indicando dependencias.

## Mejoras Propuestas Al Script De Ingesta

El script actual ya hace:

- carga de PDFs;
- limpieza con regex;
- chunking de 1000 caracteres;
- overlap de 200;
- embeddings con `all-MiniLM-L6-v2`;
- ChromaDB local;
- indexacion incremental con `SQLRecordManager`.

Mejoras sugeridas:

1. Leer `manifest_corpus_deep_learning.csv`.
2. Agregar metadatos `branch`, `topic`, `chapter`, `source_type`.
3. Separar fuentes de Deep Learning de fuentes de ML clasico.
4. Permitir filtrar retrieval por tema cuando el usuario pregunte algo especifico.
5. Crear reporte post-ingesta con numero de paginas/chunks por tema.
6. Guardar una muestra de chunks por tema para revisar profundidad.

## Tabla De Reporte Post-Ingesta

Despues de correr ingesta, generar una tabla asi:

| Tema | Documentos | Paginas | Chunks | Estado |
| --- | --- | --- | --- | --- |
| Transformers | Attention paper, BERT, D2L | - | - | Fuerte |
| CNN/ResNet | D2L, ResNet paper | - | - | Por probar |
| VAE | VAE paper | - | - | Por probar |

## Argumento Para La Presentacion

> No solo cargamos PDFs completos. Organizamos el corpus por temas del programa y usamos metadatos para que el agente pueda recuperar fragmentos especificos. Esto permite explicar con claridad que temas cubre, que fuentes usa y en que puntos todavia puede tener limitaciones.

## Proximo Paso Tecnico

Antes de tocar el agente:

1. Integrar el manifest al script de ingesta.
2. Ejecutar ingesta.
3. Correr pruebas de retrieval por tema.
4. Construir tabla de alcance del agente.
