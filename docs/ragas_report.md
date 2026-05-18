# Reporte de Evaluación RAGAS

- Modelo juez: `llama-3.1-8b-instant` (Groq)
- Embeddings: `all-MiniLM-L6-v2` (ONNX)
- Top-k contextos: `2`
- Preguntas evaluadas: `7`

> **Nota metodológica:** `answer_relevancy` devuelve `nan` porque la métrica requiere
> un modelo de embeddings con interfaz `embed_query` compatible con RAGAS ≥ 0.2.
> El resto de métricas se calcularon correctamente con `LangchainLLMWrapper` + Groq.
> Los `N/A` en el detalle por pregunta corresponden a timeouts en la evaluación paralela;
> los promedios globales se calculan sobre las filas sin error.

## Promedios globales

| Métrica | Valor | Interpretación |
| --- | --- | --- |
| `faithfulness` | 0.5500 | Fidelidad de la respuesta al contexto recuperado (1.0 = perfecta) |
| `answer_relevancy` | nan | Relevancia de la respuesta a la pregunta (1.0 = perfecta) |
| `context_precision` | 1.0000 | Precisión del contexto recuperado (proporción útil) |
| `context_recall` | 0.5556 | Cobertura del contexto respecto al ground truth |

## Detalle por pregunta

| Tema | `faithfulness` | `answer_relevancy` | `context_precision` | `context_recall` |
| --- | --- | --- | --- | --- |
| `atencion_transformers` | 0.667 | N/A | N/A | N/A |
| `entrenamiento_optimizacion` | 0.800 | N/A | N/A | N/A |
| `regularizacion_estabilidad` | 0.200 | N/A | N/A | N/A |
| `cnn_vision` | N/A | N/A | 1.000 | 0.667 |
| `generativos` | 0.400 | N/A | 1.000 | N/A |
| `secuenciales_rnn_lstm_gru` | 0.400 | N/A | 1.000 | 1.000 |
| `explicabilidad` | 0.833 | N/A | 1.000 | 0.000 |
