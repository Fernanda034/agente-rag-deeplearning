# Cobertura del corpus Deep Learning

Este archivo resume que temas del programa quedan cubiertos por el corpus versionado y que limitaciones conviene declarar.

## Corpus actual

| Tema | Fuente principal | Estado | Observacion |
| --- | --- | --- | --- |
| Atencion y Transformers | `attention_is_all_you_need.pdf` | Cubierto | Fuente fuerte para self-attention, multi-head attention y positional encoding. |
| BERT / NLP | `bert.pdf` | Cubierto | Fuente especifica para BERT; no reemplaza una guia general de NLP. |
| CNN / ResNet | `resnet_deep_residual_learning.pdf` | Cubierto | Fuente fuerte para conexiones residuales en redes profundas. |
| Optimizacion | `adam_method_for_stochastic_optimization.pdf` | Cubierto | Fuente especifica para Adam y optimizacion adaptativa. |
| Regularizacion | `dropout_jmlr.pdf` | Cubierto | Fuente fuerte para dropout y sobreajuste. |
| Normalizacion / estabilidad | `batch_normalization.pdf` | Cubierto | Fuente fuerte para Batch Normalization. |
| Secuenciales / GRU | `gru_rnn_encoder_decoder.pdf` | Parcial | Cubre GRU y encoder-decoder recurrente; LSTM queda menos fuerte si no se agrega fuente propia. |
| Explicabilidad | `grad_cam.pdf` | Cubierto | Fuente fuerte para Grad-CAM en vision. |
| VAE | `vae_auto_encoding_variational_bayes.pdf` | Cubierto | Fuente especifica para inferencia variacional y espacio latente probabilistico. |
| GANs | `generative_adversarial_nets.pdf` | Cubierto | Fuente original de GANs. |

## Cobertura por bloque academico

| Bloque del programa | Nivel actual | Limitacion declarable |
| --- | --- | --- |
| Fundamentos, tensores y MLP | Basico | Falta una fuente local especifica de MLP/tensores si se quiere responder desde cero con mas didactica. |
| Entrenamiento y optimizacion | Fuerte | Adam esta cubierto; otros optimizadores pueden depender de material general o explicacion del modelo. |
| Regularizacion y estabilidad | Fuerte | Dropout y BatchNorm estan bien cubiertos. |
| Modelos secuenciales | Basico-Medio | GRU esta cubierto; LSTM y series de tiempo quedan como mejora posterior. |
| Atencion y Transformers | Fuerte | Attention y BERT dan buena base tecnica. |
| Vision por computador | Fuerte para ResNet | Faltaria material didactico de CNN basica si el usuario pregunta desde cero. |
| Explicabilidad | Fuerte para Grad-CAM | Saliency maps generales no estan tan cubiertos. |
| Generativos | Fuerte para VAE/GAN | Autoencoders basicos y denoising autoencoders quedan como mejora posterior. |

## Resultado de evaluacion

La tabla basica de alcance generada por retrieval esta en:

```text
docs/tabla_alcance_agente.md
```

La evaluacion profunda de chunks esta en:

```text
docs/evaluacion_profundidad_chunks.md
```

Lectura rapida:

- Nivel fuerte en attention/Transformers, CNN/ResNet, Adam, Grad-CAM, VAE/GAN, dropout y BatchNorm.
- Nivel basico en secuenciales porque el corpus local tiene GRU, pero no una fuente LSTM local equivalente.

Lectura de profundidad:

- Nivel profundo en attention/Transformers, Adam, regularizacion/estabilidad y generativos VAE/GAN.
- Nivel profundo con limitacion declarada en CNN/ResNet y secuenciales/GRU.
- Nivel fuerte en explicabilidad con Grad-CAM; no cubre toda la familia de metodos interpretables.

## Decision para primera version

La parte de ingesta y preparacion del corpus esta lista para subir como v1 academica. Para no sobredimensionar el alcance, conviene presentarla como:

```text
Corpus RAG de Deep Learning con metadatos, chunking, ChromaDB y evaluacion inicial de cobertura.
```

No venderla todavia como agente final completo, porque el siguiente paso es integrar el retriever con el LLM generador y exigir citas por fuente y pagina en cada respuesta.
