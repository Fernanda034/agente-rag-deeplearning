# Registro de rama: Deep Learning

## Objetivo de la rama
Construir primero la base documental de Deep Learning dentro de una futura estructura mas amplia de Ciencia de Datos. Por tiempo, la rama se trabajara tema por tema, y ningun documento quedara aprobado para el corpus hasta que Juan Camilo lo valide.

## Estado general
- Rama: Deep Learning
- Documento guia: `DeepLearning2026 (1).pdf`
- Fecha de presentacion del curso: 22 de mayo
- Estado del corpus: En revision
- Responsable de ingesta y procesamiento: Juan Camilo Puentes

## Temas detectados en el programa
- Fundamentos: AI vs ML vs DL, tensores, shapes, MLP, capas densas, activaciones y funciones de perdida.
- Entrenamiento: descenso del gradiente, SGD, Momentum, RMSProp, Adam, schedulers y backpropagation.
- Estabilidad y regularizacion: vanishing/exploding gradients, dropout, weight decay, early stopping, inicializacion, normalizacion y BatchNorm.
- MLP aplicado: clasificacion binaria, multiclase, regresion y pipeline completo.
- Modelos secuenciales: RNN, series de tiempo, embeddings, tokenizacion, padding, LSTM y GRU.
- Atencion y Transformers: atencion basica, self-attention, multi-head attention, encoder-decoder y positional encoding.
- Vision por computador: convoluciones, CNN, filtros, stride, padding, canales, data augmentation y transfer learning.
- Explicabilidad: saliency maps, Grad-CAM y visualizacion de atencion.
- Modelos generativos: autoencoders, denoising autoencoders, deteccion de anomalias, VAE, GANs y estabilidad de GANs.

## Checkpoints de preprocesamiento
- [ ] Registrar documento candidato antes de ingresarlo al corpus.
- [ ] Confirmar que el documento pertenece a la rama Deep Learning.
- [ ] Guardar fuente, titulo, autores, ano, tema y enlace.
- [ ] Validar si el PDF es texto seleccionable o escaneado.
- [ ] Extraer texto por pagina.
- [ ] Eliminar paginas vacias o con texto inutil.
- [ ] Limpiar saltos raros, espacios duplicados, encabezados y pies repetidos cuando afecten la busqueda.
- [ ] Conservar metadatos minimos: `source`, `page`, `title`, `topic`, `branch`.
- [ ] Dividir en chunks con overlap.
- [ ] Revisar manualmente una muestra de chunks.
- [ ] Marcar el documento como aprobado, rechazado o pendiente.

## Estructura propuesta del corpus
```text
corpus/
  ciencia_datos/
    deep_learning/
      00_programa_curso/
      01_fundamentos_mlp/
      02_entrenamiento_optimizacion/
      03_regularizacion_estabilidad/
      04_secuenciales_rnn_lstm_gru/
      05_atencion_transformers/
      06_cnn_vision/
      07_explicabilidad/
      08_autoencoders_vae_gans/
      _aprobados/
      _pendientes/
      _rechazados/
```

## Campos sugeridos para registrar fuentes
| Campo | Descripcion |
| --- | --- |
| `id` | Identificador corto del documento. |
| `rama` | Deep Learning, Machine Learning, MLOps, etc. |
| `tema` | Tema del programa al que aporta. |
| `titulo` | Titulo formal del documento. |
| `autores` | Autores principales. |
| `ano` | Ano de publicacion. |
| `tipo` | Paper, libro, capitulo, guia, diapositivas o documentacion. |
| `url` | Enlace fuente. |
| `estado` | Pendiente, aprobado o rechazado. |
| `motivo` | Justificacion breve de la decision. |

## Corpus candidato pendiente de aprobacion
| Estado | Tema | Documento | Fuente |
| --- | --- | --- | --- |
| Pendiente | Fundamentos generales | Deep Learning, Goodfellow, Bengio y Courville | https://www.deeplearningbook.org/ |
| Pendiente | Implementacion practica | Dive into Deep Learning | https://d2l.ai/ |
| Pendiente | Fundamentos/MLP/backpropagation | Neural Networks and Deep Learning, Michael Nielsen | http://neuralnetworksanddeeplearning.com/ |
| Pendiente | Backpropagation | Learning representations by back-propagating errors | https://www.nature.com/articles/323533a0 |
| Pendiente | Optimizacion | Adam: A Method for Stochastic Optimization | https://arxiv.org/abs/1412.6980 |
| Pendiente | Regularizacion | Dropout: A Simple Way to Prevent Neural Networks from Overfitting | https://jmlr.org/papers/v15/srivastava14a.html |
| Pendiente | Inicializacion/estabilidad | Understanding the difficulty of training deep feedforward neural networks | https://proceedings.mlr.press/v9/glorot10a.html |
| Pendiente | BatchNorm | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | https://arxiv.org/abs/1502.03167 |
| Pendiente | RNN/LSTM | Long Short-Term Memory | https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory |
| Pendiente | GRU | Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation | https://arxiv.org/abs/1406.1078 |
| Pendiente | Transformers | Attention Is All You Need | https://arxiv.org/abs/1706.03762 |
| Pendiente | CNN / vision | Deep Residual Learning for Image Recognition | https://arxiv.org/abs/1512.03385 |
| Pendiente | CNN / vision | An Introduction to Convolutional Neural Networks | https://arxiv.org/abs/1511.08458 |
| Pendiente | Explicabilidad | Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization | https://arxiv.org/abs/1610.02391 |
| Pendiente | BERT / NLP | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | https://arxiv.org/abs/1810.04805 |
| Pendiente | GANs | Generative Adversarial Networks | https://arxiv.org/abs/1406.2661 |
| Pendiente | VAE | Auto-Encoding Variational Bayes | https://arxiv.org/abs/1312.6114 |

## Regla de aprobacion del corpus
Un documento solo pasa a `_aprobados/` cuando cumpla estas condiciones:
- Aporta directamente a un tema del programa.
- Tiene fuente confiable: libro oficial, paper original, editorial, arXiv, JMLR, PMLR, MIT Press, Keras/TensorFlow o documentacion academica reconocida.
- Su texto puede extraerse con calidad suficiente.
- No duplica de forma innecesaria otro documento ya aprobado.
- Juan Camilo lo aprueba explicitamente.

## Pendientes para fortalecer antes de la presentacion
- Definir cuantos PDFs entran en la primera version de la rama Deep Learning.
- Aprobar o rechazar el corpus candidato.
- Elegir el tamano de chunk y overlap que se explicara en la presentacion.
- Preparar una tabla simple con: pregunta, fuente recuperada, pagina y observacion.
- Aclarar que ChromaDB queda para el modulo de almacenamiento, pero depende de que la ingesta entregue chunks limpios.
