# Evaluacion profunda de chunks RAG

- Backend de embeddings: `onnx-minilm`
- Top-k por pregunta: `5`
- Escala: 0 hueco, 1 superficial, 2 clase basica, 3 tecnico, 4 profundo/comparativo.

## Resumen por tema

| Tema | Fuentes recuperadas | Promedio | Minimo | Alcance | Limitacion |
| --- | --- | --- | --- | --- | --- |
| `atencion_transformers` | attention_is_all_you_need.pdf | 4.00 | 4 | Profundo | Puede requerir refuerzo para complejidad computacional o variantes modernas. |
| `cnn_vision` | resnet_deep_residual_learning.pdf | 3.75 | 3 | Profundo | Fuerte en ResNet; CNN basica y transfer learning didactico pueden requerir fuente adicional. |
| `entrenamiento_optimizacion` | adam_method_for_stochastic_optimization.pdf | 4.00 | 4 | Profundo | Fuerte en Adam; comparacion completa con SGD/Momentum/RMSProp puede requerir capitulo didactico. |
| `explicabilidad` | grad_cam.pdf | 3.25 | 3 | Fuerte | Fuerte en Grad-CAM; saliency maps generales y limites filosoficos quedan menos cubiertos. |
| `generativos` | generative_adversarial_nets.pdf, vae_auto_encoding_variational_bayes.pdf | 3.75 | 3 | Profundo | Cubre VAE/GAN; autoencoders basicos, denoising y estabilidad practica pueden requerir complemento. |
| `regularizacion_estabilidad` | batch_normalization.pdf, dropout_jmlr.pdf | 4.00 | 4 | Profundo | Fuerte en Dropout/BatchNorm; weight decay, early stopping e inicializacion quedan menos cubiertos. |
| `secuenciales_rnn_lstm_gru` | gru_rnn_encoder_decoder.pdf | 3.50 | 3 | Profundo | Cubre GRU; LSTM, RNN simple y vanishing gradients necesitan fuente local mas directa. |

## Preguntas exigentes

| Tema | Tipo | Pregunta | Nivel | Evidencia |
| --- | --- | --- | --- | --- |
| `atencion_transformers` | definicion | Que problema resuelve self-attention frente a modelos recurrentes? | 4 | terminos=7, senales_profundidad=4, fuentes_top3=3, fuente_top1=attention_is_all_you_need.pdf p.2 |
| `atencion_transformers` | comparacion | Compara self-attention, multi-head attention y positional encoding. | 4 | terminos=6, senales_profundidad=3, fuentes_top3=3, fuente_top1=attention_is_all_you_need.pdf p.5 |
| `atencion_transformers` | aplicacion | Cuando conviene usar un Transformer encoder y cuando un decoder? | 4 | terminos=6, senales_profundidad=3, fuentes_top3=3, fuente_top1=attention_is_all_you_need.pdf p.5 |
| `atencion_transformers` | diagnostico | Que limitaciones computacionales tiene attention en secuencias largas? | 4 | terminos=6, senales_profundidad=3, fuentes_top3=3, fuente_top1=attention_is_all_you_need.pdf p.2 |
| `cnn_vision` | definicion | Que es una conexion residual en ResNet? | 3 | terminos=4, senales_profundidad=3, fuentes_top3=3, fuente_top1=resnet_deep_residual_learning.pdf p.1 |
| `cnn_vision` | comparacion | Compara una red CNN profunda convencional con una ResNet. | 4 | terminos=5, senales_profundidad=3, fuentes_top3=3, fuente_top1=resnet_deep_residual_learning.pdf p.12 |
| `cnn_vision` | aplicacion | Cuando conviene usar ResNet o transfer learning en vision por computador? | 4 | terminos=4, senales_profundidad=3, fuentes_top3=3, fuente_top1=resnet_deep_residual_learning.pdf p.1 |
| `cnn_vision` | diagnostico | Por que una red profunda puede degradar su entrenamiento y como ayuda ResNet? | 4 | terminos=7, senales_profundidad=3, fuentes_top3=3, fuente_top1=resnet_deep_residual_learning.pdf p.5 |
| `entrenamiento_optimizacion` | definicion | Que es Adam y que estimaciones mantiene durante el entrenamiento? | 4 | terminos=6, senales_profundidad=1, fuentes_top3=3, fuente_top1=adam_method_for_stochastic_optimization.pdf p.1 |
| `entrenamiento_optimizacion` | comparacion | Compara Adam con SGD, Momentum y RMSProp. | 4 | terminos=6, senales_profundidad=2, fuentes_top3=3, fuente_top1=adam_method_for_stochastic_optimization.pdf p.5 |
| `entrenamiento_optimizacion` | aplicacion | Cuando conviene usar Adam en lugar de descenso del gradiente clasico? | 4 | terminos=6, senales_profundidad=1, fuentes_top3=3, fuente_top1=adam_method_for_stochastic_optimization.pdf p.1 |
| `entrenamiento_optimizacion` | diagnostico | Que problemas de tasa de aprendizaje o gradientes intenta mitigar Adam? | 4 | terminos=6, senales_profundidad=1, fuentes_top3=3, fuente_top1=adam_method_for_stochastic_optimization.pdf p.1 |
| `explicabilidad` | definicion | Que es Grad-CAM y que mapa produce? | 3 | terminos=5, senales_profundidad=2, fuentes_top3=3, fuente_top1=grad_cam.pdf p.3 |
| `explicabilidad` | comparacion | Compara Grad-CAM con saliency maps o class activation maps. | 3 | terminos=4, senales_profundidad=2, fuentes_top3=3, fuente_top1=grad_cam.pdf p.3 |
| `explicabilidad` | aplicacion | Como se usa Grad-CAM para interpretar una prediccion de una CNN? | 3 | terminos=4, senales_profundidad=1, fuentes_top3=3, fuente_top1=grad_cam.pdf p.21 |
| `explicabilidad` | diagnostico | Que limitaciones tiene Grad-CAM para explicar decisiones del modelo? | 4 | terminos=4, senales_profundidad=3, fuentes_top3=3, fuente_top1=grad_cam.pdf p.3 |
| `generativos` | definicion | Que es una variable latente en un VAE y que aprende el modelo? | 4 | terminos=6, senales_profundidad=2, fuentes_top3=3, fuente_top1=vae_auto_encoding_variational_bayes.pdf p.8 |
| `generativos` | comparacion | Compara VAE y GAN en objetivo de entrenamiento y tipo de salida. | 4 | terminos=4, senales_profundidad=2, fuentes_top3=3, fuente_top1=generative_adversarial_nets.pdf p.1 |
| `generativos` | aplicacion | Cuando conviene usar un VAE y cuando una GAN? | 4 | terminos=6, senales_profundidad=2, fuentes_top3=3, fuente_top1=generative_adversarial_nets.pdf p.2 |
| `generativos` | diagnostico | Que problemas de estabilidad o evaluacion aparecen en modelos generativos? | 3 | terminos=6, senales_profundidad=1, fuentes_top3=3, fuente_top1=generative_adversarial_nets.pdf p.6 |
| `regularizacion_estabilidad` | definicion | Que es dropout y como reduce el sobreajuste? | 4 | terminos=5, senales_profundidad=3, fuentes_top3=3, fuente_top1=dropout_jmlr.pdf p.24 |
| `regularizacion_estabilidad` | comparacion | Compara dropout, Batch Normalization y weight decay. | 4 | terminos=8, senales_profundidad=4, fuentes_top3=3, fuente_top1=batch_normalization.pdf p.5 |
| `regularizacion_estabilidad` | aplicacion | Cuando conviene usar BatchNorm durante el entrenamiento de redes profundas? | 4 | terminos=8, senales_profundidad=4, fuentes_top3=3, fuente_top1=batch_normalization.pdf p.3 |
| `regularizacion_estabilidad` | diagnostico | Como se relacionan normalizacion, gradientes inestables y velocidad de entrenamiento? | 4 | terminos=7, senales_profundidad=3, fuentes_top3=3, fuente_top1=batch_normalization.pdf p.4 |
| `secuenciales_rnn_lstm_gru` | definicion | Que es una GRU y que puertas utiliza? | 3 | terminos=2, senales_profundidad=4, fuentes_top3=3, fuente_top1=gru_rnn_encoder_decoder.pdf p.5 |
| `secuenciales_rnn_lstm_gru` | comparacion | Compara RNN simple, LSTM y GRU. | 4 | terminos=5, senales_profundidad=3, fuentes_top3=3, fuente_top1=gru_rnn_encoder_decoder.pdf p.3 |
| `secuenciales_rnn_lstm_gru` | aplicacion | Cuando conviene usar un modelo recurrente para secuencias o traduccion? | 4 | terminos=2, senales_profundidad=4, fuentes_top3=3, fuente_top1=gru_rnn_encoder_decoder.pdf p.5 |
| `secuenciales_rnn_lstm_gru` | diagnostico | Como ayudan las compuertas a mitigar vanishing gradients en secuencias largas? | 3 | terminos=2, senales_profundidad=3, fuentes_top3=3, fuente_top1=gru_rnn_encoder_decoder.pdf p.4 |
