# 🎵 RamDomMusicSeparate: Red Neuronal para la Separación de Instrumentos

## Descripción General
**DeepAudioSplit** es un proyecto de investigación y desarrollo que aplica técnicas de **aprendizaje profundo (Deep Learning)** para la **separación de fuentes musicales**, permitiendo aislar instrumentos como **voz, bajo, percusión y acompañamiento** a partir de una mezcla estéreo completa.  
El modelo se basa en una arquitectura tipo **Encoder-Decoder (UNet)** con **módulos de atención** y un **bloque de refinamiento inspirado en modelos de difusión**, ofreciendo resultados de alta calidad perceptual.

---

## Objetivo del Proyecto
Desarrollar una **alternativa abierta, libre de licencias y altamente personalizable** frente a modelos comerciales de separación musical como **Demucs**, **Spleeter** o **Wave-U-Net**.  
El proyecto busca favorecer la **reproducibilidad científica**, la **escalabilidad** y el **control completo sobre los hiperparámetros y el flujo de entrenamiento**.

---

## Características Principales

| Característica | Descripción |
|----------------|--------------|
| **Arquitectura** | UNet híbrida con módulos de atención y difusión |
| **Dominio de trabajo** | Espectrogramas complejos (STFT) |
| **Entradas** | Magnitud logarítmica de la mezcla estéreo |
| **Salidas** | Espectrograma de la fuente objetivo (voz, batería, etc.) |
| **Frecuencia de muestreo** | 16 kHz |
| **Duración de segmento (CHUNK_SIZE)** | 4 segundos |
| **Dataset principal** | MUSDB18-HQ |
| **Framework** | PyTorch 2.4.0 |
| **Optimización** | Adam — LR: 1e-4 |
| **Funciones de pérdida** | Combinación de L1 + MSE + pérdidas espectrales específicas |

---

## Arquitectura del Modelo

### Primera Fase: **UNet Pura**
Red convolucional simétrica con *skip connections* enfocada en la reconstrucción de espectrogramas. Ideal para capturar patrones espectrales y espaciales.

### Segunda Fase: **UNet + Difusión**
Una segunda red UNet actúa como **módulo de refinamiento**, inspirada en modelos de difusión (DiffWave, SpecDiff), reduciendo artefactos y mejorando la limpieza del audio resultante.

### Tercera Fase: **UNet + BLSTM/Transformers**
Se integran mecanismos temporales (BLSTM) y de atención (Transformers con *positional encoding*) para capturar dependencias a largo plazo y mejorar la coherencia temporal.

---

## Hiperparámetros y Configuración

```python
BATCH_SIZE = 4
SAMPLE_RATE = 16000
N_FFT = 4096
HOP_LENGTH = 1024
WINDOW = 'hann'
CHUNK_SIZE = 4.0
LEARNING_RATE = 1e-4
EPOCHS = 200
 ```

Los filtros convolucionales utilizados son **asimétricos (5x1 y 1x5)**, optimizados para extraer características **espectrales** y **temporales** de forma independiente.

---

## Dataset y Preprocesamiento

El modelo utiliza el conjunto **MUSDB18-HQ**, compuesto por **150 canciones** (100 para entrenamiento, 50 para prueba).  
Cada canción incluye:

- 🎤 `vocals.wav`  
- 🥁 `drums.wav`  
- 🎸 `bass.wav`  
- 🎹 `other.wav`  
- 🎶 `mixture.wav`

El dataset se procesa mediante la clase **`AudioDataset`**, que:

- Extrae segmentos aleatorios de duración configurable.  
- Realiza *data augmentation* (pitch shifting, ruido gaussiano).  
- Admite trabajo en dominio temporal o frecuencial.  
- Normaliza los tensores al rango `[-1, 1]`.

---

## Función de Pérdida

La pérdida total combina múltiples objetivos:

- **L1 + MSE** (reconstrucción base)  
- **Pérdida de separación espectral**  
- **Pérdida de máscara espectral**  
- **Pérdida de contraste estéreo**

**Ecuación general:**

\[
L_{total} = \alpha \cdot L_{base} + \beta \cdot L_{sep} + \gamma \cdot L_{mask} + \delta \cdot L_{ch\_diff}
\]

**Valores típicos:**  
`α = 0.7`, `β = 0.2`, `γ = 0.1`, `δ = 0.05`

---

## Entrenamiento

- Entrenamiento **end-to-end en PyTorch**.  
- Monitorización mediante **hooks** para analizar activaciones y gradientes.  
- **Normalización selectiva** (solo en encoders).  
- Función de activación: **ReLU**.  
- Entrenamiento principal orientado a la **pista de batería** como caso base.

---

## Herramientas de Optimización

- Registro de **activaciones y gradientes por capa**.  
- Análisis de **estabilidad numérica**.  
- **Visualización en tiempo real** de activaciones.  
- Detección de **overfitting** y generación dinámica de nuevos datasets.

---

## Resultados Cuantitativos (SDR)

Los resultados se evaluaron utilizando la métrica **Signal-to-Distortion Ratio (SDR)**, que mide la calidad perceptual de la separación de cada fuente.  
A continuación se presentan los valores obtenidos para las pistas de **bajo**, **batería**, **otros** y **voces** sobre un conjunto de prueba de 10 canciones.

| # | Bass (dB) | Drums (dB) | Other (dB) | Vocals (dB) |
|:-:|:-------------:|:-------------:|:-------------:|:--------------:|
| 1 | 2.49 | 1.87 | 4.33 | 3.61 |
| 2 | 2.65 | 2.87 | 1.30 | 3.29 |
| 3 | 2.70 | 1.58 | 4.30 | 4.21 |
| 4 | 3.66 | 2.78 | 1.51 | 1.82 |
| 5 | 2.39 | 1.02 | 1.01 | 2.80 |
| 6 | 3.77 | 1.38 | 3.39 | 2.67 |
| 7 | 4.21 | 3.10 | 1.23 | 3.40 |
| 8 | 4.04 | 2.10 | 2.21 | 3.99 |
| 9 | 3.52 | 1.59 | 2.72 | 3.13 |
| 10 | 3.86 | 2.05 | 3.11 | 5.08 |
| **Media** | **3.32 dB** | **2.03 dB** | **2.51 dB** | **3.40 dB** |

**Interpretación:**  
El modelo muestra un rendimiento más consistente en las pistas de **bajo** y **voces**, mientras que los instrumentos de percusión presentan una mayor complejidad en la separación debido a su naturaleza transitoria.  
Estos resultados reflejan una mejora perceptible frente a modelos base tradicionales, manteniendo una separación limpia y sin artefactos significativos.


