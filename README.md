# Sistemas de Detección de Ataques DDoS basados en Deep Learning en Entornos de Red

Repositorio oficial de respaldo experimental para el póster de investigación **"Sistemas de Detección de Ataques DDoS basados en Deep Learning en Entornos de Red"**. Este proyecto implementa una arquitectura modular con `main.py` como nodo central de ejecución sobre los múltiples archivos CSV del dataset **CIC-DDoS2019** ubicados en la carpeta `data/`.

---

## Tabla de Contenidos

- [Sistemas de Detección de Ataques DDoS basados en Deep Learning en Entornos de Red](#sistemas-de-detección-de-ataques-ddos-basados-en-deep-learning-en-entornos-de-red)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [1. Introducción y Contexto](#1-introducción-y-contexto)
    - [Taxonomía de Enfoques del Estado del Arte (2019–2024)](#taxonomía-de-enfoques-del-estado-del-arte-20192024)
    - [Comparativo de Modelos Neuronales de la Literatura](#comparativo-de-modelos-neuronales-de-la-literatura)
    - [Brechas Identificadas](#brechas-identificadas)
  - [2. Metodología: Protocolo Común sobre CIC-DDoS2019](#2-metodología-protocolo-común-sobre-cic-ddos2019)
    - [Los 19 Atributos del Protocolo](#los-19-atributos-del-protocolo)
  - [3. Resultados Experimentales del Póster](#3-resultados-experimentales-del-póster)
    - [Comparación Homogénea de Modelos](#comparación-homogénea-de-modelos)
    - [Matriz de Confusión de LightGBM (Conjunto de Prueba)](#matriz-de-confusión-de-lightgbm-conjunto-de-prueba)
    - [Aportes en Interpretabilidad (XAI) y Calibración (ECE)](#aportes-en-interpretabilidad-xai-y-calibración-ece)
  - [4. Arquitectura Modular](#4-arquitectura-modular)
    - [Estructura del Proyecto](#estructura-del-proyecto)
    - [Diseño de Clases y Módulos](#diseño-de-clases-y-módulos)
  - [5. Instalación y Ejecución](#5-instalación-y-ejecución)
    - [Requisitos](#requisitos)
    - [Parámetros de Línea de Comandos (`main.py`)](#parámetros-de-línea-de-comandos-mainpy)
    - [Ejemplos de Uso](#ejemplos-de-uso)
  - [6. Artefactos de Salida](#6-artefactos-de-salida)
  - [7. Referencias Principales](#7-referencias-principales)

---

## 1. Introducción y Contexto

Los ataques DDoS saturan servicios críticos y camuflan su tráfico imitando patrones legítimos, haciendo que los sistemas de detección de intrusos (IDS) basados en firmas resulten insuficientes. Aunque la literatura reciente propone soluciones con **Deep Learning**, cada estudio emplea conjuntos de datos, preprocesamientos y métricas heterogéneas que impiden comparaciones directas.

Este trabajo **revisa 8 modelos neuronales del estado del arte (2019–2024)** y establece un **protocolo experimental homogéneo sobre CIC-DDoS2019** para determinar con rigor cuál arquitectura ofrece el mejor desempeño, explicabilidad y calibración probabilística.

### Taxonomía de Enfoques del Estado del Arte (2019–2024)

- **CNN pura**: Extrae patrones espaciales sin ingeniería manual de atributos (*Aktar & Nur, 2023; Pei et al., 2019*).
- **Residual (ResNet)**: Redes profundas sin degradación de gradiente orientadas a entornos IoT (*Hussain et al., 2020*).
- **Híbridos supervisados**: Combinan capas espaciales (CNN) y temporales (LSTM), integrando en ocasiones gradient boosting (*Al-zubidi et al., 2024; Mousa & Abdullah, 2023*).
- **No supervisados / probabilísticos**: Arquitecturas LSTM-Autoencoder y Bi-LSTM + GMM orientadas a ataques de día cero (*Wei et al., 2023; Shieh et al., 2021*).
- **Ensambles sobre logs**: Defensa en profundidad y análisis forense (*Musa & Odokuma, 2023*).

### Comparativo de Modelos Neuronales de la Literatura

| Trabajo | Modelo | Métrica Reportada | Recall | Limitación Principal |
| :--- | :--- | :---: | :---: | :--- |
| **Al-zubidi et al. (2024)** | **CNN-LSTM-XGBoost** | **Acc 99.86%** | **97.0%** | Complejidad alta (3 modelos integrados) |
| Mousa & Abdullah (2023) | LSTM-CNN | F1 97.40% | 97.3% | Mayor latencia; solo datos simulados |
| Musa & Odokuma (2023) | Ensamble ML+DL (logs) | Prec. 97.0% | 95.0% | Latencia alta; no opera por paquete |
| Aktar & Nur (2023) | CNN | Acc 99.50% | n/d | Caja negra; sin F1 ni Recall |
| Wei et al. (2023) | LSTM-Autoencoder | AUC 0.996 | 93.0% | Recall bajo; sensible al umbral |
| Shieh et al. (2021) | Bi-LSTM + GMM | Acc 99.40% | n/d | Costo prohibitivo en tiempo real |
| Hussain et al. (2020) | ResNet (IoT) | Acc 99.78% | 94.0% | Pesado para nodos IoT |
| Pei et al. (2019) | CNN (TCP Flood) | Acc 99.93% | n/d | No generaliza más allá de TCP Flood |

### Brechas Identificadas

1. **Recall insuficiente**: Ningún estudio reportado supera el 97.3% de sensibilidad, dejando flujos maliciosos sin detectar.
2. **Complejidad vs. despliegue**: Los modelos más precisos encadenan múltiples arquitecturas y exigen GPU; los ligeros sacrifican sensibilidad.
3. **Métricas incompletas**: Varios trabajos omiten F1-Score o Recall en escenarios altamente desbalanceados.
4. **Ausencia de interpretabilidad y calibración**: No explican en qué atributos fundamentan sus decisiones ni cuán confiable es la probabilidad estimada.
5. **Datos simulados o de nicho**: Escasa validación sobre tráfico variado y vectores múltiples.

---

## 2. Metodología: Protocolo Común sobre CIC-DDoS2019

```mermaid
flowchart LR
    D["<b>CIC-DDoS2019 (data/*.csv)</b><br/>2,660,377 flujos · 19 atributos"] --> P["<b>Preprocesamiento</b><br/>Imputación mediana · StandardScaler<br/>class_weight = balanced"]
    P --> T["<b>Enfoque A: Tabular</b><br/>Regresión Logística · MLP · MLP (Optuna) · LightGBM"]
    P --> S["<b>Enfoque B: Secuencial</b><br/>Ventanas 20×19 (stride=5) · CNN-LSTM"]
    T --> E["<b>Evaluación Integral</b><br/>F1, PR-AUC, ROC-AUC · XAI (Permutation, KernelSHAP) · Calibración (ECE, Brier)"]
    S --> E
```

- **Particiones estratificadas**: División global **80% entrenamiento / 10% validación / 10% prueba**.
- **Prevención de fuga de datos (*Data Leakage*)**: El imputador (`SimpleImputer(strategy="median")`) y el escalador (`StandardScaler`) se ajustan únicamente sobre la partición de entrenamiento.
- **Modelado secuencial**: Construcción de ventanas deslizantes de `SEQ_LEN = 20` flujos con salto `SEQ_STRIDE = 5`, asignando la etiqueta del último flujo de la ventana.

### Los 19 Atributos del Protocolo

1. `Flow Duration`
2. `Total Fwd Packets`
3. `Total Backward Packets`
4. `Total Length of Fwd Packets`
5. `Total Length of Bwd Packets`
6. `Fwd Packet Length Max`
7. `Fwd Packet Length Min`
8. `Fwd Packet Length Mean`
9. `Bwd Packet Length Max`
10. `Bwd Packet Length Min`
11. `Bwd Packet Length Mean`
12. `Flow Bytes/s`
13. `Flow Packets/s`
14. `Init_Win_bytes_forward`
15. `Init_Win_bytes_backward`
16. `act_data_pkt_fwd`
17. `min_seg_size_forward`
18. `Active Mean`
19. `Idle Mean`

---

## 3. Resultados Experimentales del Póster

### Comparación Homogénea de Modelos

| Modelo | Enfoque | F1-Score | PR-AUC | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| Regresión Logística (base) | Tabular | 0.6533 | 0.8229 | 0.7821 |
| CNN-LSTM híbrido | Secuencial | 0.8939 | 0.9527 | 0.9550 |
| MLP (Optuna) | Tabular | 0.9809 | 0.9981 | 0.9920 |
| **LightGBM** | **Tabular** | **0.9973** | **0.9999** | **0.9989** |

> **Hallazgo principal**: Contrario a la tendencia del estado del arte que favorece modelos híbridos secuenciales complejos, bajo un protocolo estandarizado sobre CIC-DDoS2019 el enfoque tabular (**LightGBM** y **MLP optimizado con Optuna**) supera ampliamente al híbrido secuencial **CNN-LSTM**.

### Matriz de Confusión de LightGBM (Conjunto de Prueba)

Evaluado sobre **532,076 flujos de prueba**:

| | Predicción: Benigno | Predicción: Ataque |
| :--- | :---: | :---: |
| **Real: Benigno** | **420,430** (Verdadero Negativo) | **553** (Falso Positivo) |
| **Real: Ataque** | **46** (Falso Negativo) | **111,047** (Verdadero Positivo) |

- **Ataques no detectados (FN)**: Solo 46 de 111,093 ataques.
- **Tasa de Falsos Positivos (FPR)**: $\approx 0.13\%$.

### Aportes en Interpretabilidad (XAI) y Calibración (ECE)

- **Interpretabilidad (XAI)**:
  - **LightGBM (Permutation Importance)**: Aprende la **firma de la herramienta atacante** (`Init_Win_bytes_forward`, `min_seg_size_forward`).
  - **CNN-LSTM (KernelSHAP)**: Aprende el **efecto volumétrico temporal** (`Flow Bytes/s`, `Flow Duration`).
- **Calibración de Confianza (ECE)**:
  - **LightGBM**: Reducción del ECE de `0.000695` $\rightarrow$ `0.000073` mediante regresión isotónica.
  - **CNN-LSTM**: Reducción del ECE de `0.024381` $\rightarrow$ `0.015389` mediante *Temperature Scaling*.

---

## 4. Arquitectura Modular

El código ha sido completamente refactorizado siguiendo principios de **Programación Orientada a Objetos (POO)**, responsabilidad única y modularidad, manteniendo `main.py` como el nodo central de ejecución.

### Estructura del Proyecto

```text
tln02poster/
├── main.py                      # Nodo central de ejecución (CLIApplication)
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Documentación integral
├── data/                        # Carpeta con múltiples archivos CSV de CIC-DDoS2019
│   ├── DrDoS_DNS.csv
│   ├── DrDoS_LDAP.csv
│   ├── DrDoS_MSSQL.csv
│   ├── DrDoS_NTP.csv
│   ├── DrDoS_NetBIOS.csv
│   ├── DrDoS_SNMP.csv
│   ├── DrDoS_SSDP.csv
│   ├── DrDoS_UDP.csv
│   ├── Syn.csv
│   ├── TFTP.csv
│   └── UDPLag.csv
├── src/                         # Paquete modular POO
│   ├── __init__.py
│   ├── config.py                # Clase ExperimentConfig (parámetros y constantes)
│   ├── dataset.py               # Clases DataSplit, DataPreprocessor y DatasetManager
│   ├── models.py                # Jerarquía BaseDetector, MLP, LightGBM, CNN-LSTM y Optuna
│   ├── evaluation.py            # Clase MetricsEvaluator (F1, PR-AUC, ROC-AUC, Matriz de Confusión)
│   ├── explainability.py        # Clase ExplainabilityAnalyzer (Permutation Importance y KernelSHAP)
│   ├── calibration.py           # Clase ConfidenceCalibrator (ECE, Brier, Isotónica, Platt, Temp)
│   ├── reporting.py             # Clase ReportGenerator (exportación JSON, CSV y gráficas)
│   └── pipeline.py              # Clase orquestadora DDoSDetectionPipeline
└── resultados_poster/           # Salidas generadas por la ejecución
```

### Diseño de Clases y Módulos

- [`CLIApplication`] (`main.py`): Punto de entrada que procesa argumentos de consola (con `--data data` por defecto) e instancia el pipeline.
- [`ExperimentConfig`] (`src/config.py`): Encapsula constantes del protocolo (19 atributos, longitudes de ventana, hiperparámetros de la tesis y rutas).
- [`DataPreprocessor`] y [`DatasetManager`] (`src/dataset.py`): Gestionan la lectura multi-CSV desde `data/`, limpieza de etiquetas, submuestreo preservando el orden temporal y particionamiento estratificado tabular y secuencial sin fuga de datos.
- [`BaseDetector`], [`LogisticRegressionDetector`], [`MLPDetector`], [`OptunaMLPTuner`], [`LightGBMDetector`] y [`CNNLSTMDetector`] (`src/models.py`): Polimorfismo y herencia para entrenar y predecir probabilidades con una interfaz unificada.
- [`MetricsEvaluator`](`src/evaluation.py`): Calcula métricas de clasificación, construye la tabla comparativa y reporta la matriz de confusión.
- [`ExplainabilityAnalyzer`](`src/explainability.py`): Ejecuta *Permutation Importance* sobre LightGBM y *KernelSHAP* sobre CNN-LSTM.
- [`ConfidenceCalibrator`](`src/calibration.py`): Calcula ECE y Brier Score, ajustando calibradores isotónicos, sigmoideos (Platt) y escalado por temperatura ($T$).
- [`ReportGenerator`](`src/reporting.py`): Serializa resultados en JSON, exporta tablas CSV y genera el gráfico comparativo `f1_modelos.png`.
- [`DDoSDetectionPipeline`](`src/pipeline.py`): Coordina de extremo a extremo todas las etapas del protocolo experimental.

---

## 5. Instalación y Ejecución

### Requisitos

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Parámetros de Línea de Comandos (`main.py`)

| Argumento | Por defecto | Descripción |
| :--- | :---: | :--- |
| `--data` | `data` | Carpeta con archivos CSV o ruta a un archivo CSV individual. |
| `--label-col` | `auto` | Columna de etiqueta (`auto` detecta `Label`, `label`, `y`, `Class`). |
| `--out` | `resultados_poster` | Carpeta de salida para almacenar métricas, tablas y gráficos. |
| `--seed` | `42` | Semilla de reproducibilidad global. |
| `--sample` | `0` | Submuestreo preservando el orden temporal (`0` procesa todos los flujos). |
| `--epochs` | `30` | Número máximo de épocas para MLP y CNN-LSTM. |
| `--optuna-trials` | `0` | Número de trials de Optuna (`0` usa los hiperparámetros óptimos de la tesis). |
| `--gpu` | `False` | Activa `device='gpu'` en LightGBM. |
| `--skip-shap` | `False` | Omite el cálculo de KernelSHAP sobre CNN-LSTM. |
| `--shap-samples` | `100` | Cantidad de secuencias de prueba evaluadas en KernelSHAP. |

### Ejemplos de Uso

```bash
# Ejecución directa sobre todos los CSV de la carpeta data/
python main.py

# Ejecución rápida de validación (submuestreo ordenado de 300,000 flujos y 5 épocas)
python main.py --sample 300000 --epochs 5 --skip-shap

# Ejecución completa habilitando GPU para LightGBM
python main.py --data data --label-col Label --gpu
```

---

## 6. Artefactos de Salida

Todos los resultados se almacenan automáticamente en `resultados_poster/` (o el directorio indicado en `--out`):

- **`resultados.json`**: Resumen estructurado con estadísticas del dataset, métricas de cada modelo, matriz de confusión, top 3 de variables XAI y temperatura de calibración.
- **`tabla_modelos.csv`**: Tabla comparativa ordenada por F1-Score con `f1`, `precision`, `recall`, `pr_auc`, `roc_auc` y `accuracy`.
- **`importancias.csv`**: Ranking completo de los 19 atributos según *Permutation Importance* (LightGBM) y *KernelSHAP* (CNN-LSTM).
- **`calibracion.csv`**: Valores de ECE y Brier Score antes y después de la calibración probabilística.
- **`f1_modelos.png`**: Gráfico comparativo de F1-Score en alta resolución (200 DPI).

---

## 7. Referencias Principales

- Al-zubidi et al. (2024). *Journal of Intelligent Systems*, 33(1).
- Mousa & Abdullah (2023). *Future Internet*, 15(8).
- Musa & Odokuma (2023). *Scientia Africana*, 22(3).
- Aktar & Nur (2023). *Computers & Security*, 129.
- Wei et al. (2023). *arXiv:2305.09475*.
- Shieh et al. (2021). *Detection of unknown DDoS attacks with deep learning and GMM*.
- Hussain et al. (2020). *IEEE INMIC*.
- Pei et al. (2019). *J. Phys.: Conf. Ser.*, 1237(3).
