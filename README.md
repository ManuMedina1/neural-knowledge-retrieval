# ⚔️ NLP Avanzado: Steam Reviews & BASH Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-DeBERTa%20%7C%20RoBERTa%20%7C%20Phi--3-yellow)
![Steam](https://img.shields.io/badge/Data-Steam%20Reviews-black?logo=steam)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Implementación de técnicas SOTA en NLP: Fine-Tuning eficiente (LoRA) para análisis de sentimiento en videojuegos y arquitectura RAG modular para asistencia técnica en Linux.**

---

## 📖 Descripción del Proyecto

Este repositorio explora dos paradigmas fundamentales del Procesamiento del Lenguaje Natural moderno aplicados a casos de uso reales:

1.  **Clasificación de Opiniones (LoRA):** ¿Puede un modelo entender la frustración de un jugador de *Geometry Dash*? Entrenamos modelos para distinguir reseñas positivas de negativas.
2.  **Asistente Técnico (RAG):** Un sistema experto capaz de responder dudas sobre comandos de **BASH** consultando un manual oficial, reduciendo las alucinaciones de los LLMs.

---

## 🚀 Módulo 1: LoRA Fine-Tuning (Geometry Dash Reviews)

Extrajimos reseñas reales de Steam del juego **Geometry Dash (ID: 322170)** para entrenar un clasificador de sentimientos robusto.

* **El Reto:** El lenguaje "gamer" es complejo, lleno de jerga, ironía y memes (ej. *"Phobos is hard but I am noob"*).
* **La Solución:** Aplicar **LoRA (Low-Rank Adaptation)** a modelos base potentes para especializarlos sin reentrenar todos sus parámetros.

### 🛠️ Modelos y Configuración
| Modelo Base | Configuración LoRA | Params Entrenables | Resultados (F1-Score) |
| :--- | :--- | :--- | :--- |
| **Microsoft DeBERTa v3** | `r=4`, `alpha=32`, `dropout=0.1` | **~0.15%** | ⭐ **Muy Alto** |
| **Facebook RoBERTa** | `r=4`, `alpha=32`, `dropout=0.1` | **~0.65%** | ⭐ **Alto** |

> **Nota:** Se comparó el rendimiento entrenando solo las matrices de proyección (`query`, `value`, `key`), logrando resultados de nivel profesional con una fracción del coste computacional.

---

## 📚 Módulo 2: Sistema RAG (BASH Manual)

Implementación "from-scratch" de un sistema de **Retrieval-Augmented Generation** para consultar el manual de referencia de GNU Bash (`manual_bash.txt`).

A diferencia de las librerías estándar, aquí implementamos la **lógica de recuperación manualmente** para entender las matemáticas detrás del RAG.

### ⚙️ Arquitectura del Pipeline
1.  **Ingesta:** Lectura y limpieza del `manual_bash.txt`.
2.  **Chunking:** División inteligente del texto con solapamiento (`GeneradorChunks`).
3.  **Embeddings:** Vectorización usando `jina-embeddings-v2-base-es` o `paraphrase-multilingual-mpnet`.
4.  **Retrieval:** Búsqueda por **Similitud del Coseno** (implementación propia con NumPy/Scikit-Learn).
5.  **Generación:** Comparativa entre **Microsoft Phi-3.5** y **Qwen 2.5-1.5B**.

### 🆚 Comparativa: Qwen vs Phi-3.5

**Pregunta:** *"¿Qué hace el comando grep en Bash?"*

| Modelo | Respuesta Generada | Calidad |
| :--- | :--- | :--- |
| **Qwen 2.5** | Explica detalladamente el uso de expresiones regulares y opciones como `-n` o `-i`. | 🟢 **Muy detallada** |
| **Phi-3.5** | Resumen conciso sobre buscar patrones y aclara qué *no* hace (ej. no comprime archivos). | 🟡 **Directa y concisa** |

*(Puedes ver los logs completos en la carpeta `results/`)*

---

## 📂 Estructura del Repositorio

```bash
├── data/

├── notebooks/
│   ├── LoRA.ipynb            # Extracción Steam + Fine-Tuning (DeBERTa/RoBERTa)
│   ├── RAG.ipynb             # Pipeline RAG artesanal con Phi-3.5/Qwen
│   └── manual_bash.txt       # Corpus de conocimiento para el RAG
├── results/
│   ├── respuestas_normal.txt   # Logs de ejecución con Qwen
│   └── respuestas_especial.txt # Logs de ejecución con Phi-3.5
├── requirements.txt          # Dependencias (transformers, peft, steam-reviews)
└── README.md
