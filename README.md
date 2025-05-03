# Clasificador de Correos con LLM

Este proyecto utiliza un modelo de lenguaje local (Mistral) con `Ollama` para clasificar correos electrónicos en una de las siguientes categorías:

- **Trabajo**
- **Personal**
- **Publicidad**
- **Spam**

La interfaz está construida con **Gradio** y permite probar distintos estilos de *prompt engineering*, validar resultados con un archivo CSV y visualizar métricas comparativas.

---

##Archivos del proyecto

| Archivo                        | Descripción |
|-------------------------------|-------------|
| `main.py`                     | Script principal del proyecto |
| `test_correos.csv`            | Correos de prueba con su categoría real |
| `resultados_validacion.csv`   | Predicciones generadas en la validación |
| `comparacion_prompts.csv`     | Resultados de comparación entre distintos prompts |
| `requirements.txt`            | Librerías necesarias para ejecutar el proyecto |
| `.gitignore`                  | Excluye archivos y carpetas innecesarias del repositorio |
| `README.md`                   | Este archivo de documentación |

---

##Cómo ejecutar

### 1. Clonar el repositorio

```bash
git clone https://github.com/oscararcos1/clasificador-correos.git
cd clasificador-correos



