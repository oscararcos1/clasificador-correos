import ollama
import pandas as pd
import gradio as gr
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Diccionario de Prompts
prompts_dict = {
    "Estilo original": """Clasifica el siguiente correo electrónico en una de las siguientes categorías: Trabajo, Personal, Publicidad, Spam.
Devuelve solo el nombre de la categoría.

Correo: {correo}""",

    "Ejemplos y formato explícito": """Eres un modelo experto en clasificación de correos electrónicos.

Ejemplo 1:
Correo: "¡Gran venta de fin de semana en laptops y accesorios!"
Categoría: Publicidad

Ejemplo 2:
Correo: "¿Vamos a almorzar mañana en el centro?"
Categoría: Personal

Correo: "{correo}"
Devuelve solo una de las siguientes categorías: Trabajo, Personal, Publicidad, Spam.
Formato de salida: Categoría: [nombre]""",

    "Rol experto + pasos": """Actúa como un asistente experto en análisis de correos electrónicos. Sigue los pasos a continuación:

1. Lee el contenido del correo.
2. Analiza la intención del mensaje.
3. Determina a cuál de las siguientes categorías pertenece: Trabajo, Personal, Publicidad, Spam.
4. Responde únicamente con el nombre de la categoría correspondiente.

Correo: {correo}""",

    "Definiciones + sin explicaciones": """Clasifica el siguiente correo electrónico en una de estas categorías: Trabajo, Personal, Publicidad o Spam.

Ten en cuenta:
- Trabajo: comunicaciones laborales o profesionales.
- Personal: mensajes informales entre amigos o familia.
- Publicidad: promociones, ofertas, descuentos.
- Spam: mensajes irrelevantes o maliciosos.

No expliques tu razonamiento ni justifiques tu respuesta. Devuelve únicamente el nombre de la categoría.

Correo: {correo}"""
}

# Categorías válidas y función de limpieza
CATEGORIAS_VALIDAS = ["trabajo", "personal", "publicidad", "spam"]

def limpiar_categoria(respuesta):
    respuesta = respuesta.strip().lower()
    for cat in CATEGORIAS_VALIDAS:
        if cat in respuesta:
            return cat
    return "otro"

# Clasificación individual
def clasificar_correo(correo, estilo="Estilo original"):
    prompt = prompts_dict[estilo].format(correo=correo)
    respuesta = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return respuesta['message']['content'].strip()

# Validación con test_correos.csv
def validar_modelo():
    df = pd.read_csv("test_correos.csv")
    y_true, y_pred, resultados = [], [], []

    for _, row in df.iterrows():
        pred = clasificar_correo(row['correo'])
        pred_limpia = limpiar_categoria(pred)
        y_true.append(row['categoria'].lower())
        y_pred.append(pred_limpia)

        resultados.append({
            "correo": row['correo'],
            "categoría real": row['categoria'],
            "predicción": pred_limpia,
            "correcto": pred_limpia == row['categoria'].lower()
        })

        print(f"Real: {row['categoria']} | Predicho: {pred_limpia}")

    print("\n📊 Reporte de clasificación:")
    print(classification_report(y_true, y_pred))

    pd.DataFrame(resultados).to_csv("resultados_validacion.csv", index=False)
    print("✅ Resultados guardados en resultados_validacion.csv")

# Validación desde archivo CSV subido
def interfaz_validacion(archivo_csv):
    df = pd.read_csv(archivo_csv.name)
    y_true, y_pred, resultados = [], [], []

    for _, row in df.iterrows():
        pred = clasificar_correo(row['correo'])
        pred_limpia = limpiar_categoria(pred)
        y_true.append(row['categoria'].lower())
        y_pred.append(pred_limpia)
        resultados.append({
            "correo": row['correo'],
            "categoría real": row['categoria'],
            "predicción": pred_limpia,
            "correcto": pred_limpia == row['categoria'].lower()
        })

    reporte = classification_report(y_true, y_pred, digits=2)
    pd.DataFrame(resultados).to_csv("resultados_validacion.csv", index=False)
    return pd.DataFrame(resultados), reporte

# Comparar todos los prompts
def comparar_prompts():
    resultados_totales = []
    df = pd.read_csv("test_correos.csv")

    for estilo in prompts_dict:
        print(f"\n🔍 Probando estilo de prompt: {estilo}")
        for _, row in df.iterrows():
            pred = clasificar_correo(row['correo'], estilo=estilo)
            pred_limpia = limpiar_categoria(pred)
            resultados_totales.append({
                "correo": row['correo'],
                "categoria_real": row['categoria'],
                "prediccion": pred_limpia,
                "correcto": pred_limpia == row['categoria'].lower(),
                "prompt": estilo
            })

    pd.DataFrame(resultados_totales).to_csv("comparacion_prompts.csv", index=False)
    print("✅ Resultados guardados en comparacion_prompts.csv")

# Gráfico comparativo
def graficar_metricas_prompts():
    df = pd.read_csv("comparacion_prompts.csv")
    resumen = df.groupby("prompt").agg({"correcto": ["mean", "count"]}).reset_index()
    resumen.columns = ["prompt", "accuracy", "n_ejemplos"]
    resumen["accuracy"] *= 100

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=resumen, x="prompt", y="accuracy", palette="Set2", ax=ax)
    ax.set_title("Precisión por estilo de prompt")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Prompt")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Interfaz Gradio
def interfaz_llm(correo, estilo):
    return clasificar_correo(correo, estilo)

demo = gr.TabbedInterface(
    interface_list=[
        gr.Interface(
            fn=interfaz_llm,
            inputs=[
                gr.Textbox(label="Pega el correo aquí"),
                gr.Dropdown(label="Estilo de Prompt", choices=list(prompts_dict.keys()))
            ],
            outputs=gr.Textbox(label="Categoría"),
            title="Clasificador de Correos con IA",
            description="Clasifica correos usando diferentes estrategias de prompt engineering.",
            theme="soft"
        ),
        gr.Interface(
            fn=interfaz_validacion,
            inputs=gr.File(label="Sube un archivo CSV con columnas 'correo' y 'categoria'"),
            outputs=[
                gr.Dataframe(label="Predicciones"),
                gr.Textbox(label="Reporte de Clasificación")
            ],
            title="Validador por Archivo CSV",
            description="Sube tu archivo test_correos.csv para evaluar el modelo y ver su desempeño.",
            theme="soft"
        ),
        gr.Interface(
            fn=graficar_metricas_prompts,
            inputs=[],
            outputs=gr.Plot(label="Gráfico de Accuracy por Prompt"),
            title="Gráfico Comparativo",
            description="Visualiza el rendimiento de cada prompt en precisión.",
            theme="soft"
        )
    ],
    tab_names=["Clasificador Manual", "Validación por CSV", "Gráfico Comparativo"]
)

# Ejecución
if __name__ == "__main__":
    comparar_prompts()
    graficar_metricas_prompts()
    demo.launch(share=True)
