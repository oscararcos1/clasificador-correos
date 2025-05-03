import ollama
import pandas as pd
import gradio as gr
from sklearn.metrics import classification_report
import io

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


# Función para clasificar un correo
def clasificar_correo(correo, estilo="Estilo original"):
    prompt = prompts_dict[estilo].format(correo=correo)
    
    respuesta = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return respuesta['message']['content'].strip()

def interfaz_llm(correo, estilo):
    return clasificar_correo(correo, estilo)

# Interacción desde consola
#if __name__ == "__main__":
 #   correo_input = input("Pega el correo aquí: ")
  #  categoria = clasificar_correo(correo_input)
   # print(f"Categoría: {categoria}")

def validar_modelo():
    df = pd.read_csv("test_correos.csv")
    y_true = []
    y_pred = []
    resultados = []  # ← Esta es la lista que estaba faltando

    for _, row in df.iterrows():
        pred = clasificar_correo(row['correo'])
        y_true.append(row['categoria'].lower())
        y_pred.append(pred.lower())

        resultados.append({
            "correo": row['correo'],
            "categoría real": row['categoria'],
            "predicción": pred,
            "correcto": pred.lower() == row['categoria'].lower()
        })

        print(f"Real: {row['categoria']} | Predicho: {pred}")

    print("\n📊 Reporte de clasificación:")
    print(classification_report(y_true, y_pred))

    # Guardar en archivo CSV
    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv("resultados_validacion.csv", index=False)
    print("✅ Resultados guardados en resultados_validacion.csv")

# Ejecuta esta función si quieres validar:
validar_modelo()

def interfaz_validacion(archivo_csv):
    df = pd.read_csv(archivo_csv.name)
    y_true = []
    y_pred = []
    resultados = []

    for _, row in df.iterrows():
        pred = clasificar_correo(row['correo'])
        y_true.append(row['categoria'].lower())
        y_pred.append(pred.lower())
        resultados.append({
            "correo": row['correo'],
            "categoría real": row['categoria'],
            "predicción": pred,
            "correcto": pred.lower() == row['categoria'].lower()
        })

    # Reporte de clasificación como texto plano
    reporte = classification_report(y_true, y_pred, digits=2)

    # Guardar archivo local
    pd.DataFrame(resultados).to_csv("resultados_validacion.csv", index=False)

    # Retornar tabla + reporte en string
    return pd.DataFrame(resultados), reporte

#def interfaz_llm(correo):
 #   categoria = clasificar_correo(correo)
  #  return f"Categoría: {categoria}"

# Lanza la app web local
demo = gr.TabbedInterface(
    interface_list=[
        gr.Interface(
            fn=interfaz_llm,
            inputs=[
                gr.Textbox(label="Pega el correo aquí"),
                gr.Dropdown(label="Estilo de Prompt", choices=list(prompts_dict.keys())),
            ],
            outputs=gr.Textbox(label="Categoría"),
            title="Clasificador de Correos con IA",
            description="Clasifica correos usando diferentes estrategias de prompt engineering.",
            theme="soft"
        )
        ,
        
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
        )
    ],
    tab_names=["Clasificador Manual", "Validación por CSV"]
)

demo.launch(share=True)

