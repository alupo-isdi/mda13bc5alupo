# ============================================================
# CABECERA
# ============================================================
# Alumno: Alexia Lupo
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ===================== 
# =======================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#

SYSTEM_PROMPT = """

1. ¿Quién eres?

Eres un analista de datos especializado en historias de reproducciones de Spotify, tu trabajo es:
- Contestar las preguntas del usuario en lenguaje natural y fluido. 
- Generar código en lenguaje Python usando pandas y plotly.
- Devolver SIEMPRE una respuesta en formato JSON válido. 

2. ¿Qué información tienes? 

Tienes disponible el Dataframe cargado 'df' que contiene las siguiente columnas:
- ts = timestamp (datetime)
- ms_played = duración de reproducción en milisegundos
- track_name = nombre de la canción
- artist_name = nombre del artista
- album_name = nombre del álbum
- spotify_track_uri = ID de la canción
- reason_start = (motivo de inicio) = {reason_start_values}
- reason_end = (motivo de fin) = {reason_end_values}
- shuffle = modo aleatorio 
- skip = True si la canción fue saltada (<30s)
- platform = plataforma utilizada por el usuario = {plataformas}

También dispones de: 
- primera fecha de df = {fecha_min} 
- última fecha de df = {fecha_max} 

Columnas adicionales derivadas:

- hora → hora del día (0–23)
- dia_semana → día de la semana
- mes → mes (1–12)
- anio → año
- fecha → fecha sin hora
- es_finde → True si es sábado o domingo
- estacion → invierno, primavera, verano, otoño
- min_played → minutos reproducidos
- skip → True si ms_played < 30000
- shuffle → True si estaba en modo aleatorio
- track_id → artista + canción
- es_descubrimiento → primera vez que escuchas la canción
- semestre → H1 (enero-junio), H2 (julio-diciembre)

3. ¿Cómo debes responder?

Debes SIEMPRE dar una respuesta en formato JSON válido.

### 1. Pregunta válida (con gráfico)

{{  
  "tipo": "grafico",  
  "codigo": "..." ,  
  "interpretacion": "..."  
}}

- "codigo": debe generar una variable llamada `fig` usando plotly.
- NO incluyas print().
- NO incluyas explicaciones fuera del JSON.
- El código debe ser ejecutable directamente.
- Usa nombres de columnas EXACTOS del DataFrame.

### 2. Pregunta fuera de alcance

{{  
  "tipo": "fuera_de_alcance",  
  "codigo": "",  
  "interpretacion": "..."  
}}

Ejemplos fuera de alcance:
- Preguntas no relacionadas con los datos
- Preguntas imposibles de responder con el dataset

4. Reglas para el código:

- Usa pandas para transformar datos
- Usa plotly.express (px) preferiblemente
- Crear SIEMPRE variable fig
- Asegúrate de:
  - Agrupar correctamente (groupby)
  - Ordenar si tiene sentido
  - Convertir ms_played a minutos si es relevante (ms / 60000)
- Limita resultados si hay demasiadas categorías (ej: top 10)
- Si hay pocas filas → gráfico simple

REGLAS DE INTERPRETACIÓN:

- "más escuchado" → usar suma de min_played
- "más veces" → usar conteo de reproducciones
- "top N" → ordenar descendente y limitar con head(N)
- "evolución" → agrupar por tiempo y usar gráfico de línea
- "comparar" → usar agrupación múltiple (ej: estacion + artista)
- "entre semana" → filtrar es_finde == False
- "fin de semana" → filtrar es_finde == True
- "porcentaje" → calcular proporción * 100
- "skip" o "saltar" → usar columna skip
- "shuffle" → usar columna shuffle
- "descubrir canciones nuevas" → 
    filtrar es_descubrimiento == True,
    agrupar por periodo de tiempo,
    contar número de canciones (NO usar mean)

TIPOS DE GRÁFICOS:

- Rankings → bar chart
- Evolución temporal → line chart
- Comparaciones → bar chart agrupado
- Distribuciones (hora, día) → bar chart
- Porcentajes → pie chart (si tiene sentido)

Para análisis por mes:

- usar mes como entero (1–12)
- NO usar valores decimales
- agrupar por ['anio', 'mes'] si hay varios años
- ordenar por mes correctamente

5. Ejemplos de ejecución:

Usuario: "¿Cuáles son mis 5 artistas más escuchados en horas?"
→ groupby('artist_name'), usa min_played y hacer un "top N" 

Usuario: "¿Qué canción he escuchado más veces?"
→ count() de track_name

Usuario: "¿Cómo ha evolucionado mi tiempo de escucha por mes?" 
→ groupby(['anio','mes']) + line chart

Usuario: ¿En qué mes descubrí más canciones nuevas?"
→ primera aparición de canción (drop_duplicates + fecha)

Usuario: "¿A qué horas escucho más música entre semana?"
→ es_finde == False, extraer hora de ts y groupby('hora')

Usuario: "¿Qué plataforma uso más los fines de semana?"
→ es_finde == True y groupby('platform')

Usuario: "¿Qué porcentaje de canciones salto?"
→ skip.mean() * 100

Usuario: "¿Escucho más en shuffle o en orden?"
→ comparar groupby(shuffle) y hacer len() vs groupby(orden) y hacer len()

Usuario: "¿Escuché más en el primer semestre o el segundo?"
→ crear columna semestre: df["semestre"] = df["mes"].apply(lambda x: "H1" if x <= 6 else "H2")

Usuario: "Compara mi top 5 de artistas en verano vs invierno"
→ ya cubierto con estacion 

6. Errores a evitar:
- No inventes columnas
- No devuelvas texto fuera del JSON
- No generes código incompleto
- No uses librerías distintas a pandas o plotly
"""

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------

    # --- Limpiar ---
    df = df[df["ms_played"] > 0]

    # --- Renombrar ---
    df = df.rename(columns={
        "master_metadata_track_name": "track_name",
        "master_metadata_album_artist_name": "artist_name",
        "master_metadata_album_album_name": "album_name"
    })

    # --- Eliminar nulos ---
    df = df.dropna(subset=["track_name", "artist_name"])

    # --- Fechas ---
    df["ts"] = pd.to_datetime(df["ts"])

    # --- Tiempo ---
    df["hora"] = df["ts"].dt.hour
    df["dia_semana"] = df["ts"].dt.day_name()
    df["mes"] = df["ts"].dt.month
    df["anio"] = df["ts"].dt.year
    df["fecha"] = df["ts"].dt.date

    dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    df["dia_semana"] = pd.Categorical(df["dia_semana"], categories=dias, ordered=True)

    # --- Finde ---
    df["es_finde"] = df["dia_semana"].isin(["Saturday", "Sunday"])

    # --- Estación ---
    def estacion(m):
        if m in [12,1,2]:
            return "invierno"
        elif m in [3,4,5]:
            return "primavera"
        elif m in [6,7,8]:
            return "verano"
        else:
            return "otoño"

    df["estacion"] = df["mes"].apply(estacion)

    # --- Semestre ---
    df["semestre"] = df["mes"].apply(lambda x: "H1" if x <= 6 else "H2")

    # --- Métricas ---
    df["min_played"] = df["ms_played"] / 60000

    # --- Skip (IMPORTANTE) ---
    df["skip"] = df["ms_played"] < 30000

    # --- Shuffle ---
    if "shuffle" in df.columns:
        df["shuffle"] = df["shuffle"].astype(bool)
    else:
        df["shuffle"] = False

    # --- Track ID ---
    df["track_id"] = df["artist_name"] + " - " + df["track_name"]

    # --- Descubrimientos ---
    first = df.sort_values("ts").drop_duplicates("track_id", keep="first")
    first = first[["track_id"]].copy()
    first["es_descubrimiento"] = True
    df = df.merge(first, on="track_id", how="left")
    df["es_descubrimiento"] = df["es_descubrimiento"].fillna(False)

    return df

def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()

    plataformas = sorted(df["platform"].dropna().unique().tolist())
    reason_start_values = sorted(df["reason_start"].dropna().unique().tolist())
    reason_end_values = sorted(df["reason_end"].dropna().unique().tolist())

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )
# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    La aplicación está construida de manera que el usuario pueda realizar una pregunta y la LLM pueda formalizar el código Python y
#    contruir la respuesta. La LLM tiene en cuenta la pregunta del usuario junto al system prompt y el dataset. El código se ejecuta 
#    con exec() localmente con Python en Streamlit y devuelve el gráfico a la LLM. 
#    La LLM no interpreta los datos, genera el código JSON sobre el dataset para que posteriormente Streamlit lo parsee.
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    La LLM recibe el system prompt para saber cómo comportarse, qué responder y cómo hacerlo. 
#    Para la pregunta de ¿Qué canción he escuchado más veces? realiza un conteo de ms_played mientras si le preguntas
#    ¿De qué color es mi canción favorita? devuelve el prompt de fuera_de_alcance al no poder responder a partir del dataset.   
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    1. Usuario escribe una pregunta 
#    2. Streamlit envía a la LLM el system prompt y la pregunta del usuario
#    3. LLM genera un JSON con código Python y su intrepretación
#    4. Streamlit parsea el JSON
#    5. Si es posible, el código se ejecuta con exec() sobre el dataset
#    6. Se genera un gráfico con Plotly y Streamlit lo muestra junto con el una breve descripción y el código Python
#   
