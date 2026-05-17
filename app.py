import streamlit as st
import sys
import os

# Añadir el directorio actual al path para importar el agente
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Python no permite importar módulos cuyos nombres empiecen por números usando 'import' tradicional.
# Usamos importlib para sortear esta limitación sin tener que renombrar tu carpeta '03_agente'.
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("agent", "./03_agente/agent.py")
agent_module = importlib.util.module_from_spec(spec)
sys.modules["agent"] = agent_module
spec.loader.exec_module(agent_module)

get_agent = agent_module.get_agent

load_dotenv()

st.set_page_config(page_title="Asistente de Deep Learning", page_icon="🧠", layout="wide")

st.title("🧠 Asistente Virtual: Deep Learning")
st.markdown("Pregúntame cualquier duda sobre el temario de Deep Learning o Machine Learning.")

# Verificar que la API key exista
if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "tu_clave_aqui":
    st.error("⚠️ No se encontró una API Key de Groq válida. Por favor, edita el archivo .env con tu clave.")
    st.stop()

# Inicializar el agente en session state para no recargarlo en cada rerun
if "agent_executor" not in st.session_state:
    try:
        with st.spinner("Inicializando agente y cargando base de conocimiento..."):
            st.session_state.agent_executor = get_agent()
    except FileNotFoundError:
        st.error("❌ No se encontró la base de datos vectorial en `vector_db/`. Por favor corre el script de ingesta (`01_ingesta/ingestion.py`) primero.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error inicializando el agente: {e}")
        st.stop()

# Inicializar historial de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostrar el historial de chat en la UI
for message in st.session_state.chat_history:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])

# Input del usuario
prompt = st.chat_input("Escribe tu pregunta de Deep Learning aquí...")

if prompt:
    # 1. Mostrar el mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Añadir al historial
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # 2. Generar y mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Buscando en libros y papers..."):
            # Formatear historial para LangChain
            formatted_history = []
            for msg in st.session_state.chat_history[:-1]: # Excluir el prompt actual
                if msg["role"] == "user":
                    formatted_history.append(("human", msg["content"]))
                else:
                    formatted_history.append(("assistant", msg["content"]))
            
            try:
                # Ejecutar el agente
                response = st.session_state.agent_executor.invoke({
                    "input": prompt,
                    "chat_history": formatted_history
                })
                
                answer = response["output"]
                st.markdown(answer)
                
                # Añadir al historial
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Ocurrió un error al generar la respuesta: {e}")
