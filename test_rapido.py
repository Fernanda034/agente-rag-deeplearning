import sys
sys.path.insert(0, '.')
sys.path.insert(0, './03_agente')

from agent import get_agent

print("Inicializando agente...")
agent = get_agent()

result = agent.invoke({
    "input": "¿Qué es batch normalization?",
    "chat_history": []
})
print("\n=== RESPUESTA ===")
print(result["output"])