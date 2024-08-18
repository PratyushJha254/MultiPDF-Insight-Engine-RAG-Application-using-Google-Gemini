from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

@RunnableLambda
def format_memory(memory_output):
    chat_history = memory_output.get("chat_history", [])
    return "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])