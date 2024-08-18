from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
import os
from ChatMemory import memory, format_memory



def HyDE(user_question):
    template = """Please write a scientific paper passage to answer the question. Refer to the history for better context.
        Question: {question}
        History: {chat_history}
        Passage:"""
    prompt_hyde = PromptTemplate(template = template, input_variables = ["question", "chat_history"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", api_key=os.getenv("GOOGLE_API_KEY"))
    generate_docs_for_retrieval = (
        {"question": RunnablePassthrough(), "chat_history": memory.load_memory_variables | format_memory}
        | prompt_hyde
        | model
    )
    hypothetical_document = generate_docs_for_retrieval.invoke({"question": user_question}).content
    return hypothetical_document

def main():
    user_question = "What is an LLM?"
    response = HyDE(user_question)
    print(response)


if __name__=="__main__":
    main()