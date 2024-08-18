import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough
from AgenticChunking import get_text_chunks_agentic_chunking
from TextSplitting import get_text_chunks_text_splitting
from HyDE import HyDE
from ChatMemory import memory, format_memory


# make an initializer
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# print("API Key set:", bool(os.getenv("GOOGLE_API_KEY")))

# api_key = os.getenv("GOOGLE_API_KEY")
# print(f"API key starts with: {api_key[:5]}..." if api_key else "API key not found")

# try:
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel('gemini-pro')
#     response = model.generate_content("Hello, World!")
#     print("API key is valid. Test response:", response.text)
# except Exception as e:
#     print(f"Error: {e}")



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text, chunking_mechanism='text splitting'):
    if(chunking_mechanism=='text splitting'):
        chunks = get_text_chunks_text_splitting(text)
    elif(chunking_mechanism=='agentic'):
        chunks = get_text_chunks_agentic_chunking(text)
    return chunks



def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")




def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. Refer to the chat history for better context
    provided to you.\n\n
    Context:\n {context}?\n
    Chat History:\n {chat_history}\n
    Question: \n{question}\n

    Answer:
    """


    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "chat_history", "question"])

    chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "chat_history": (memory.load_memory_variables | format_memory)}
    | prompt
    | model
)
    # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db =  FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    hypothetical_document = HyDE(user_question)
    docs = new_db.similarity_search(hypothetical_document, k=4)
    # docs = new_db.similarity_search(user_question, k=4)


    chain = get_conversational_chain()

    response = chain.invoke(
        input={
            "context": docs,
            "question": user_question
        }
    ).content
    memory.save_context({"input": user_question}, {"output": response})
    # response = chain(
    #     {"input_documents":docs, "question": user_question}
    #     , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response)




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text, chunking_mechanism='text splitting')
                # text_chunks = get_text_chunks(raw_text, chunking_mechanism='agentic')
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()