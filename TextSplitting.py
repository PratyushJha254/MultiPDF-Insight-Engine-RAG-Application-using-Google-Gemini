from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_text_chunks_text_splitting(text):
    # 1. Character text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)

    return chunks