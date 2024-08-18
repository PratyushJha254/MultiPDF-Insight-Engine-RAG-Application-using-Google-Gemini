from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from PyPDF2 import PdfReader
import re
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




def extract_paragraphs(text):
    paragraphs = []
    # Remove excess whitespace and split into lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    current_paragraph = []
    for line in lines:
        if line.endswith(('.', '!', '?')) or len(line) < 50:  # Assuming short lines are likely end of paragraphs
            current_paragraph.append(line)
            paragraph_text = ' '.join(current_paragraph)
            paragraphs.append(paragraph_text)
            current_paragraph = []
        else:
            current_paragraph.append(line)

    # Add any remaining lines as a paragraph
    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph)
        paragraphs.append(paragraph_text)

    # Post-processing to merge short paragraphs and split very long ones
    merged_paragraphs = []
    for para in paragraphs:
        if len(merged_paragraphs) > 0 and len(merged_paragraphs[-1]) + len(para) < 1000:
            merged_paragraphs[-1] += ' ' + para
        elif len(para) > 1500:  # Split very long paragraphs
            split_paras = re.split(r'(?<=[.!?])\s+', para)
            merged_paragraphs.extend(split_paras)
        else:
            merged_paragraphs.append(para)

    return merged_paragraphs




def get_propositions(text, runnable):
    runnable_output = runnable.invoke({
        "input": text
    }).content
    propositions=[outputs.strip("\"") for outputs in runnable_output.strip("```json\n[\n").strip("\n```\n]").split(',\n')]
    return propositions



def get_text_chunks_agentic_chunking(text, type='proportion based'):
    # 5. Agentic chunking
    if(type=='proportion based'):
        # uses LLMs
        prompt_template = hub.pull('wfh/proposal-indexing', api_url="https://api.hub.langchain.com")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", api_key=os.getenv("GOOGLE_API_KEY"))
        runnable = prompt_template | llm
        paragraphs = extract_paragraphs(text)
        text_propositions = []
        for i, para in enumerate(paragraphs):

            propositions = get_propositions(para, runnable)
            text_propositions.extend(propositions)
            # if i%5==0:
            #     time.sleep(60)

    return text_propositions



def get_pdf_text(pdf_path):
    text=""
    pdf_reader= PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text+= page.extract_text()
    return  text



def main():
    pdf_path=r"C:\Users\praty\Desktop\RAG Implementation\attention.pdf"
    text = get_pdf_text(pdf_path=pdf_path)
    propositions = get_text_chunks_agentic_chunking(text)
    print("propositions reached successfully")


if __name__ == "__main__":
    main()