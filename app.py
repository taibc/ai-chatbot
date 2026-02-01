import os
from pypdf import PdfReader
import ollama

def load_pdfs(folder_path: str) -> str:
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
    return "\n".join(texts)


def ask_llm(context: str, question: str) -> str:
    prompt = f"""
Bạn là trợ lý AI. Chỉ trả lời dựa trên nội dung tài liệu.

TÀI LIỆU:
{context}

CÂU HỎI:
{question}
"""

    res = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return res["message"]["content"]
