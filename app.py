import os
from pypdf import PdfReader
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_pdfs(folder_path: str) -> str:
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                if page.extract_text():
                    texts.append(page.extract_text())
    return "\n".join(texts)


def ask_llm(context: str, question: str) -> str:
    prompt = f"""
Chỉ trả lời dựa trên nội dung tài liệu sau:

{context}

Câu hỏi: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
