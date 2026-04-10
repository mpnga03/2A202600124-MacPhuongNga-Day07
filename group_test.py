from __future__ import annotations

import os
import sys
from pathlib import Path
import chromadb
from dotenv import load_dotenv

# Giả định bạn đã lưu class FixedSizeChunker trong file src/chunking.py
# Nếu chưa, bạn có thể copy trực tiếp class FixedSizeChunker thả vào file này.
from src.chunking import FixedSizeChunker
from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document

SAMPLE_FILES = [
    "data/luat_lao_dong.md",
]

TEST_QUERIES = [
    "Bộ luật Lao động năm 2019 (Luật số 45/2019/QH14) chính thức có hiệu lực thi hành kể từ ngày tháng năm nào?",
    "Theo Bộ luật Lao động 2019, hợp đồng lao động được phân loại thành mấy loại chính? Đó là những loại nào?",
    "Quy định pháp luật không cho phép áp dụng thời gian thử việc đối với trường hợp người lao động giao kết loại hợp đồng lao động nào?",
    "Theo quy định, thời gian thử việc tối đa đối với công việc của người quản lý doanh nghiệp (theo quy định của Luật Doanh nghiệp, Luật Quản lý, sử dụng vốn nhà nước đầu tư vào sản xuất, kinh doanh tại doanh nghiệp) là bao nhiêu ngày?",
    "Trong dịp lễ Quốc khánh 02/9, người lao động được nghỉ làm việc và hưởng nguyên lương tổng cộng bao nhiêu ngày?",
    "Lộ trình điều chỉnh tuổi nghỉ hưu đối với người lao động làm việc trong điều kiện lao động bình thường được thực hiện cho đến khi đạt mức độ tuổi nào đối với nam và nữ?"
]

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """Gọi LLM thật từ OpenAI để trả lời dựa trên context."""
    from openai import OpenAI
    import os
    
    # Khởi tạo client (tự động lấy OPENAI_API_KEY từ biến môi trường)
    try:
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Hoặc "gpt-3.5-turbo" tùy bạn chọn
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý pháp lý AI. Hãy trả lời câu hỏi chính xác dựa trên tài liệu được cung cấp."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Nhiệt độ thấp (0 - 0.2) giúp RAG trả lời chính xác, không bịaa đặt
            #max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"[Lỗi khi gọi OpenAI API]: {e}"


class ChromaEmbedderAdapter:
    """Lớp cầu nối giúp các Embedder (Local/OpenAI) tương thích 100% với chuẩn của ChromaDB"""
    def __init__(self, base_embedder):
        self.base_embedder = base_embedder
        
    def __call__(self, input: list[str]) -> list[list[float]]:
        # Dùng cho lúc nạp dữ liệu (Upsert) - Xử lý danh sách text
        embeddings = []
        for text in input:
            vector = self.base_embedder(text) 
            embeddings.append(vector)
        return embeddings
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Dự phòng nếu hệ thống gọi đích danh hàm embed_documents
        return self.__call__(texts)

    # Đổi tên tham số từ 'query' thành 'input' để khớp với cách Chroma gọi
    def embed_query(self, input) -> list[float] | list[list[float]]:
        # Dùng cho lúc tìm kiếm (Query)
        if isinstance(input, str):
            return self.base_embedder(input)
        else:
            return self.__call__(input)
            
    def name(self) -> str:
        # Vượt qua bài kiểm tra bảo mật (conflict check) của ChromaDB
        return getattr(self.base_embedder, '_backend_name', 'custom_embedder')

    

def run_manual_demo(questions: list[str] | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    queries = questions if questions else TEST_QUERIES

    print("=== Khởi tạo hệ thống RAG với ChromaDB ===")
    
    # 1. KHỞI TẠO EMBEDDER CỦA BẠN (Đọc từ file .env)
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"[*] Đang sử dụng Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    
    # Khởi tạo Adapter bọc embedder lại cho ChromaDB
    chroma_embedding_fn = ChromaEmbedderAdapter(embedder)

    # 2. KHỞI TẠO CHROMA DB PERSISTENT CLIENT
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    
    # Tạo hoặc lấy collection, truyền Adapter vào
    collection = chroma_client.get_or_create_collection(
        name="luat_lao_dong",
        embedding_function=chroma_embedding_fn
    )

    # 3. KIỂM TRA VÀ NẠP DỮ LIỆU
    if collection.count() == 0:
        print("[-] Cơ sở dữ liệu trống. Đang đọc file, băm nhỏ và nhúng (embed) vào ChromaDB...")
        
        docs = load_documents_from_files(files)
        if not docs:
            print("\nKhông tìm thấy file tài liệu nào hợp lệ.")
            return 1

        # Chunking
        chunker = FixedSizeChunker(chunk_size=800, overlap=100)
        
        ids = []
        documents = []
        metadatas = []
        
        for doc in docs:
            text_chunks = chunker.chunk(doc.content)
            for i, text in enumerate(text_chunks):
                ids.append(f"{doc.id}_chunk_{i}")
                documents.append(text)
                metadatas.append(doc.metadata)
                
        # Lưu vào ChromaDB
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[+] Đã lưu vĩnh viễn {collection.count()} chunks vào ổ cứng (thư mục ./chroma_storage).")
    else:
        print(f"[+] Cơ sở dữ liệu đã có sẵn {collection.count()} chunks. Bỏ qua bước load file và embedding!")

    # 4. CHẠY BỘ CÂU HỎI TEST
    print("\n=== BẮT ĐẦU CHẠY BỘ CÂU HỎI TEST ===")
    for i, query in enumerate(queries, start=1):
        print(f"\n[{i}/{len(queries)}] Câu hỏi: {query}")
        
        # TÌM KIẾM BẰNG CHROMA DB (Retriever)
        results = collection.query(
            query_texts=[query],
            n_results=3 # Lấy top 3 chunks liên quan nhất
        )
        
        if results['distances'] and len(results['distances'][0]) > 0:
            # ChromaDB trả về list các list. Ta lấy phần tử đầu tiên của top 1.
            top_distance = results['distances'][0][0] 
            print(f"  [Retriever] Top 1 Distance (Càng nhỏ càng tốt): {top_distance:.3f}")
        else:
            print("  [Retriever] Không tìm thấy đoạn văn bản nào phù hợp.")
            continue
            
        # GỌI LLM (Generator)
        context_string = "\n\n".join(results['documents'][0])
        print("  [Context for LLM]:")
        print(context_string[20:])
        
        prompt = f"""Bạn là một trợ lý pháp lý AI chuyên nghiệp. Hãy đọc tài liệu được cung cấp và trả lời câu hỏi của người dùng.

        <instructions>
        - Trả lời NGẮN GỌN, súc tích và ĐÚNG TRỌNG TÂM. Không giải thích dài dòng.
        - Trình bày kết quả bằng định dạng Markdown (sử dụng in đậm, danh sách gạch đầu dòng để dễ đọc).
        - TUYỆT ĐỐI CHỈ dựa vào thông tin trong phần <context> bên dưới. 
        - Nếu thông tin trong <context> không đủ để trả lời, hãy nói: "Tôi không tìm thấy thông tin này trong tài liệu."
        </instructions>

        <context>
        {context_string}
        </context>

        <question>
        {query}
        </question>

        **Câu trả lời:**
        """
        print("  [Agent Answer]:")
        try:
            answer = demo_llm(prompt) # Hàm LLM của bạn
            print(f"  => {answer}")
        except Exception as e:
            print(f"  => [Lỗi khi trả lời]: {e}")
            
    print("\n=== HOÀN THÀNH BÀI TEST ===")
    return 0


def main() -> int:
    user_input = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    
    if user_input:
        # Nếu gõ câu hỏi tay -> Chỉ chạy câu đó
        queries = [user_input]
    else:
        # Nếu không gõ gì -> Chạy toàn bộ danh sách
        queries = TEST_QUERIES 
    
    return run_manual_demo(questions=queries)


if __name__ == "__main__":
    raise SystemExit(main())