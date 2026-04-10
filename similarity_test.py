import os
import numpy as np
from dotenv import load_dotenv

# Import trực tiếp các cấu hình và class Embedder từ project của bạn
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)

# ==========================================
# ĐẦU VÀO: DANH SÁCH CÁC CẶP CÂU CẦN SO SÁNH
# ==========================================
SENTENCE_PAIRS = [
    # Cặp 1: Thay đổi từ vựng (Từ đồng nghĩa) nhưng ngữ cảnh y hệt nhau.
    # Kỳ vọng: Điểm rất cao (> 0.7). Test khả năng nhận diện "mèo đen" = "mèo mun", "ngủ say" = "nằm thiu thiu".
    (
        "Con mèo đen đang ngủ say trên chiếc ghế sofa ngoài phòng khách.",
        "Chú mèo mun đang nằm thiu thiu trên ghế nệm ở ngoài sảnh."
    ),

    # Cặp 2: Đảo ngược cấu trúc (Chủ động - Bị động).
    # Kỳ vọng: Điểm rất cao (> 0.7). Test xem mô hình có hiểu việc "A đánh bại B" cũng chính là "B thất bại trước A" hay không.
    (
        "Đội tuyển Việt Nam đã xuất sắc đánh bại Thái Lan trong trận chung kết.",
        "Thái Lan đã phải nhận thất bại trước đội tuyển Việt Nam ở trận đấu cuối cùng."
    ),

    # Cặp 3: Khác ý nghĩa một chút, nhưng chung chủ đề (Thời tiết).
    # Kỳ vọng: Điểm trung bình (0.4 - 0.6). Hai câu nói về thời tiết, mưa gió nhưng không hoàn toàn tương đương về mặt thông tin.
    (
        "Trời hôm nay mưa rất to và có sấm chớp dữ dội.",
        "Mùa mưa năm nay đến sớm hơn thường lệ làm nhiều tuyến phố bị ngập lụt."
    ),

    # Cặp 4: Phủ định (Nghĩa trái ngược hoàn toàn).
    # Kỳ vọng: Đây là nhược điểm chung của nhiều model nhỏ. Chúng chia sẻ nhiều từ khóa (điện thoại, pin, chụp ảnh) nên model kém sẽ cho điểm cao, nhưng model hiểu sâu sẽ cho điểm thấp hơn cặp 1, 2.
    (
        "Chiếc máy này có thời lượng pin rất trâu và camera chụp ảnh cực kỳ đẹp.",
        "Chiếc máy này pin cực kỳ yếu và chụp ảnh rất xấu."
    ),

    # Cặp 5: Cú lừa từ vựng (Chơi chữ đồng âm dị nghĩa).
    # Kỳ vọng: Điểm phải rất thấp (< 0.2). Cùng xuất hiện chữ "táo" và "cam", nhưng một bên là trái cây, một bên là thương hiệu công nghệ. Model xịn sẽ không bao giờ bị lừa bởi mặt chữ.
    (
        "Sáng nay tôi ra chợ mua một ít quả táo và quả cam về làm nước ép.",
        "Hãng Apple chuẩn bị ra mắt mẫu iPhone mới có phiên bản màu cam."
    )
]

def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Hàm tính toán Cosine Similarity giữa 2 vector (score từ -1 đến 1)"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    # Tính độ lớn (norm) của vector để tránh lỗi chia cho 0
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    # Công thức: (v1 dot v2) / (|v1| * |v2|)
    return np.dot(v1, v2) / (norm_v1 * norm_v2)


def main():
    print("=== 1. KHỞI TẠO EMBEDDING MODEL ===")
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

    print(f"[*] Đang sử dụng mô hình nhúng: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}\n")

    print("=== 2. TÍNH ĐIỂM TƯƠNG ĐỒNG (COSINE SIMILARITY) ===")
    for i, (cau_1, cau_2) in enumerate(SENTENCE_PAIRS, start=1):
        # Sử dụng hàm __call__ của embedder bạn đã viết để biến text thành list[float]
        vec_1 = embedder(cau_1)
        vec_2 = embedder(cau_2)
        
        # Tính điểm
        score = compute_cosine_similarity(vec_1, vec_2)
        
        print(f"Cặp {i}:")
        print(f"  - Câu A: {cau_1}")
        print(f"  - Câu B: {cau_2}")
        print(f"  => Score (Độ tương đồng): {score:.4f}\n")


if __name__ == "__main__":
    main()