# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Mạc Phương Nga
**Nhóm:** C401-F1
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity giữa 2 vector là khi chúng có hướng gần giống nhau trong không gian vector, tức là chúng biểu diễn ý nghĩa tương tự. Điều này thường xảy ra khi 2 câu có nội dung hoặc ngữ cảnh gần giống nhau.

**Ví dụ HIGH similarity:**
- Sentence A: Chẩn đoán hình ảnh
- Sentence B: Siêu âm
- Tại sao tương đồng: 

**Ví dụ LOW similarity:**
- Sentence A: Chẩn đoán hình ảnh
- Sentence B: Học máy
- Tại sao khác:

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì trong không gian embedding vector, khoảng cách Euclidean có thể bị ảnh hưởng bởi độ dài của vector (magnitude), trong khi cosine similarity chỉ đo lường góc giữa 2 vector, giúp tập trung vào hướng (nghĩa) hơn là độ dài. Điều này làm cho cosine similarity phù hợp hơn để đánh giá sự tương đồng về mặt ngữ nghĩa giữa các câu.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> [(10,000 - 500) / (500 - 50)] + 1 = 23 chunks
> *Đáp án: 23 chunks*

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap = 100, số chunks: [(10,000 - 500) / (500 - 100)] + 1 = 25 chunks. Overlap nhiều hơn giúp giữ nguyên ngữ cảnh giữa các chunks, đặc biệt quan trọng khi câu bị cắt ngang. Điều này có thể cải thiện chất lượng retrieval vì các chunks sẽ chứa nhiều thông tin liên quan hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn


**Domain:** Luật lao động Việt Nam


**Tại sao nhóm chọn domain này?**
> Luật lao động Việt Nam là một lĩnh vực phức tạp với nhiều quy định khác nhau. Việc áp dụng RAG cho domain này sẽ giúp người dùng dễ dàng tra cứu thông tin và giải đáp các thắc mắc liên quan đến luật lao động.
### Data Inventory


| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 |Bộ luật lao động 2019 | https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/12/45.signed.pdf | 193202 | Không có |


### Baseline Analysis


Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:


| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Luật lao động Việt Nam | FixedSizeChunker (`fixed_size`) | 1074 | 199.87 | Không |
| Luật lao động Việt Nam | SentenceChunker (`by_sentences`) | 554 | 346.92 | Có |
| Luật lao động Việt Nam | RecursiveChunker (`recursive`) | 1652 | 115.27 | Có |

### Strategy Của Tôi

**Loại:** FixedSizeChunker với chunk_size=800, overlap=100

**Mô tả cách hoạt động:**
> Chia docs thành các đoạn 800 ký tự, mỗi đoạn chồng lấn 100 ký tự với đoạn trước đó. Điều này giúp đảm bảo rằng các câu không bị cắt ngang một cách thô bạo, đồng thời vẫn giữ được ngữ cảnh gần đó.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Chọn FixedSizeChunker vì luật pháp thường có cấu trúc câu dài và phức tạp. Chunk size lớn giúp giữ nguyên ý nghĩa của các điều luật, trong khi overlap đảm bảo rằng các phần quan trọng không bị cắt ngang, từ đó cải thiện chất lượng retrieval.
> Chia mỗi người 1 strategy khác nhau để so sánh hiệu quả.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Bộ luật Lao Động Việt Nam 2019 | **của tôi** | 297 | 800 | Good |

### So Sánh Với Thành Viên Khác


| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Lê Duy Anh | Custom Strategy (Regex Based Chunking) | 8.5 | Bảo toàn ngữ cảnh tốt | Khi điều luật quá dài, đoạn chunk sinh ra sẽ vượt qua giới hạn context window. Hao phí khi embedding. Sự thừa thãi khi truy xuất.  |
| Lại Gia Khánh | Semantic Chunking | 8 | Giữ nguyên đơn vị nghĩa (câu/điều), cải thiện độ chính xác truy vấn và khả năng trích dẫn nguồn; giảm nhiễu khi trả lời câu hỏi chuyên sâu. | Phụ thuộc vào chất lượng embedding và ngưỡng similarity; cần tinh chỉnh threshold; tốn tài nguyên hơn và có thể tạo chunk kích thước không đồng đều. |
| Mạc Phương Nga | FixedSizeChunker | 10 | Xử lý đơn giản, nhanh. Kiểm soát được lượng token đưa vào LLM | Phụ thuộc nhiều vào chunk_size và overlap, cần kiểm thử nhiều lần để tìm cặp thông số tối ưu. |
| Nguyễn Phạm Trà My | AgenticChunker |6| Linh hoạt trong việc quản lý ngữ cảnh | Chi phí cao và tốc độ xử lý chậm do phụ thuộc hoàn toàn vào việc gọi API từ LLM cho từng đoạn văn bản.|
| Trương Minh Sơn |Parent–Child |892 | 287.34| 7.8/10|Trả lời câu hỏi, tìm chunks khá chính xác, Retrieval tìm đúng chunk quan trọng (Top-1 thường chứa đáp án).| Test thêm queries, có queries bị lan man không đúng trọng tâm dù tìm đúng đoạn chunk đoạn thông tin cần trả lời, có case bị lost-track information.’Top-K còn nhiều chunk không liên quan → context bị nhiễu
| Bùi Trần Gia Bảo| DocumentStructureChunker | 6/10| Giữ nguyên cấu trúc tài liệu (heading, section), rất phù hợp với văn bản markdown pháp lý, giúp truy xuất theo ngữ cảnh rõ ràng. | Phụ thuộc vào chất lượng định dạng markdown; nếu cấu trúc không chuẩn hoặc quá dài, chunk có thể mất cân bằng và ảnh hưởng hiệu quả retrieval. |



**Strategy nào tốt nhất cho domain này? Tại sao?**
> Strategy FixedSizeChunker với chunk_size=800 và overlap=100 là tốt nhất cho domain luật pháp vì nó giúp giữ nguyên ý nghĩa của các điều luật dài, đồng thời đảm bảo rằng các phần quan trọng không bị cắt ngang. Điều này cải thiện chất lượng retrieval và giúp agent trả lời chính xác hơn.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> SentenceChunker sử dụng Regex để tìm các dấu kết thúc câu. Sau đó gom 3 câu lại thành 1 chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Chunker đệ quy tài liệu bằng các separator theo thứ tự ưu tiên cho đến khi len(chunk) <= chunk_size.>

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Dữ liệu được lưu trữ dưới dạng một list các dict, mỗi dict chứa text gốc, metadata và embedded vector. Khi search, tính cosine similarity giữa query embedding và document embeddings, sau đó trả về top-k kết quả có điểm số cao nhất.

**`search_with_filter` + `delete_document`** — approach:
> Lọc theo metadata trước khi tính similarity để thu hẹp phạm vi tìm kiếm. Khi delete, loại bỏ document khỏi list và cập nhật lại embedding store.

### KnowledgeBaseAgent

**`answer`** — approach:
> Dùng `EmbeddingStore` tìm kiếm các đoạn văn có liên quan nhất đến câu hỏi người dùng. Đoạn này được đưa vào prompt cùng câu hỏi. Bằng cách inject context trực tiếp vào prompt, agent có thể tạo ra câu trả lời dựa trên dữ liệu cụ thể trong kho tri thức thay vì chỉ dựa vào kiến thức huấn luyện sẵn có của mô hình.

### Test Results

```
================================================= test session starts =================================================
platform win32 -- Python 3.10.11, pytest-9.0.3, pluggy-1.6.0 -- D:\Project\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Project\2A202600124-MacPhuongNga-Day07
plugins: anyio-4.13.0
collected 42 items                                                                                                     

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                            [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                     [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                              [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                               [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                    [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                    [ 14%] 
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                          [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                           [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                         [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                           [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                           [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                      [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                  [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                            [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                   [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                       [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                 [ 40%] 
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                       [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                           [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                             [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                               [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                     [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                          [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                            [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                             [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                      [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                     [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                            [ 71%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                               [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                     [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                          [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                            [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                             [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                      [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                     [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                            [ 71%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                          [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                            [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                             [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                      [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                     [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                            [ 71%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                             [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                      [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                     [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                            [ 71%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                      [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                     [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                            [ 71%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                            [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                            [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                       [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                           [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                 [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                           [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED        [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                      [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                     [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filterreturns_true_for_existing_doc PASSED       [100%]

================================================= 42 passed in 0.13s ==================================================
```

**Số tests pass:**  42/ 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán (Logic người) | Actual Score | Đúng dự đoán? |
|------|-----------|-----------|-----------------------|--------------|---------------|
| 1 | "Con mèo đen đang ngủ say trên chiếc ghế sofa ngoài phòng khách." | "Chú mèo mun đang nằm thiu thiu trên ghế nệm ở ngoài sảnh." | High (Cùng nghĩa) | 0.5706 | Không hẳn (Thấp hơn kỳ vọng) |
| 2 | "Đội tuyển Việt Nam đã xuất sắc đánh bại Thái Lan trong trận chung kết." | "Thái Lan đã phải nhận thất bại trước đội tuyển Việt Nam ở trận đấu cuối cùng." | High (Cùng nghĩa) | 0.6827 | Có |
| 3 | "Trời hôm nay mưa rất to và có sấm chớp dữ dội." | "Mùa mưa năm nay đến sớm hơn thường lệ làm nhiều tuyến phố bị ngập lụt." | Low/Medium (Cùng chủ đề) | 0.5189 | Có |
| 4 | "Chiếc máy này có thời lượng pin rất trâu và camera chụp ảnh cực kỳ đẹp." | "Chiếc máy này pin cực kỳ yếu và chụp ảnh rất xấu." | Low (Trái nghĩa hoàn toàn) | 0.8633 | **Sai hoàn toàn** |
| 5 | "Sáng nay tôi ra chợ mua một ít quả táo và quả cam về làm nước ép." | "Hãng Apple chuẩn bị ra mắt mẫu iPhone mới có phiên bản màu cam." | Low (Khác ngữ cảnh) | 0.3537 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất chắc chắn là **Cặp 4**. Mặc dù hai câu mang ý nghĩa hoàn toàn trái ngược nhau (khen vs chê), điểm similarity lại cao nhất trong cả 5 cặp (0.8633). Ngược lại, Cặp 1 có ý nghĩa giống nhau 100% (chỉ dùng từ đồng nghĩa) lại chỉ đạt 0.5706.
> 
> Điều này cho thấy mô hình embedding đang sử dụng (all-MiniLM-L6-v2) bị phụ thuộc rất nặng vào **Lexical Overlap (Sự trùng lặp từ vựng)**. Nó nhận diện được hai câu Cặp 4 cùng nói về một chủ đề hẹp (chiếc máy, pin, camera, chụp ảnh) và gom chúng lại rất gần nhau trong không gian vector. Tuy nhiên, nó lại thất bại trong việc nắm bắt các từ mang tính từ chối/phủ định định hướng logic (như "yếu", "xấu" so với "trâu", "đẹp"). Ngược lại, ở Cặp 1, vì không có từ nào viết giống hệt nhau (mèo đen/mèo mun, sofa/ghế nệm), mô hình đã đánh giá chúng xa nhau hơn thực tế.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | "Bộ luật Lao động năm 2019 (Luật số 45/2019/QH14) chính thức có hiệu lực thi hành kể từ ngày tháng năm nào?" | "Ngày 01 tháng 01 năm 2021" |
| 2 | "Theo Bộ luật Lao động 2019, hợp đồng lao động được phân loại thành mấy loại chính? Đó là những loại nào?" | "Gồm 02 loại chính: Hợp đồng lao động không xác định thời hạn và Hợp đồng lao động xác định thời hạn (thời hạn không quá 36 tháng). (Lưu ý: Đã bỏ Hợp đồng lao động theo mùa vụ hoặc theo một công việc nhất định có thời hạn dưới 12 tháng so với bộ luật cũ)." |
| 3 | "Quy định pháp luật không cho phép áp dụng thời gian thử việc đối với trường hợp người lao động giao kết loại hợp đồng lao động nào?" | "Không áp dụng thử việc đối với người lao động giao kết hợp đồng lao động có thời hạn dưới 01 tháng." |
| 4 | "Theo quy định, thời gian thử việc tối đa đối với công việc của người quản lý doanh nghiệp (theo quy định của Luật Doanh nghiệp, Luật Quản lý, sử dụng vốn nhà nước đầu tư vào sản xuất, kinh doanh tại doanh nghiệp) là bao nhiêu ngày?"| "Không quá 180 ngày." |
| 5 | "Trong dịp lễ Quốc khánh 02/9, người lao động được nghỉ làm việc và hưởng nguyên lương tổng cộng bao nhiêu ngày?" | "02 ngày (Bao gồm ngày 02 tháng 9 dương lịch và 01 ngày liền kề trước hoặc sau ngày 02 tháng 9)." |
| 6 | "Lộ trình điều chỉnh tuổi nghỉ hưu đối với người lao động làm việc trong điều kiện lao động bình thường được thực hiện cho đến khi đạt mức độ tuổi nào đối với nam và nữ?" | "Nam đạt đủ 62 tuổi (vào năm 2028) và Nữ đạt đủ 60 tuổi (vào năm 2035)." |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Bộ luật Lao động năm 2019 (Luật số 45/2019/QH14) chính thức có hiệu lực thi hành kể từ ngày tháng năm nào? | "iải quyết tranh chấp hoặc một trong các bên ..." | 0.881 | Có | Bộ luật Lao động năm 2019 có hiệu lực thi hành từ ngày **01 tháng 01 năm 2021** |
| 2 | Theo Bộ luật Lao động 2019, hợp đồng lao động được phân loại thành mấy loại chính? Đó là những loại nào? | "ức hợp đồng lao động 1. Hợp đồng lao động ..." | 0.851 | Có | "- Hợp đồng lao động được phân loại thành 2 loại chính: Hợp đồng lao động không xác định thời hạn, Hợp đồng lao động xác định thời hạn." |
| 3 | uy định pháp luật không cho phép áp dụng thời gian thử việc đối với trường hợp người lao động giao kết loại hợp đồng lao động nào? | "ng lao động có hiệu lực kể từ ngày ..." | 0.656 | Có | "Không áp dụng thử việc đối với người lao động giao kết hợp đồng lao động có thời hạn dưới 01 tháng." |
| 4 | "Theo quy định, thời gian thử việc tối đa đối với công việc của người quản lý doanh nghiệp (theo quy định của Luật Doanh nghiệp, Luật Quản lý, sử dụng vốn nhà nước đầu tư vào sản xuất, kinh doanh tại doanh nghiệp) là bao nhiêu ngày? | "a công việc nhưng chỉ được thử việc một lần..." | 0.624 | Có | "Tối đa 180 ngày."|
| 5 | "Trong dịp lễ Quốc khánh 02/9, người lao động được nghỉ làm việc và hưởng nguyên lương tổng cộng bao nhiêu ngày?" | "ằng tuần - 1. Mỗi tuần,..." | 0.758 | Có | "Người lao động được nghỉ 02 ngày (ngày 02 tháng 9 và 01 ngày liền kề trước hoặc sau)." |
| 6 | "Lộ trình điều chỉnh tuổi nghỉ hưu đối với người lao động làm việc trong điều kiện lao động bình thường được thực hiện cho đến khi đạt mức độ tuổi nào đối với nam và nữ?" | "g lao động đóng bảo hiểm ..." | 0.715 | Có | "Đối với lao động nam: đủ 62 tuổi vào năm 2028. Đối với lao động nữ: đủ 60 tuổi vào năm 2035." |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 6/6 

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Việc chunk bằng FixedSizeChunker với chunk_size lớn hơn (800) và overlap nhiều hơn (100) đã giúp cải thiện đáng kể chất lượng retrieval. Điều này đặc biệt quan trọng trong domain luật pháp, nơi các câu thường dài và chứa nhiều thông tin quan trọng. 
> Agentic chunking chỉ nên dùng khi các strategy khác không hiệu quả.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nên thử nghiệm từ các strategy đơn giản trước (FixedSizeChunker) trước khi chuyển sang các strategy phức tạp hơn (RecursiveChunker). Đôi khi, một strategy đơn giản nhưng được điều chỉnh tốt (chunk_size và overlap phù hợp) có thể mang lại kết quả retrieval tốt hơn so với các strategy phức tạp.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Thử nghiệm với chunk_size nhỏ hơn (ví dụ 500) để xem liệu có cải thiện retrieval cho các câu hỏi cụ thể hơn không. 

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 8 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 0 / 5 |
| **Tổng** | | **83 / 100** |
