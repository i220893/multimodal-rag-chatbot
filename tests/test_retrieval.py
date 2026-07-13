from src.retrieval.retriever import Retriever

retriever = Retriever()

print("\n🔍 Text Query Test ----")
res = retriever.search_text("What was the revenue in 2023?", k=5)
print(res)

print("\n🖼 Image Query Test ----")
# Pick a page image from data/page_images
res_img = retriever.search_image("data/page_images/page_1.png", k=5)
print(res_img)
