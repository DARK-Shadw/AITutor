from db import VDB

vector_db = VDB("Syllabus", pdf_path="Data\syllabus_cs_2023.pdf", chunk_size=1500, chunk_overlap=150)

query = "Dijkstra's algorithm"

result = vector_db.retriever.invoke(query)

for res in result:
    print(res.page_content)
    print("#"*20)

class Agent:
    def __init__(self):
        pass