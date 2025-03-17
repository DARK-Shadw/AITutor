from db import VDB

# vector_db = VDB("Test", pdf_path="Data\dbms.pdf")     Load or create Vector Data Base


vector_db = VDB("syllabus", pdf_path="Data\syllabus_cs_2023.pdf", chunk_size=1000, chunk_overlap=300)

query = "o Data structures – Arrays- one dimensional and two dimensional- - representation.Linked lists- singly, doubly and circular- Applications of linked lists"

print()
print("------Plain------")
print()
result = vector_db.similarity_search(query, k=3)
for res in result:
    print(res.page_content)
    print("*" *10)

print()
print("------Vector------")
print()

result = vector_db.similarity_search_with_vector(query, k=3)
for res in result:
    print(res.page_content)
    print("*" *10)

print()
print("------Score------")
print()
result = vector_db.similarity_search_with_score(query, k=3)
for data, score in result:
    print(f"S: {score:3f}, Data: {data.page_content}")
