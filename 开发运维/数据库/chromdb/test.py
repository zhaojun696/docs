import chromadb
client = chromadb.PersistentClient(path="./")
collection = client.create_collection(name="testname")

# 获取一个存在的Collection对象
collection = client.get_collection("testname")
collection.add(
    documents=["这是一个文档", "这是另一个文档"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)