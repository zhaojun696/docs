# 可视化

## pyvis



#%%
from py2neo import Graph
from pyvis.network import Network

# 连接到Neo4j数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# 查询Neo4j数据库中的节点和关系
query = """
MATCH (n)-[r]->(m)
RETURN n, r, m
"""
data = graph.run(query).data()

# 创建一个pyvis网络图
net = Network(notebook=True, height="750px", width="100%")

# 添加节点和边到网络图中
for record in data:
    src_node = record['n']
    rel = record['r']
    tgt_node = record['m']
    
    src_id = src_node.identity
    tgt_id = tgt_node.identity
    
    src_label = src_node.get('name', str(src_id))
    tgt_label = tgt_node.get('name', str(tgt_id))
    
    # 获取节点的属性信息
    src_properties = "<br>".join([f"{k}: {v}" for k, v in src_node.items()])
    tgt_properties = "<br>".join([f"{k}: {v}" for k, v in tgt_node.items()])
    
    net.add_node(src_id, label=src_label, title=src_properties)
    net.add_node(tgt_id, label=tgt_label, title=tgt_properties)
    net.add_edge(src_id, tgt_id, title=rel.type())

# 显示网络图
net.show("example.html")