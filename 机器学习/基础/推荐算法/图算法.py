#%%
from py2neo import Graph, Node, Relationship

# 连接到Neo4j数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4j"))

# 清空数据库
graph.delete_all()

# 创建用户节点
users = ["Alice", "Bob", "Charlie"]
for user in users:
    graph.create(Node("User", name=user))

# 创建电影节点
movies = {
    "Inception": ["Action", "Sci-Fi"],
    "The Matrix": ["Action", "Sci-Fi"],
    "Forrest Gump": ["Drama", "Romance"],
    "The Shawshank Redemption": ["Drama"]
}
for movie, genres in movies.items():
    movie_node = Node("Movie", title=movie)
    graph.create(movie_node)
    for genre in genres:
        genre_node = Node("Genre", name=genre)
        graph.merge(genre_node, "Genre", "name")
        graph.create(Relationship(movie_node, "IS_OF_GENRE", genre_node))

# 创建用户和电影之间的关系
ratings = [
    ("Alice", "Inception", 5),
    ("Alice", "The Matrix", 4),
    ("Bob", "Forrest Gump", 5),
    ("Bob", "The Shawshank Redemption", 5),
    ("Charlie", "Inception", 4),
    ("Charlie", "The Matrix", 5)
]
for user, movie, rating in ratings:
    user_node = graph.nodes.match("User", name=user).first()
    movie_node = graph.nodes.match("Movie", title=movie).first()
    graph.create(Relationship(user_node, "RATED", movie_node, rating=rating))


#%%
# 为Alice推荐电影
query = """
MATCH (u:User {name: 'Alice'})-[:RATED]->(m1:Movie)<-[:RATED]-(u2:User)-[:RATED]->(m2:Movie)
WHERE NOT (u)-[:RATED]->(m2)
RETURN m2.title AS Recommendation, COUNT(*) AS Frequency
ORDER BY Frequency DESC
LIMIT 5
"""
recommendations = graph.run(query).data()
print("Recommendations for Alice:")
for rec in recommendations:
    print(f"{rec['Recommendation']} (Frequency: {rec['Frequency']})")