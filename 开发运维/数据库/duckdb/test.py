#%%
import duckdb

# 连接到DuckDB内存数据库
con = duckdb.connect(':memory:')

# 创建示例表
con.execute("CREATE TABLE sales (date DATE, amount INTEGER)")
con.execute("INSERT INTO sales VALUES ('2022-01-01', 100), ('2022-01-02', 200), ('2022-01-03', 150), ('2022-01-04', 300)")

# 使用窗口函数实现滑动窗口
result = con.execute("""
    SELECT date, amount, 
           SUM(amount) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS sliding_sum
    FROM sales
""")

# 获取查询结果并打印
result_df = result.fetchdf()
result_df
#%%
import duckdb
from contextlib import suppress
import numpy as np
import pandas as pd
import statsmodels.api as sm

def mock(i: int):
    nobs = 10
    X = np.random.random((nobs, 2))
    beta = [1, .1, .5]
    e = np.random.random(nobs)
    y = np.dot(sm.add_constant(X), beta) + e
    return pd.DataFrame(X, columns=["x1", "x2"]).assign(y=y, key=f"c{i:0>4}").filter(["key", "x1", "x2", "y"])


df = pd.concat([mock(i) for i in range(10000)])
df



def ols4(x: list, y: list) -> float:
    x=np.array(x)
    y=np.array(y)
    X = sm.add_constant(np.array(x))
    res = sm.OLS(y, X).fit()
    return res.params[0]

with suppress(Exception):
    duckdb.remove_function("ols4")

duckdb.create_function("ols4", ols4)

sql = """
with tmp as (
    select key, ols4(list((x1, x2)), list(y)) as coef
    from df
    group by key
)
select key,coef
from tmp
order by key
"""
%timeit duckdb.sql(sql).df()
# duckdb.sql(sql).df()



#%%



#%%
#%%
import duckdb as db
db.install_extension('vss')
db.load_extension('vss')
#%%
db.execute('CREATE TABLE my_vector_table (vec FLOAT[3]);')
db.execute('CREATE INDEX my_hnsw_index ON my_vector_table USING HNSW (vec);')
db.execute('''
INSERT INTO my_vector_table 
  SELECT  array_value(a, b, c) 
  FROM range(1, 10) ra(a),range(1, 10) rb(b),range(1, 10) rc(c);
''')
#%%
db.query('SELECT * FROM my_vector_table;')
#%%
db.query('''
SELECT vec,array_distance(vec, [1, 2, 3]::FLOAT[3]) as dis FROM my_vector_table 
ORDER BY dis
LIMIT 3;

''')


#%%
#%%
import duckdb as db
db.install_extension('fts')
db.load_extension('fts')

#%%
db.execute('''
CREATE TABLE documents5 (
    document_identifier VARCHAR,
    text_content VARCHAR,
    author VARCHAR,
    doc_version INTEGER
);



''')




db.execute('''
INSERT INTO documents5
    VALUES ('doc1',
            'The mallard is a dabbling duck that breeds throughout the temperate.',
            'Hannes Mühleisen',
            3),
           ('doc2',
            'The cat is a domestic species of small carnivorous mammal.',
            'Laurens Kuiper',
            2
           );
''')
db.execute('''
INSERT INTO documents5
    VALUES 
           ('doc3',
            '微云是腾讯公司为用户精心打造的一项智能云服务, 您可以通过微云方便地在手机和电脑之间同步文件、推送照片和传输数据。',
            'Laurens Kuiper',
            4
           );
''')

db.execute('''
INSERT INTO documents5
    VALUES 
           ('doc5',
            '微云 腾讯公司 用户 精心打造 智能云 服务',
            'zhaojun zhangpiao1',
            6
           );
''')

db.execute('''
INSERT INTO documents5
    VALUES 
           ('doc6',
            'ac bb xx love xx xx',
            'make cc',
            7
           );
''')

db.execute('''
INSERT INTO documents5
    VALUES 
           ('doc7',
            'Weiyun is an intelligent cloud service carefully designed by Tencent for users. You can easily synchronize files, push photos, and transfer data between your phone and computer through Weiyun.',
            'Laurens Kuiper',
            8
           );
''')

db.execute('''
PRAGMA create_fts_index(
    'documents5', 'document_identifier', 'text_content', 'author'
);
''')

#%%
db.query('from documents5').df()
#%%
db.query('''
SELECT document_identifier, text_content, score
FROM (
    SELECT *, fts_main_documents5.match_bm25(
        document_identifier,
        'Mühleisen',
        fields := 'author,text_content'
    ) AS score
    FROM documents5
) sq
WHERE score IS NOT NULL
  AND doc_version > 2
ORDER BY score DESC;
''').fetchdf()
#%%
db.query('''
SELECT document_identifier, text_content, score
FROM (
    SELECT *, fts_main_documents5.match_bm25(
        document_identifier,
        '微云'
    ) AS score
    FROM documents5
) sq
WHERE score IS NOT NULL
ORDER BY score DESC;
''').fetchdf()

#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm

def mock(i: int):
    nobs = 10
    X = np.random.random((nobs, 2))
    beta = [1, .1, .5]
    e = np.random.random(nobs)
    y = np.dot(sm.add_constant(X), beta) + e
    return pd.DataFrame(X, columns=["x1", "x2"]).assign(y=y, key=f"c{i:0>4}").filter(["key", "x1", "x2", "y"])


df = pd.concat([mock(i) for i in range(100)])
#%%
import duckdb
from contextlib import suppress

def ols4(x: list, y: list) -> list[float]:
    return [len(x),2.,3.,4.]

with suppress(Exception):
    duckdb.remove_function("ols4")

duckdb.create_function("ols4", ols4)
# duckdb.remove_function("ols4")

#%%
sql = """
with tmp as (
    select key, ols4(list((x1, x2)), list(y)) as coef
    from df
    group by key
)
select key, coef[1] as const, coef[2] as x1, coef[3] as x2
from tmp
order by key
"""
duckdb.sql(sql).df()