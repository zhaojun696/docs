#%%

token='v1U13Q0h69AFIITYwtiR8ZxaAGEHgy2lPPBF1Ca7'
import http.client

conn = http.client.HTTPConnection("154.64.231.144:8080")

headers = { 'xc-token': token }

conn.request("GET", "/api/v2/tables/mtz1bobywo4jqgt/records?offset=0&limit=25&where=&viewId=vwnmn46i3j0hv0z3", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
# %%
import requests
token='v1U13Q0h69AFIITYwtiR8ZxaAGEHgy2lPPBF1Ca7'

# 定义API的URL
url = "http://154.64.231.144:8080/api/v2/storage/upload"

# 查询参数
query_params = {
    'path':'upload/test/cc',
    # "scope": "workspacePics"
}

# 请求头
headers = {
    "xc-token":token
}

with open('D:/test.jpg', 'rb') as file:
    files = {'file': ('filename.jpg', file, 'image/jpeg')}  # 指定文件名和 mimetype
    response = requests.post(url, params=query_params, 
                files=files, headers=headers)


    # 检查响应状态码
    if response.status_code == 200:
        print("文件上传成功!")
        print(response.json())  # 如果响应是JSON格式，可以解析并打印
    else:
        print(f"文件上传失败，状态码: {response.status_code}")
        print(response.text)  # 打印错误信息

