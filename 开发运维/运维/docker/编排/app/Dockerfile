# fast_app/Dockerfile
FROM ubuntu:latest


# 设置环境变量以避免交互
ENV DEBIAN_FRONTEND=noninteractive

# 安装必要的依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装 FastAPI 和 Uvicorn
RUN pip3 install fastapi uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置工作目录
WORKDIR /app

# 复制应用代码到容器中
COPY . /app



# 暴露端口
EXPOSE 8000

# 运行应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]