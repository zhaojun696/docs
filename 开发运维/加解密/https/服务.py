from fastapi import FastAPI,Body
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/submit")
def submit_data(data: dict = Body(...)):
    return {"received_data": data}

if __name__ == "__main__":
    uvicorn.run(
        "服务:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="private_key.pem",
        ssl_certfile="certificate.pem"
    )