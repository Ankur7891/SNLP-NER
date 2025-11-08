from fastapi import FastAPI, Body
from src.agent import run_agent

app = FastAPI(title="Biomedical NER & Relation Extraction API", version="1.0")


@app.post("/extract")
async def extract(text: str = Body(..., embed=True)):
    """Extract Biomedical Entities and Relations."""
    result = run_agent(text)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=True)
