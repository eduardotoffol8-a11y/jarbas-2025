from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os

# Importa as funções do nosso módulo de lógica
from core import rag

# Cria a instância da aplicação FastAPI
app = FastAPI(
    title="API do Projeto Jarbas",
    description="API para gerenciar a memória e a inteligência do assistente Jarbas.",
    version="1.0.0"
)

# Define um diretório para uploads temporários
PASTA_UPLOADS = "temp_uploads"
os.makedirs(PASTA_UPLOADS, exist_ok=True)

# --- MODELOS DE DADOS (para validação de entrada/saída) ---

class QueryRequest(BaseModel):
    pergunta: str

class QueryResponse(BaseModel):
    resposta: str

class UploadResponse(BaseModel):
    status: str
    mensagem: str
    nome_arquivo: str


# --- ENDPOINTS DA API ---

@app.post("/upload", response_model=UploadResponse)
async def endpoint_upload_pdf(file: UploadFile = File(...)):
    """
    Recebe um arquivo PDF, o processa e atualiza a memória do Jarbas.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Formato de arquivo inválido. Apenas PDFs são aceitos.")

    # Salva o arquivo temporariamente
    caminho_temporario = os.path.join(PASTA_UPLOADS, file.filename)
    with open(caminho_temporario, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Chama a função de processamento do nosso módulo RAG
        rag.processar_e_armazenar_pdf(caminho_temporario, file.filename)
        return {
            "status": "sucesso",
            "mensagem": "Arquivo processado e memória do Jarbas atualizada.",
            "nome_arquivo": file.filename
        }
    except Exception as e:
        # Em caso de erro, retorna uma mensagem clara
        raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo: {str(e)}")
    finally:
        # Limpa o arquivo temporário após o processamento
        os.remove(caminho_temporario)


@app.post("/query", response_model=QueryResponse)
async def endpoint_query(request: QueryRequest):
    """
    Recebe uma pergunta, busca na memória e retorna a resposta da IA.
    """
    try:
        contexto = rag.buscar_contexto_relevante(request.pergunta)
        if not contexto:
            return {"resposta": "Não encontrei informações relevantes nos documentos para responder a essa pergunta."}
            
        resposta_final = rag.gerar_resposta_com_contexto(contexto, request.pergunta)
        return {"resposta": resposta_final}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar a resposta: {str(e)}")


@app.get("/")
def read_root():
    return {"Projeto": "Jarbas", "Status": "API Online"}