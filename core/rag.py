import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
from . import config

# Configura a API do Google
genai.configure(api_key=config.GOOGLE_API_KEY)

# --- CONFIGURAÇÕES ---
NOME_MODELO_EMBEDDING = "models/text-embedding-004"
NOME_MODELO_GERACAO = "gemini-1.5-flash"
NOME_COLECAO_DB = "jarbas_memory"
CAMINHO_DB = "db"

# --- INICIALIZAÇÃO DO BANCO DE DADOS E MODELOS ---

# Inicializa o cliente do ChromaDB para persistir os dados na pasta 'db'
client = chromadb.PersistentClient(path=CAMINHO_DB)

# Função de embedding usando a API do Google
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=config.GOOGLE_API_KEY)

# Obtém ou cria a coleção no banco de dados
# A função de embedding é especificada aqui para que o ChromaDB saiba como lidar com as buscas
collection = client.get_or_create_collection(
    name=NOME_COLECAO_DB,
    embedding_function=google_ef
)

# Inicializa o modelo generativo do Gemini
model = genai.GenerativeModel(NOME_MODELO_GERACAO)


# --- FUNÇÕES PRINCIPAIS ---

def processar_e_armazenar_pdf(caminho_do_arquivo: str, nome_original_arquivo: str):
    """
    Lê um arquivo PDF, extrai o texto, o divide em chunks e o armazena no ChromaDB.
    """
    print(f"Iniciando processamento do arquivo: {nome_original_arquivo}")

    # 1. Extrair texto do PDF
    doc = fitz.open(caminho_do_arquivo)
    texto_completo = ""
    for page in doc:
        texto_completo += page.get_text()
    doc.close()

    if not texto_completo.strip():
        print("AVISO: Nenhum texto foi extraído do PDF.")
        return

    # 2. Dividir o texto em chunks (pedaços)
    # Uma estratégia simples: dividir por parágrafos (linhas em branco)
    chunks = [chunk for chunk in texto_completo.split('\n\n') if chunk.strip()]
    
    if not chunks:
        print("AVISO: Não foi possível dividir o texto em chunks.")
        return

    print(f"Texto dividido em {len(chunks)} chunks. Armazenando no banco de dados...")

    # 3. Adicionar os chunks ao ChromaDB
    # O ID de cada chunk será o nome do arquivo + o número do chunk
    ids = [f"{nome_original_arquivo}_{i}" for i, _ in enumerate(chunks)]
    
    # O ChromaDB usará a `google_ef` configurada na coleção para gerar os embeddings automaticamente
    collection.add(
        documents=chunks,
        ids=ids
    )
    print("Arquivo processado e armazenado na memória do Jarbas com sucesso.")


def buscar_contexto_relevante(pergunta_usuario: str) -> str:
    """
    Busca no ChromaDB os chunks mais relevantes para a pergunta do usuário.
    """
    print(f"Buscando contexto para a pergunta: '{pergunta_usuario}'")
    
    # Realiza a busca na coleção. Pede os 3 resultados mais relevantes.
    results = collection.query(
        query_texts=[pergunta_usuario],
        n_results=3
    )
    
    # Concatena os documentos encontrados em uma única string de contexto
    contexto = "\n\n---\n\n".join(results['documents'][0])
    print("Contexto encontrado.")
    return contexto


def gerar_resposta_com_contexto(contexto: str, pergunta: str) -> str:
    """
    Usa o modelo Gemini para gerar uma resposta com base no contexto e na pergunta.
    """
    print("Gerando resposta com o modelo de IA...")
    
    prompt = f"""
    Você é o Jarbas, uma inteligência artificial assistente.
    Sua tarefa é responder à pergunta do usuário de forma clara, objetiva e exclusivamente com base no contexto fornecido.
    Não utilize nenhum conhecimento externo. Se a resposta não estiver no contexto, diga "A informação não foi encontrada nos documentos fornecidos."

    **Contexto:**
    {contexto}

    **Pergunta do Usuário:**
    {pergunta}

    **Sua Resposta:**
    """
    
    # Chama a API do Gemini para gerar a resposta
    response = model.generate_content(prompt)
    
    print("Resposta gerada.")
    return response.text