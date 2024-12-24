from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from supabase import create_client
from dotenv import load_dotenv
import os
import tiktoken

# Carregar variáveis do .env apenas no desenvolvimento local
if os.path.exists(".env"):
    load_dotenv()

# Configurando o cliente Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Inicializar FastAPI
router = APIRouter()

# Modelo para o corpo da requisição
class EmbeddingRequest(BaseModel):
    content: str
    id_knowledge: str
    id_company: str
    id_agente: str

# Inicializar componentes do LangChain
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50, separator=" ")
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)

@router.post("/generate-embeddings-texto/")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        # Verificar se a empresa está registrada
        company_response = supabase.table("companies").select("id").eq("id", request.id_company).execute()
        if not company_response.data:
            raise HTTPException(status_code=400, detail="Sua empresa não está registrada. Acesse nossa página de vendas para assinar o app.")

        # Consultar saldo de tokens da empresa
        tokens_data = supabase.table("tokens").select("tokens_ia").eq("id_company", request.id_company).execute()
        if not tokens_data.data or tokens_data.data[0]["tokens_ia"] <= 0:
            raise HTTPException(status_code=400, detail="Você atingiu o limite de tokens. Acesse o link para adquirir mais.")

        token_balance = tokens_data.data[0]["tokens_ia"]
        print(f"Quantidade de Tokens disponíveis: {token_balance}")

        # Inicializar o encoder para contagem de tokens
        enc = tiktoken.encoding_for_model("text-embedding-ada-002")
        total_tokens_used = 0

        # Dividir o conteúdo em pedaços
        chunks = text_splitter.split_text(request.content)
        print(f"Número de pedaços gerados: {len(chunks)}")

        # Processar cada pedaço e gerar embeddings
        for i, chunk in enumerate(chunks):
            # Contar tokens usados para o chunk atual
            tokens_in_chunk = len(enc.encode(chunk))
            total_tokens_used += tokens_in_chunk

            # Verificar se há saldo suficiente
            if total_tokens_used > token_balance:
                raise HTTPException(
                    status_code=400,
                    detail=f"Você atingiu o limite de tokens disponíveis. Tokens necessários: {total_tokens_used}, tokens disponíveis: {token_balance}."
                )

            print(f"[{i+1}] Processando chunk: {chunk[:50]}... Tokens: {tokens_in_chunk}")
            embedding = embeddings.embed_query(chunk)

            # Inserir no Supabase
            response = supabase.table("embeddings").insert({
                "id_knowledge": request.id_knowledge,
                "content": chunk,
                "embedding": embedding,
                "agent_id": request.id_agente
            }).execute()

            if not response.data:
                raise HTTPException(status_code=500, detail=f"Erro ao salvar no Supabase: {response}")

        # Atualizar o campo "tokens_ia" na tabela "tokens"
        updated_tokens = token_balance - total_tokens_used
        supabase.table("tokens").update({"tokens_ia": updated_tokens}).eq("id_company", request.id_company).execute()
        print(f"Saldo atualizado de tokens: {updated_tokens}")

        return {"message": "Embeddings gerados e salvos com sucesso!", "tokens_used": total_tokens_used, "remaining_tokens": updated_tokens}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
