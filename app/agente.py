import os
import json
import tiktoken
from typing import List, Dict, Union, Any, Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from supabase import create_client
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv
import asyncio
from app.routes import pdftoken, process_docx_token, textotoken, txttoken, urlsitetoken, urlyoutubetoken
from fastapi.middleware.cors import CORSMiddleware

# Carregar variáveis do .env apenas no desenvolvimento local
if os.path.exists(".env"):
    load_dotenv()

# Configurações iniciais
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
API_KEY = os.getenv("API_KEY")

# Conectar ao cliente Supabase
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Carregar embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Nova implementação da Memória
class ChatMemoryBuffer(BaseMemory):
    """An updated implementation of conversation memory buffer."""
    
    chat_memory: List[BaseMessage] = []  # Definindo valor padrão
    output_key: Optional[str] = "answer"
    memory_key: str = "chat_history"
    return_messages: bool = True

    class Config:
        arbitrary_types_allowed = True  # Permite tipos arbitrários no Pydantic

    def __init__(
        self,
        chat_memory: Optional[List[BaseMessage]] = None,
        output_key: Optional[str] = "answer",
        memory_key: str = "chat_history",
        return_messages: bool = True
    ):
        """Initialize memory buffer with optional parameters."""
        super().__init__()
        self.chat_memory = chat_memory if chat_memory is not None else []
        self.output_key = output_key
        self.memory_key = memory_key
        self.return_messages = return_messages

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory = []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables from chat history."""
        if self.return_messages:
            return {self.memory_key: self.chat_memory}
        
        memory_string = ""
        for message in self.chat_memory:
            if isinstance(message, HumanMessage):
                memory_string += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                memory_string += f"Assistant: {message.content}\n"
        
        return {self.memory_key: memory_string.strip()}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if self.output_key is None:
            raise ValueError("output_key cannot be None")
            
        human_message = inputs.get("question", "")
        if human_message:
            self.chat_memory.append(HumanMessage(content=human_message))
        
        ai_message = outputs.get(self.output_key, "")
        if ai_message:
            self.chat_memory.append(AIMessage(content=ai_message))

    def add_message(self, message: Union[str, BaseMessage], is_human: bool = True) -> None:
        """Add a message to the chat memory."""
        if isinstance(message, str):
            message = HumanMessage(content=message) if is_human else AIMessage(content=message)
        self.chat_memory.append(message)

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]

class ChatSession:
    def __init__(self, agent_id: str, id_company: str):
        self.agent_id = agent_id
        self.id_company = id_company
        self.memory = ChatMemoryBuffer(
            chat_memory=[],  # Inicializando explicitamente com lista vazia
            output_key="answer",
            memory_key="chat_history",
            return_messages=True
        )

    async def initialize_memory(self):
        """Carrega o histórico existente do Supabase para esta sessão específica"""
        previous_messages = await load_memory(self.agent_id)
        for msg in previous_messages:
            if msg["type"] == "human":
                self.memory.add_message(msg["content"], is_human=True)
            elif msg["type"] == "ai":
                self.memory.add_message(msg["content"], is_human=False)
        return self

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}

    async def get_session(self, agent_id: str, id_company: str) -> ChatSession:
        """Retorna uma sessão existente ou cria uma nova"""
        session_key = f"{agent_id}:{id_company}"
        if session_key not in self._sessions:
            session = ChatSession(agent_id, id_company)
            await session.initialize_memory()
            self._sessions[session_key] = session
        return self._sessions[session_key]

    def clear_session(self, agent_id: str, id_company: str):
        """Remove uma sessão específica"""
        session_key = f"{agent_id}:{id_company}"
        if session_key in self._sessions:
            del self._sessions[session_key]

# Inicializar o gerenciador de sessões globalmente
session_manager = SessionManager()

# Função para atualizar saldo de tokens no Supabase
async def atualizar_saldo_tokens(idcompany, tokens_utilizados):
    response = supabase_client.from_("tokens").select("tokens_ia").eq("id_company", idcompany).execute()

    if response.data:
        saldo_atual = response.data[0]["tokens_ia"]
        novo_saldo = saldo_atual - tokens_utilizados

        supabase_client.from_("tokens").update({"tokens_ia": novo_saldo}).eq("id_company", idcompany).execute()
    else:
        print(f"Erro: Nenhuma empresa cadastrada com {idcompany}.")

async def verifica_token_company(idcompany: str):
    try:
        response = supabase_client.table("companies").select("id").eq("id", idcompany).execute()

        if not response.data:
            return False, "Sua empresa não está registrada. Acesse nossa página de vendas para assinar o app."

        tokens_data = supabase_client.table("tokens").select("tokens_ia").eq("id_company", idcompany).execute()

        if not tokens_data.data or not tokens_data.data[0].get("tokens_ia"):
            return False, "Erro: Não foi possível recuperar os tokens disponíveis."

        token_balance = tokens_data.data[0]["tokens_ia"]
        print(f"Quantidade de Tokens do usuário: {token_balance}")

        margem_saida_tokens = 1200

        print(f"Quantidade de Tokens necessária: {margem_saida_tokens}")

        if token_balance < margem_saida_tokens:
            return False, f"Você precisa de pelo menos {margem_saida_tokens} tokens para prosseguir. Atualmente, você possui {token_balance} tokens. Acesse o link para adquirir mais."

        return True, "Usuário e tokens verificados com sucesso."

    except Exception as e:
        print(f"Erro ao verificar tokens: {e}")
        return False, "Ocorreu um erro ao verificar seus tokens. Tente novamente mais tarde."

def calculate_cost(tokens: int, model: str = "gpt-4"):
    if model == "gpt-4":
        prompt_price_per_1k = 0.03
        completion_price_per_1k = 0.06
    elif model == "gpt-3.5":
        prompt_price_per_1k = 0.002
        completion_price_per_1k = 0.002
    else:
        raise ValueError(f"Modelo {model} não suportado para cálculo de custo.")

    return {
        "prompt_cost": (tokens / 1000) * prompt_price_per_1k,
        "completion_cost": (tokens / 1000) * completion_price_per_1k,
    }

def count_tokens(text: str, model: str = "gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

async def custom_retrieval(query, match_threshold=0.75, match_count=5, agent_id=None):
    query_embedding = await asyncio.to_thread(embeddings.embed_query, query)
    
    def rpc_call():
        response = supabase_client.rpc(
            "match_vectors",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
                "agent_id": agent_id,
            },
        ).execute()
        return response

    response = await asyncio.to_thread(rpc_call)

    documents = []
    for result in response.data:
        documents.append(
            Document(
                page_content=result["content"],
                metadata={
                    "id": result["id"],
                    "similarity": result["similarity"],
                    "agent_id": result["agent_id"],
                },
            )
        )
    return documents

async def save_memory(agent_id: str, messages: list):
    existing_data = supabase_client.table("chat_memory").select("chat_history").eq("agent_id", agent_id).execute()
    existing_messages = []
    if existing_data.data:
        existing_messages = json.loads(existing_data.data[0]["chat_history"])

    new_messages = [msg for msg in messages if msg not in existing_messages]
    combined_messages = existing_messages + new_messages

    user_message = next((msg["content"] for msg in reversed(combined_messages) if msg["type"] == "human"), "")
    assistant_message = next((msg["content"] for msg in reversed(combined_messages) if msg["type"] == "ai"), "")

    data = {
        "agent_id": agent_id,
        "chat_history": json.dumps(combined_messages),
        "user_message": user_message,
        "assistant_message": assistant_message,
    }

    response = supabase_client.table("chat_memory").upsert(data).execute()
    if response.data is None:
        raise Exception(f"Erro ao salvar memória: {response.errors}")

async def load_memory(agent_id: str):
    response = supabase_client.table("chat_memory").select("chat_history").eq("agent_id", agent_id).order("created_at", desc=True).limit(1).execute()
    if response.data is None or len(response.data) == 0:
        return []
    return json.loads(response.data[0]["chat_history"])

async def get_prompt(agent_id: str):
    response = supabase_client.table("agents").select("prompt_template").eq("id", agent_id).execute()
    if response.data and len(response.data) > 0:
        return response.data[0]["prompt_template"]
    return "Nenhum prompt adicional foi fornecido."

# Função para limpar a memória no Supabase
async def clear_supabase_memory(agent_id: str):
    try:
        # Deletar registros do chat_memory para o agent_id específico
        response = supabase_client.table("chat_memory").delete().eq("agent_id", agent_id).execute()
        return True
    except Exception as e:
        print(f"Erro ao limpar memória no Supabase: {e}")
        return False

class CustomRetriever(BaseRetriever, BaseModel):
    agent_id: str

    async def _aget_relevant_documents(self, query: str):
        return await custom_retrieval(query, agent_id=self.agent_id)

    def _get_relevant_documents(self, query: str):
        return asyncio.run(self._aget_relevant_documents(query))

llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key=OPENAI_API_KEY)

app = FastAPI(title="API Question and Embbedings QA", description="API para o Assistente de QA com LangChain e Supabase")

# Configuração do middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Alterar para os domínios confiáveis em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdftoken.router, prefix="/api/v1", tags=["PDF Tokens"])
app.include_router(process_docx_token.router, prefix="/api/v1", tags=["DOCX Tokens"])
app.include_router(textotoken.router, prefix="/api/v1", tags=["Texto Tokens"])
app.include_router(txttoken.router, prefix="/api/v1", tags=["TXT Tokens"])
app.include_router(urlsitetoken.router, prefix="/api/v1", tags=["Site Tokens"])
app.include_router(urlyoutubetoken.router, prefix="/api/v1", tags=["YouTube Tokens"])

class QuestionRequest(BaseModel):
    question: str
    agent_id: str
    id_company: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Union[str, float]]]
    chat_history: List[Dict[str, str]]
    tokens_info: Dict[str, Union[Dict[str, float], float]]

class ClearMemoryRequest(BaseModel):
    agent_id: str
    id_company: str

class ClearMemoryResponse(BaseModel):
    message: str
    agent_id: str
    status: str

@app.post("/clear-memory", response_model=ClearMemoryResponse)
async def clear_agent_memory(
    request: ClearMemoryRequest,
    authorization: str = Header(...)
):
    try:
        # Verificar autorização
        if authorization != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Chave de autorização inválida."
            )

        # Verificar se a empresa existe e tem permissão
        sucesso, mensagem = await verifica_token_company(request.id_company)
        if not sucesso:
            raise HTTPException(
                status_code=403,
                detail=mensagem
            )

        # Limpar a sessão em memória
        session_manager.clear_session(request.agent_id, request.id_company)

        # Limpar dados no Supabase
        supabase_cleared = await clear_supabase_memory(request.agent_id)

        if not supabase_cleared:
            raise HTTPException(
                status_code=500,
                detail="Erro ao limpar memória no banco de dados."
            )

        return ClearMemoryResponse(
            message="Memória do agente limpa com sucesso",
            agent_id=request.agent_id,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao limpar memória: {str(e)}")
  

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, authorization: str = Header(...)):
    try:
        if authorization != API_KEY:
            raise HTTPException(status_code=401, detail="Chave de autorização inválida.")
        
        # Verificar tokens
        sucesso, mensagem_token = await verifica_token_company(request.id_company)
        if not sucesso:
            raise HTTPException(status_code=400, detail=mensagem_token)
        
        # Obter ou criar sessão para este usuário/agente
        session = await session_manager.get_session(request.agent_id, request.id_company)
        
        # Adicionar a nova pergunta à memória se ainda não existir
        if not any(isinstance(msg, HumanMessage) and msg.content == request.question 
                  for msg in session.memory.chat_memory):
            session.memory.add_message(request.question, is_human=True)

        # Buscar contexto dinâmico
        dynamic_context = await get_prompt(request.agent_id)

        # Criar mensagem do sistema com o contexto dinâmico
        system_message = SystemMessagePromptTemplate.from_template(
            f"{dynamic_context}"
        )

        # Configurar o prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessagePromptTemplate.from_template(
                "Pergunta: {question}\n\nHistórico da Conversa:\n{chat_history}\n\nContexto:\n{context}"
            )
        ])

        # Configurar retriever e chain
        retriever = CustomRetriever(agent_id=request.agent_id)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=session.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": chat_prompt}
        )

        # Calcular tokens e custos de entrada
        user_query_tokens = count_tokens(request.question, model="gpt-4")
        context_tokens = count_tokens(dynamic_context, model="gpt-4")
        total_tokens_prompt = user_query_tokens + context_tokens
        prompt_cost = calculate_cost(total_tokens_prompt, model="gpt-4")

        # Obter resposta
        response = await asyncio.to_thread(qa_chain.invoke, {"question": request.question})

        # Calcular tokens e custos de saída
        answer_tokens = count_tokens(response["answer"], model="gpt-4")
        completion_cost = calculate_cost(answer_tokens, model="gpt-4")
        total = total_tokens_prompt + answer_tokens
        total_cost = prompt_cost["prompt_cost"] + completion_cost["completion_cost"]

        # Adicionar resposta à memória se ainda não existir
        if not any(isinstance(msg, AIMessage) and msg.content == response["answer"] 
                  for msg in session.memory.chat_memory):
            session.memory.add_message(response["answer"], is_human=False)

        # Preparar fontes
        sources = [
            {
                "id": str(source.metadata.get("id", "")),
                "similarity": f"{source.metadata.get('similarity', 0.0):.2f}"
            }
            for source in response.get("source_documents", [])
        ]

        # Atualizar histórico
        current_messages = [
            {"type": "human", "content": request.question},
            {"type": "ai", "content": response["answer"]}
        ]
        
        # Salvar no Supabase
        await save_memory(request.agent_id, current_messages)

        # Atualizar saldo de tokens
        await atualizar_saldo_tokens(request.id_company, total)

        # Formatar histórico para resposta
        formatted_history = []
        seen_pairs = set()
        current_input = None

        for msg in session.memory.chat_memory:
            if isinstance(msg, HumanMessage):
                current_input = msg.content
            elif isinstance(msg, AIMessage) and current_input is not None:
                pair = (current_input, msg.content)
                if pair not in seen_pairs:
                    formatted_history.append({
                        "input": current_input,
                        "output": msg.content
                    })
                    seen_pairs.add(pair)
                current_input = None

        # Retornar resposta formatada
        return AnswerResponse(
            answer=response["answer"],
            sources=sources,
            chat_history=formatted_history,
            tokens_info={
                "input_tokens": {
                    "tokens_question": user_query_tokens,
                    "tokens_prompt": context_tokens,
                    "total_input_tokens": total_tokens_prompt,
                    "estimative_cost": prompt_cost["prompt_cost"]
                },
                "output_tokens": {
                    "tokens_response": answer_tokens,
                    "estimative_cost": completion_cost["completion_cost"]
                },
                "total_tokens": total,
                "total_cost": total_cost
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {str(e)}")

# Opcional: Função para limpar sessões antigas
async def clear_session_after_timeout(agent_id: str, id_company: str, timeout: int = 3600):
    """Limpa a sessão após um período de inatividade (padrão: 1 hora)"""
    await asyncio.sleep(timeout)
    session_manager.clear_session(agent_id, id_company)