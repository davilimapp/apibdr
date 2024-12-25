import os
import json
import tiktoken
from typing import List, Dict, Union
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from supabase import create_client
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
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
# Carregar a chave de autorização do ambiente
API_KEY = os.getenv("API_KEY")

# Conectar ao cliente Supabase
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Carregar embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

 # Função para atualizar saldo de tokens no Supabase
async def atualizar_saldo_tokens(idcompany, tokens_utilizados):
     response = supabase_client.from_("tokens").select("tokens_ia").eq("id_company", idcompany).execute()

     if response.data:
         saldo_atual = response.data[0]["tokens_ia"]
         novo_saldo = saldo_atual - tokens_utilizados

         # Atualizar o saldo de tokens
         supabase_client.from_("tokens").update({"tokens_ia": novo_saldo}).eq("id_company", idcompany).execute()
     else:
         print(f"Erro: Nenhuma empresa cadastrada com {idcompany}.")

async def verifica_token_company(idcompany: str):
    """
    Verifica se a empresa possui saldo suficiente de tokens para processar a mensagem.

    Args:
        
        idcompany (str): ID da empresa no Supabase.
        prompt (str): Prompt template a ser utilizado.
        template (str): Template formatado com informações do agente.

    Returns:
        tuple: (bool, str) indicando sucesso/falha e mensagem descritiva.
    """
    try:
        # Verificar se a empresa está registrada
        response = supabase_client.table("companies").select("id").eq("id", idcompany).execute()

        if not response.data:
            return False, "Sua empresa não está registrada. Acesse nossa página de vendas para assinar o app."

        # Consultar saldo de tokens da empresa
        tokens_data = supabase_client.table("tokens").select("tokens_ia").eq("id_company", idcompany).execute()

        # Verificar se a resposta contém dados válidos
        if not tokens_data.data or not tokens_data.data[0].get("tokens_ia"):
            return False, "Erro: Não foi possível recuperar os tokens disponíveis."

        # Obter o saldo de tokens
        token_balance = tokens_data.data[0]["tokens_ia"]
        print(f"Quantidade de Tokens do usuário: {token_balance}")

        margem_saida_tokens = 1200  # Ajuste conforme necessário

        print(f"Quantidade de Tokens necessária: {margem_saida_tokens}")  # Comentar em breve

        # Verificar se o saldo de tokens é suficiente
        if token_balance < margem_saida_tokens:
            return False, f"Você precisa de pelo menos {margem_saida_tokens} tokens para prosseguir. Atualmente, você possui {token_balance} tokens. Acesse o link para adquirir mais."

        return True, "Usuário e tokens verificados com sucesso."

    except Exception as e:
        print(f"Erro ao verificar tokens: {e}")
        return False, "Ocorreu um erro ao verificar seus tokens. Tente novamente mais tarde."

# Função para calcular custo
def calculate_cost(tokens: int, model: str = "gpt-4"):
    """
    Calcula o custo com base no número de tokens e no modelo utilizado.
    """
    if model == "gpt-4":
        # Preços em dólares por 1k tokens
        prompt_price_per_1k = 0.03  # Para entrada
        completion_price_per_1k = 0.06  # Para saída
    elif model == "gpt-3.5":
        prompt_price_per_1k = 0.002
        completion_price_per_1k = 0.002
    else:
        raise ValueError(f"Modelo {model} não suportado para cálculo de custo.")

    # Calcula o custo para entrada e saída
    return {
        "prompt_cost": (tokens / 1000) * prompt_price_per_1k,
        "completion_cost": (tokens / 1000) * completion_price_per_1k,
    }

# Função para contar tokens
def count_tokens(text: str, model: str = "gpt-4"):
    """
    Conta os tokens em um texto para o modelo especificado.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Função personalizada para buscar embeddings usando match_vectors
async def custom_retrieval(query, match_threshold=0.75, match_count=5, agent_id=None):
    # Gera o embedding da consulta de forma assíncrona
    query_embedding = await asyncio.to_thread(embeddings.embed_query, query)
    
    # Executa a chamada RPC de forma síncrona dentro do asyncio.to_thread
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

    # Processa os documentos retornados (usando response.data)
    documents = []
    for result in response.data:  # Acesso direto à propriedade 'data'
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

# Função para salvar histórico no Supabase
async def save_memory(agent_id: str, messages: list):
    """
    Salva o histórico de mensagens no Supabase no formato JSON, incluindo campos separados para mensagens do usuário e do assistente.
    """
    # Recuperar o histórico existente no banco de dados
    existing_data = supabase_client.table("chat_memory").select("chat_history").eq("agent_id", agent_id).execute()
    existing_messages = []
    if existing_data.data:
        existing_messages = json.loads(existing_data.data[0]["chat_history"])

    # Evitar duplicação de mensagens
    new_messages = [msg for msg in messages if msg not in existing_messages]
    combined_messages = existing_messages + new_messages

    # Capturar a última mensagem do usuário e do assistente
    user_message = next((msg["content"] for msg in reversed(combined_messages) if msg["type"] == "human"), "")
    assistant_message = next((msg["content"] for msg in reversed(combined_messages) if msg["type"] == "ai"), "")

    data = {
        "agent_id": agent_id,
        "chat_history": json.dumps(combined_messages),  # Converte as mensagens para JSON
        "user_message": user_message,  # Salva a última mensagem do usuário
        "assistant_message": assistant_message,  # Salva a última mensagem do assistente
    }

    response = supabase_client.table("chat_memory").upsert(data).execute()
    if response.data is None:
        raise Exception(f"Erro ao salvar memória: {response.errors}")

# Função para carregar histórico do Supabase
async def load_memory(agent_id: str):
    """
    Carrega o histórico de mensagens do Supabase e retorna como lista de dicionários.
    """
    response = supabase_client.table("chat_memory").select("chat_history").eq("agent_id", agent_id).order("created_at", desc=True).limit(1).execute()
    if response.data is None or len(response.data) == 0:
        return []  # Retorna histórico vazio se não houver registros
    return json.loads(response.data[0]["chat_history"])  # Converte JSON de volta para lista de mensagens

# Função para buscar dados do Supabase
async def get_prompt(agent_id: str):
    response = supabase_client.table("agents").select("prompt_template").eq("id", agent_id).execute()
    if response.data and len(response.data) > 0:
        return response.data[0]["prompt_template"]
    return "Nenhum prompt adicional foi fornecido."

# Implementação da classe CustomRetriever como um BaseRetriever
class CustomRetriever(BaseRetriever, BaseModel):
    agent_id: str

    async def _aget_relevant_documents(self, query: str):
        return await custom_retrieval(query, agent_id=self.agent_id)

    def _get_relevant_documents(self, query: str):
        return asyncio.run(self._aget_relevant_documents(query))

# Inicializando o modelo de linguagem
llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key=OPENAI_API_KEY)

# Configuração da memória para manter contexto
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# Configuração do FastAPI
app = FastAPI(title="API Question and Embbedings QA", description="API para o Assistente de QA com LangChain e Supabase")

# Configuração do middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Alterar para os domínios confiáveis em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar os routers
app.include_router(pdftoken.router, prefix="/api/v1", tags=["PDF Tokens"])
app.include_router(process_docx_token.router, prefix="/api/v1", tags=["DOCX Tokens"])
app.include_router(textotoken.router, prefix="/api/v1", tags=["Texto Tokens"])
app.include_router(txttoken.router, prefix="/api/v1", tags=["TXT Tokens"])
app.include_router(urlsitetoken.router, prefix="/api/v1", tags=["Site Tokens"])
app.include_router(urlyoutubetoken.router, prefix="/api/v1", tags=["YouTube Tokens"])

# Modelo de entrada
class QuestionRequest(BaseModel):
    question: str
    agent_id: str
    id_company: str

# Modelo de resposta
class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Union[str, float]]]
    chat_history: List[Dict[str, str]]
    tokens_info: Dict[str, Union[Dict[str, float], float]]

# Endpoint principal   
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, authorization: str = Header(...)):
    try:
         # Verificar se a chave foi fornecida
        if authorization != API_KEY:
            raise HTTPException(status_code=401, detail="Chave de autorização inválida.")
        
        await verifica_token_company(request.id_company)

        # Consultas quantidade de tokens da empresa
        sucesso, mensagem_token = await verifica_token_company(request.id_company)

        if not sucesso:
            raise HTTPException(status_code=400, detail=mensagem_token)
        
        # Se tokens suficientes, continuar para criar o chat
        if sucesso:
            # Carregar histórico do Supabase
            previous_messages = await load_memory(request.agent_id)

            # Adicionar mensagens existentes no Supabase ao ConversationBufferMemory
            for msg in previous_messages:
                if not any(
                    existing_msg.content == msg["content"] and existing_msg.type == msg["type"]
                    for existing_msg in memory.chat_memory.messages
                ):
                    if msg["type"] == "human":
                        memory.chat_memory.add_user_message(msg["content"])
                    elif msg["type"] == "ai":
                        memory.chat_memory.add_ai_message(msg["content"])

            # Adicionar a pergunta atual do usuário à memória (apenas se ainda não estiver adicionada)
            if not any(
                existing_msg.content == request.question and existing_msg.type == "human"
                for existing_msg in memory.chat_memory.messages
            ):
                memory.chat_memory.add_user_message(request.question)

            # Buscar contexto dinâmico do banco de dados
            dynamic_context = await get_prompt(request.agent_id)

            # Criar mensagem do sistema com o valor dinâmico
            system_message = SystemMessagePromptTemplate.from_template(
                f"{dynamic_context}"
            )

            # Criar o ChatPromptTemplate usando o contexto dinâmico
            chat_prompt = ChatPromptTemplate.from_messages([
                system_message,
                HumanMessagePromptTemplate.from_template(
                    "Pergunta: {question}\\n\\nHistórico da Conversa:\\n{chat_history}\\n\\nContexto:\\n{context}"
                )
            ])

            # Configurando retriever com agent_id recebido
            retriever = CustomRetriever(agent_id=request.agent_id)
            
            # Configurando ConversationalRetrievalChain com retriever dinâmico
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": chat_prompt}
            )

            # Exibir contagem de tokens
            user_query_tokens = count_tokens(request.question, model="gpt-4")
            context_tokens = count_tokens(dynamic_context, model="gpt-4")
            total_tokens_prompt = user_query_tokens + context_tokens

            # Calcula custo para o prompt
            prompt_cost = calculate_cost(total_tokens_prompt, model="gpt-4")

            # Obter resposta da cadeia
            response = await asyncio.to_thread(qa_chain.invoke, {"question": request.question})

            
            # Contar tokens da resposta
            answer_tokens = count_tokens(response["answer"], model="gpt-4")
            

            # Calcula custo para a resposta
            completion_cost = calculate_cost(answer_tokens, model="gpt-4")
            
            total = total_tokens_prompt + answer_tokens
            # Total de custos
            total_cost = prompt_cost["prompt_cost"] + completion_cost["completion_cost"]
            

            # Adicionar a resposta da IA na memória (apenas se ainda não estiver adicionada)
            if not any(
                existing_msg.content == response["answer"] and existing_msg.type == "ai"
                for existing_msg in memory.chat_memory.messages
            ):
                memory.chat_memory.add_ai_message(response["answer"])

            # Fontes utilizadas
            sources = [
                {
                    "id": str(source.metadata.get("id", "")),
                    "similarity": f"{source.metadata.get('similarity', 0.0):.2f}"
                }
                for source in response.get("source_documents", [])
            ]

            # Histórico atualizado com a pergunta e a resposta atuais
            updated_messages = previous_messages + [
                {"type": "human", "content": request.question},
                {"type": "ai", "content": response["answer"]}
            ]

            # Salvar o histórico atualizado no Supabase
            await save_memory(request.agent_id, updated_messages)

            await atualizar_saldo_tokens(request.id_company, total)

            # Formatar o histórico diretamente da memória para evitar duplicação
            formatted_history = []
            seen_pairs = set()  # Para evitar duplicatas
            for message_pair in memory.chat_memory.messages:
                if message_pair.type == "human":
                    input_message = message_pair.content
                elif message_pair.type == "ai":
                    output_message = message_pair.content
                    # Verificar se o par já foi adicionado
                    pair = (input_message, output_message)
                    if pair not in seen_pairs:
                        formatted_history.append({"input": input_message, "output": output_message})
                        seen_pairs.add(pair)

            return AnswerResponse(
                answer=response["answer"],  # Resposta gerada pelo modelo
                sources=sources,  # Fontes utilizadas na resposta
                chat_history=formatted_history,  # Histórico de conversa formatado
                tokens_info={
                    "input_tokens": {
                        "tokens_question": user_query_tokens,
                        "tokens_prompt": context_tokens,  # Tokens utilizados na entrada
                        "total_input_tokens": total_tokens_prompt,
                        "estimative_cost": prompt_cost["prompt_cost"]  # Custo estimado dos tokens de entrada
                    },
                    "output_tokens": {
                        "tokens_response": answer_tokens,  # Tokens utilizados na saída
                        "estimative_cost": completion_cost["completion_cost"]  # Custo estimado dos tokens de saída
                    },
                    "total_tokens": total,
                    "total_cost": total_cost  # Custo total estimado (entrada + saída)
                }
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {str(e)}")

