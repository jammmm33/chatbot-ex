import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (ChatPromptTemplate, FewShotPromptTemplate,
                                    MessagesPlaceholder, PromptTemplate)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import answer_examples


## 환경변수 읽어오기 ======================================================
load_dotenv()

## LLM 생성 ===============================================================
def load_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)

## Embedding 설정 + Vector Store Index 가져오기 ===========================
def load_vectorstore():

    ## 임베딩 모델 지정
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    index_name= 'quiz04-law1-2'
    embedding=OpenAIEmbeddings(model="text-embedding-3-large")

    ## 저장된 인덱스 가져오기
    database = PineconeVectorStore.from_existing_index(
        index_name= index_name,
        embedding=embedding,
    )
    return database

### 세션별 히스토리 저장 ==================================================
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

## 히스토리 기반 리트리버 =================================================
def build_history_aware_retriever(llm, retriever):
    contextualize_q_system_prompt = (
    '''채팅 히스토리와 가장 최근 사용자의 질문이 주어졌을 때,  
    그 질문이 이전 대화의 문맥을 참고할 수 있다는 점을 고려하여,  
    이전 히스토리 없이도 이해 가능한 독립적인 질문으로 바꿔주세요.  
    질문에 답변하지는 마세요.  
    필요하다면 문장을 재작성하고, 그렇지 않으면 그대로 반환하세요.'''
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever


def build_few_shot_examples() -> str:
    example_prompt = PromptTemplate.from_template("Question: {input}\n\nAnswer: {answer}")

    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_examples,         ## 질문/답변 예시들 (전체 type은 list, 각 질문/답변 type은 dict)
        example_prompt=example_prompt,    ## 단일 예시 포맷
        prefix='다음 질문에 답변하세요.', ## 예시들 위로 축되는 텍스트(도입부)
        suffix="Question: {input}",       ## 예시들 뒤로 축되는 텍스트(실제사용자 질문 변수)
        input_variables=["input"],        ## suffix에서 사용할 변수
    )

    return few_shot_prompt.format(input='{input}')
    


def build_qa_prompt():
    ### [keyword dictionary] =================================================
    ## 1. 기본 형태 (가장 일반적인 형태)
    ## * 장점 : 키 하나당 설명 하나, 단순+빠름
    ## * 용도 (실무 활용 예): FAQ 챗봇, 버튼식 응답
    # keyword_dictionary = {
    #     '임대인': '임대인은 주택을 임차인에게 제공하고, 계약 종료 시 보증금을 반환할 의무가 있는 자 입니다.',
    #     '주택' : '주택”이란 「주택임대차보호법」 제2조에 따른 주거용 건물(공부상 주거용 건물이 아니라도 임대차계약 체결 당시 임대차목적물의 구조와 실질이 주거용 건물이고 임차인의 실제 용도가 주거용인 경우를 포함한다)을 말한다.',
    # }

    ## 2. 질문형 키워드 (질문 다양성 대응)
    ## 장점: 유사 질문을 여러 키로 분기하여 모두 같은 대답으로 연결, fallback 대응(예시수단, 대체안)
    ## 용도(실무사용 예): 단답 챗봇, 키워드 FAQ 챗봇
    # keyword_dictionary = {
    #     '임대인 일려줘': '🍔임대인은 주택을 임차인에게 제공하고, 계약 종료 시 보증금을 반환할 의무가 있는 자 입니다.',
    #     '주택 알려줘' : '🍕주택”이란 「주택임대차보호법」 제2조에 따른 주거용 건물(공부상 주거용 건물이 아니라도 임대차계약 체결 당시 임대차목적물의 구조와 실질이 주거용 건물이고 임차인의 실제 용도가 주거용인 경우를 포함한다)을 말한다.',
    #     '임대인': '🍔임대인은 주택을 임차인에게 제공하고, 계약 종료 시 보증금을 반환할 의무가 있는 자 입니다.',
    #     '주택' : '🍕주택”이란 「주택임대차보호법」 제2조에 따른 주거용 건물(공부상 주거용 건물이 아니라도 임대차계약 체결 당시 임대차목적물의 구조와 실질이 주거용 건물이고 임차인의 실제 용도가 주거용인 경우를 포함한다)을 말한다.',
    # }

    ## 3. 키워드 + 태그 기반 딕셔너리
    keyword_dictionary = {
        '임대인' : {
            'definition' : '전세사기피해자법 제2조 제2항에 따른 임대인 정의입니다.',
            'source': '전세사기피해자법 제 2조 2항',
            'tag' : ['용어', '법률', '기초'],
        },
        '주택' : {
            'definition' :'주택”이란 「주택임대차보호법」 제2조에 따른 주거용 건물(공부상 주거용 건물이 아니라도 임대차계약 체결 당시 임대차목적물의 구조와 실질이 주거용 건물이고 임차인의 실제 용도가 주거용인 경우를 포함한다)을 말한다.',
            'source': '전세사기피해자법 제 2조 1항',
            'tag' : ['용어', '법률', '기초']
        },
    }

    dictionary_text = '\n'.join([
        f'{k} {v["tag"]}: {v["definition"]} [출처: {v["source"]}]' 
        for k, v in keyword_dictionary.items()
    ])


    system_prompt = (
    '''
    -당신은 전세사기피해 법률 전문가 입니다. 
    -[context]와 [keyword_dictionary]를 참고하여 사용자의 질문에 답변하세요.
    -답변에는 해당 조항을'[XX법 시행령 제X조]'형식으로 문단 메시지 표시하세요.
    -전세사기피해 법률에 대한 정보 이외에는 '답변을 할 수 없습니다.' 로 답하세요.
    -필요하다면 문서에서 직접 인용하거나 요약된 조항을 덧붙이세요.
    -사용자가 '알려주세요', '설명해주세요'  질문에도 답변하세요.
    -항목별로 표시해서 답변해주세요.
    
    [context]
    {context} 

    [keyword_dictionary]
    {dictionary_text}
    '''
    )
    formmated_few_shot_prompt = build_few_shot_examples()

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ('assistant', formmated_few_shot_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    ).partial(dictionary_text=dictionary_text)

    print(f'\nqa_prompt >>\n{qa_prompt.partial_variables}')

    return qa_prompt

## 전체 chain 구성 ======================================================
def build_conversational_chain():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    ## LLM 모델 지정
    llm= load_llm()

    ## vector store에서 index 정보
    database = load_vectorstore()
    retriever = database.as_retriever()
    # (search_kwargs={"k": 2})

    history_aware_retriever = build_history_aware_retriever(llm, retriever)
    qa_prompt = build_qa_prompt()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        ).pick('answer')

    return conversational_rag_chain


## AI Message ===========================================================
def stream_ai_message(user_message, session_id = 'default'):
    qa_chain = build_conversational_chain()

    ai_message = qa_chain.stream(
        {'input': user_message},
        config={"configurable": {"session_id": session_id}}
    )
    print(f'대화이력 >> {get_session_history(session_id)} \n😎\n')
    print('='*50 + '\n')
    print(f'[stream_ai_message 함수 내 출력] session_id >> {session_id}')

    ##########################################################################
    ## vector store에서 검색된 문서 확인
    retriever = load_vectorstore().as_retriever(search_kwargs = {'k': 2})
    search_result = retriever.invoke(user_message)

    print(f'\nPinecone 검색 결과 >> \n{search_result[0].page_content[:100]}')
    ##########################################################################

    return ai_message