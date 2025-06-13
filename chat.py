import streamlit as st
from llm import stream_ai_message


st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='🤖' ) 
st.title('전세사기피해 상담 챗봇👀')

print('\n\n==start==')
print('before) st.session_state >>', st.session_state)

##############################################################
import uuid

# print('UUID >>', uuid.uuid4())
## 타입 --> class --> str 저장

## 세션 ID에 고유한 값 설정
## [방법 1] 새로고침(F5) 하면 새로 발금
# if 'session_id' not in st.session_state:
#     ## 세션 ID 저장
#     # st.session_state['session_id'] = str(uuid.uuid4())
#     st.session_state.session_id = str(uuid.uuid4())

#     # 출력
#     print("st.session_state.session_id >>", st.session_state.session_id)

## [방법 2] URL의 Parameter에 저장
# query_params = st.query_params
# print('query_params >>', st.query_params)

# st.query_params.update({'age': 39})

## Query parameter에
## session_id가 있으면, 값을 가져오고
## session_id가 없으면, 파라미터 설정
## [키와 값]
## ket : session_id
## value : UUID

## Query parameter
print('st.query_params >>', st.query_params)
print('session_id 값 추출 1 >>', st.query_params.session_id)
print('session_id 값 추출 2 >>', st.query_params['session_id'])

query_params = st.query_params

if 'session_id' in query_params:
    session_id = query_params['session_id']
    print('URLdo session_id가 있다면, UUID를 가져와서 변수 저장')
else:
    session_id = str(uuid.uuid4())
    st.query_params.update({'session_id': session_id})
    print('URL에 session_id가 없다면, UUID를 생성하여 추가')

print('after) st.session_state >>', st.session_state)
##############################################################

## streamlit 내부 세션: session id 저장
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = session_id
    print('[streamlit 내부 세션] st.session_state.session_id >>', st.session_state.session_id)

## streamlit 내부 세션: 메시지 리스트 초기화
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

## 이전 채팅 내용 화면 출력
for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])


## 채팅 메시지 =======================================================================
placeholder = '전세사기피해와 관련된 질문을 작성해 주세요.'
if user_question := st.chat_input(placeholder=placeholder): ## prompt 창
    ## 사용자 메시지
    with st.chat_message('user'):
        ## 사용자 메시지 화면 출력
        st.write(user_question)
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

    ## AI 메시지
    with st.spinner('답변을 생성하는 중입니다.'):
        # ai_message = get_ai_message(user_question)

        session_id = st.session_state.session_id
        ai_message = stream_ai_message(user_question, session_id=session_id)

        with st.chat_message('ai'):
            ## AI 메시지 화면 출력
            ai_message = st.write_stream(ai_message)
        st.session_state.message_list.append({'role': 'ai', 'content': ai_message})

# print({f'after: {st.session_state.message_list}'})