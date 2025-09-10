import streamlit as st
import os
from dotenv import load_dotenv

# API 및 모델 관련 라이브러리
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.google_search.tool import GoogleSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from PIL import Image

# .env 파일에서 환경 변수 로드
load_dotenv()

# Streamlit 앱 구성
st.set_page_config(
    page_title="AI API 플랫폼",
    page_icon="🤖"
)
st.title("AI API 플랫폼 🤖")
st.markdown("다양한 AI 에이전트와 대화하고, 웹 검색 및 이미지 생성 기능을 사용해 보세요.")

# 사이드바에 API 키 입력 필드 및 링크 추가
with st.sidebar:
    st.header("API 설정")
    st.markdown("API 키를 입력하거나 .env 파일에 저장하세요.")
    openai_api_key = st.text_input("OpenAI API 키", type="password")
    google_api_key = st.text_input("Google API 키", type="password")
    st.markdown("---")
    st.header("다른 AI 플랫폼 🌐")
    st.markdown("[ChatGPT 웹](https://chat.openai.com)")
    st.markdown("[Gemini 웹](https://gemini.google.com)")

# .env 파일에 키가 있으면 사용
if os.getenv("OPENAI_API_KEY") and not openai_api_key:
    openai_api_key = os.getenv("OPENAI_API_KEY")
if os.getenv("GOOGLE_API_KEY") and not google_api_key:
    google_api_key = os.getenv("GOOGLE_API_KEY")

# 모델 선택
model_choice = st.selectbox(
    "모델을 선택하세요:",
    ["OpenAI GPT-4o", "OpenAI GPT-3.5-turbo"]
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = {}
if model_choice not in st.session_state.messages:
    st.session_state.messages[model_choice] = []

# 모델별로 다른 대화 기록 표시
for message in st.session_state.messages[model_choice]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message["type"] == "image":
            st.image(message["content"])
        else:
            st.markdown(message["content"])

# 프롬프트 입력 및 처리
if prompt := st.chat_input("무엇을 도와드릴까요?"):
    st.session_state.messages[model_choice].append({"role": "user", "content": prompt, "type": "text"})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 이미지 생성 명령 처리 (예: "이미지 생성: 귀여운 고양이")
    if prompt.startswith("이미지 생성:"):
        image_prompt = prompt.replace("이미지 생성:", "").strip()
        with st.chat_message("assistant"):
            st.markdown("이미지를 생성 중입니다... 🎨")
            try:
                if not openai_api_key:
                    st.error("OpenAI API 키가 필요합니다. 사이드바에 입력해주세요.")
                else:
                    # DALL-E API 호출 (예시)
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_api_key)
                    response = client.images.generate(
                        prompt=image_prompt,
                        n=1,
                        size="512x512"
                    )
                    image_url = response.data[0].url
                    st.image(image_url)
                    st.session_state.messages[model_choice].append({"role": "assistant", "content": image_url, "type": "image"})
            except Exception as e:
                st.error(f"이미지 생성에 실패했습니다: {e}")
        st.experimental_rerun()
    else:
        # 챗봇 에이전트 응답 처리
        with st.chat_message("assistant"):
            if not openai_api_key:
                st.error("OpenAI API 키가 필요합니다. 사이드바에 입력해주세요.")
            else:
                try:
                    # 에이전트 설정
                    llm = ChatOpenAI(model=model_choice.split()[-1], temperature=0, api_key=openai_api_key)
                    
                    tools = []
                    # 웹 검색 기능 추가
                    if google_api_key:
                        os.environ["GOOGLE_API_KEY"] = google_api_key
                        search_tool = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())
                        tools.append(search_tool)

                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "당신은 유능한 AI 비서입니다. 사용자의 질문에 답변하세요."),
                        MessagesPlaceholder("chat_history", optional=True),
                        ("human", "{input}"),
                        MessagesPlaceholder("agent_scratchpad"),
                    ])

                    agent = create_tool_calling_agent(llm, tools, prompt_template)
                    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                    # 대화 기록 형식 변환
                    langchain_history = []
                    for msg in st.session_state.messages[model_choice]:
                        if msg["role"] == "user":
                            langchain_history.append(("human", msg["content"]))
                        else:
                            langchain_history.append(("assistant", msg["content"]))
                            
                    response_stream = agent_executor.stream({"input": prompt, "chat_history": langchain_history})
                    response = ""
                    for chunk in response_stream:
                        if "output" in chunk:
                            response += chunk["output"]
                            st.write(chunk["output"])
                    
                    st.session_state.messages[model_choice].append({"role": "assistant", "content": response, "type": "text"})
                except Exception as e:
                    st.error(f"에러가 발생했습니다: {e}")
        st.experimental_rerun()