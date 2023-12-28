import json
import match
from io import StringIO
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from streamlit_extras.mention import mention


st.set_page_config(page_title="DISC-LawLLM")
st.title("🦜LawLLM-With-LangChain")
st.caption("🚀 A streamlit chatbot powered by FudanDISC-LLM, running on Ubuntu-22.04")

"""
该问答系统以`LangChain`为基本框架，完成了向量数据库的构建与文本检索。更多有关`LangChain`的内容请访问[langchain-io.com](https://www.langchain-io.com/)
"""

@st.cache_resource()
def init_model():
    model_path = "/root/DISC-LawLLM/model"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, local_files_only = True, offload_folder = "offload"
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path, local_files_only = True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True, local_files_only = True
    )
    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages
    del st.session_state.dialogs


def init_chat_history():
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("您好，我是 DISC-LawLLM，很高兴为您服务💖")

   

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "dialogs" not in st.session_state:
        st.session_state.dialogs = []

    else: 
        for message in st.session_state.dialogs:
            avatar = "🙋‍♂️" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "result" in message:
                    with st.expander("查看检索结果", expanded=False):
                        st.markdown(message["result"])
                

    return st.session_state.messages, st.session_state.dialogs


def main():
    model, tokenizer = init_model()
    messages, dialogs = init_chat_history()
    with st.sidebar:
        mention(
            label = "Source Code",
            icon = "github",
            url = "https://github.com/GrayZ77/LawLLM",
        )
    
    
    _stream = st.sidebar.checkbox("开启流式输出", value=True)
    _match = st.sidebar.checkbox("开启法条检索")
           

    if _match:
        _num = st.sidebar.slider("请选择检索法条数", 1, 5, 3)
        if prompt := st.chat_input("Shift + Enter 换行，Enter 发送"):
            with st.chat_message("user", avatar="🙋‍♂️"):
                st.markdown(prompt)
            result = match.quest(prompt, num=_num)
            question = f"以自己的思考为主，参考给出的内容，回答下面的问题。如果给出的内容没有参考价值，就忽略掉这些内容。\n\n参考：{result}\n\n问题：{prompt}"
            dialogs.append({"role": "user", "content": prompt})
            messages.append({"role": "user", "content": question})
            print(f"[user] {question}", flush=True)
            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                with st.expander("查看检索结果", expanded=False):
                    st.markdown(result)
                for response in model.chat(tokenizer, messages, stream=_stream):
                    placeholder.markdown(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            messages.append({"role": "assistant", "content": response})
            dialogs.append({"role": "assistant", "content": response, "result": result})
            print(json.dumps(messages, ensure_ascii=False), flush=True)
    
    else:
        if prompt := st.chat_input("Shift + Enter 换行，Enter 发送"):
            with st.chat_message("user", avatar="🙋‍♂️"):
                st.markdown(prompt)
            dialogs.append({"role": "user", "content": prompt})
            messages.append({"role": "user", "content": prompt})
            print(f"[user] {prompt}", flush=True)
            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                for response in model.chat(tokenizer, messages, stream=_stream):
                    placeholder.markdown(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            messages.append({"role": "assistant", "content": response})
            dialogs.append({"role": "assistant", "content": response})
            print(json.dumps(messages, ensure_ascii=False), flush=True)

    st.button("清空对话", on_click=clear_chat_history)



if __name__ == "__main__":
    main()
