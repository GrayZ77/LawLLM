import json
from io import StringIO
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


st.set_page_config(page_title="FudanDISC-LawLLM")
st.title("FudanDISC-LawLLM🤖️")

@st.cache_resource
def init_model():
    model_path = "/root/DISC-LawLLM/data"
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


def init_chat_history():
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("您好，我是复旦 DISC-LawLLM，很高兴为您服务💖")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🙋‍♂️" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    model, tokenizer = init_model()
    st.text("上传文件进行内容总结：")

    uploaded_file = st.file_uploader("Choose a file")
    messages = init_chat_history()
    
    if uploaded_file is not None:
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        string_data = "请对以下内容中涉及的法律知识进行总结括：\n" + string_data
        messages.append({"role": "user", "content": string_data})
        print(f"[user] {string_data}", flush=True)
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)
        st.button("清空对话", on_click=clear_chat_history)
        
    if prompt := st.chat_input("Shift + Enter 换行，Enter 发送"):
        with st.chat_message("user", avatar="🙋‍♂️"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
