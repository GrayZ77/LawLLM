# LawLLM-With-Langchain

## 本地部署方法

首先将本仓库clone至本地

```bash
git clone https://github.com/GrayZ77/LawLLM.git
```

然后安装推理

```bash
pip install -r requirements.txt
```

随后，将模型文件下载到`LawLLM`目录下，并将模型文件目录重命名为`model`

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/ShengbinYue/DISC-LawLLM
```

当然，你也可以使用`transformer`在运行时自动下载，因此需要将`web_ui.py`文件中的以下代码

```python
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
```

修改为

```python
@st.cache_resource()
def init_model():
    model_path = "ShengbinYue/DISC-LawLLM"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, offload_folder = "offload"
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    return model, tokenizer
```

随后运行`web_ui.py`即可

```bash
streamlit run web_ui.py --server.port 8888
```



## 创建本地向量数据库

如果想要在应用中开启`法条检索`功能，首先应当在项目根目录下创建文件夹，文件夹应命名为`法律文书`，或者使用自定义的名称，并修改`match.py`中对应的名称

随后，运行`web_ui.py`，并在页面中勾选`开启法条检索`，输入问题并提交后，系统会自动创建向量数据库，并保存在项目根目录下名为`VectorDataBase`的文件夹下。
