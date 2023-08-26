from langchain.llms import LlamaCpp
from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain.agents import load_tools
import os

openai_api_key = os.environ.get('OPENAI_API_KEY') 

# Added a paramater for GPU layer numbers
n_gpu_layers = os.environ.get('N_GPU_LAYERS') 

# Added custom directory path for CUDA dynamic library 
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/extras/CUPTI/lib64")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include")
os.add_dll_directory("C:/tools/cuda/bin")


llm = LlamaCpp(temperature=0, model_path='models\llama-2-7b-chat.ggmlv3.q4_0.bin', n_ctx=1000, n_batch=8,  verbose=False, n_gpu_layers=n_gpu_layers)
# llm =  OpenAI(model_name="text-davinci-003",  temperature=0, openai_api_key=openai_api_key)

tools = load_tools(
    ['llm-math'],
    llm=llm
)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
)

agent.run("what is (4.5*2.1)^2.2?")