"""示例：调用智谱 AI（ZhipuAI）客户端进行对话测试。

用来演示项目所用的大模型 SDK（zai 包 / GLM-4.5-air 模型）的基本用法。
需要先在 .env 或环境中配置 ZHIPUAN_API_KEY。
"""
from zai import ZhipuAiClient  # 智谱 AI 官方 Python SDK 客户端
import os  # 读取环境变量
from dotenv import load_dotenv  # 从 .env 加载环境变量

load_dotenv()  # 加载项目根目录的 .env 文件
# 读取环境变量（智谱开放平台的 API Key）
api_key = os.getenv("ZHIPUAN_API_KEY")
# 初始化客户端
client = ZhipuAiClient(api_key=api_key)

# 创建聊天补全请求：指定模型与用户消息
response = client.chat.completions.create(
    model="GLM-4.5-air",
    messages=[
        {"role": "user", "content": "你好，请介绍一下自己, Z.ai!"}
    ]
)
# 打印模型返回的第一条消息内容
print(response.choices[0].message.content)