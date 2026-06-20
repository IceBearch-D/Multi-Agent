from zai import ZhipuAiClient
import os
from dotenv import load_dotenv
load_dotenv()
# 读取环境变量
api_key = os.getenv("ZHIPUAN_API_KEY")
# Initialize client
client = ZhipuAiClient(api_key=api_key)

# Create chat completion
response = client.chat.completions.create(
    model="GLM-4.5-air",
    messages=[
        {"role": "user", "content": "你好，请介绍一下自己, Z.ai!"}
    ]
)
print(response.choices[0].message.content)