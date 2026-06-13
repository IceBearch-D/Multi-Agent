import os
from dotenv import load_dotenv

# 加载 .env 文件中的变量
load_dotenv()

# 读取 Key
openai_key = os.environ.get("OPENAI_API_KEY")
deepseek_key = os.environ.get("DEEPSEEK_API_KEY")

# 验证是否读取成功
if not openai_key:
    raise ValueError("请在 .env 文件中配置 OPENAI_API_KEY")

print(f"OpenAI Key: {openai_key[:10]}...")    # 只打印前几位，避免泄露
print(f"DeepSeek Key: {deepseek_key[:10] if deepseek_key else '未配置'}...")

# 实际使用
from openai import OpenAI

client = OpenAI(api_key=openai_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "用一句话介绍你自己"}]
)

print(f"\n回复: {response.choices[0].message.content}")
