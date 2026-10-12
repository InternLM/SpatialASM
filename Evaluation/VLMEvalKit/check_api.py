# from vlmeval.api import OpenAIWrapper
# model = OpenAIWrapper('gpt-4o', verbose=True)
# msgs = [dict(type='text', value='Hello!')]
# code, answer, resp = model.generate_inner(msgs)
# print(code, answer, resp)

import os
from vlmeval.api import GPT4V

model = GPT4V(
    model="kimi-k3",
    api_base="http://token.pjlab.org.cn/v1/chat/completions",
    key="sk-6737b8d5df4945742a5b6c4133d1857b",
    temperature=0,
    retry=1,
    timeout=60,
    verbose=True,
)

ret_code, answer, log = model.generate_inner([
    {"type": "text", "value": "你好，请回复测试成功"}
])

print("ret_code:", ret_code)
print("answer:", answer)
print("log:", log)