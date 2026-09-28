from langchain_openai import ChatOpenAI

MODEL = ChatOpenAI(
    model="qwen2.5:3b",
    temperature=0,
    base_url="http://127.0.0.1:8080/v1",
    api_key="dummy"
)

if __name__ == "__main__":
    result = MODEL.invoke(input="how are you?")
    print(result)
