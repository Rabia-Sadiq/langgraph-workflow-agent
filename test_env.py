import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
model = os.getenv("MODEL_NAME")
print("Key:", key)
print("Model:", model)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=key,
)

response = llm.invoke("say hello")
print(response.content)