import configparser
from dotenv import load_dotenv
from openai import OpenAI
import os

class OpenaiAdapter:

    load_dotenv()
    config = configparser.ConfigParser()
    config.read('config.ini')
    retry_limit = int(config.get('CONFIG', 'retry_limit', fallback=5))

    def __init__(self):
        self.client = OpenAI(
            api_key = os.getenv('OPENAI_API_KEY')
        )
    
    def openai_chat(self, openai_model, prompt, temperature=1):
        system_prompt = [{"role": "system", "content": prompt}]
        for i in range(self.retry_limit):
            try:
                response = self.client.chat.completions.create(
                    messages=system_prompt,
                    model=openai_model,
                    temperature=temperature
                )
                text = response.choices[0].message.content
                return text
            except Exception as error:
                print(f"GPT呼び出し時にエラーが発生しました:{error}")
                if i == self.retry_limit - 1:
                    return None  # エラー時はNoneを返す
                continue