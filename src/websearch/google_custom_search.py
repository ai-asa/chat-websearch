# %%

from dotenv import load_dotenv
import os
import json
from time import sleep
from googleapiclient.discovery import build

# ここに取得したAPIキーと検索エンジンIDを設定

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

def get_search_response(keyword, max_results=10, custom_search_engine_id=GOOGLE_CSE_ID):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    responses = []
    
    try:
        result = service.cse().list(
            q=keyword,
            cx=custom_search_engine_id,
            lr='lang_ja',
            num=max_results,# 1リクエストで10件取得可能
        ).execute()
        responses.append(result)
    except Exception as e:
        print("Error:", e)
    return responses

def main():
    target_keyword = "NYダウ　平均株価"
    api_response = get_search_response(target_keyword)
    print(api_response)

if __name__ == '__main__':
    main()

# %%
