import os
import requests
from dotenv import load_dotenv

class BingWebSearch:
    BASE_URL = "https://api.bing.microsoft.com/v7.0/search"

    def __init__(self, api_key=None):
        load_dotenv()
        self.api_key = api_key or os.getenv("BING_API_KEY")

        if not self.api_key:
            raise ValueError("Bing API key is required")

    def search(self, query, **params):
        """
        Bing Web Search APIを使用して検索を実行します
        
        Args:
            query (str): 検索クエリ
            **params: その他の検索パラメータ（mkt, count等）
            
        Returns:
            dict: 検索結果
        """
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }

        search_params = {
            "q": query,
            **params
        }

        response = requests.get(self.BASE_URL, headers=headers, params=search_params)
        response.raise_for_status()
        return response.json() 