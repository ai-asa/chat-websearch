# %%
from duckduckgo_search import DDGS

class DuckDuckGoInstantAnswer:
    def search(self, query, search_type="text", region="jp-jp", safesearch="off", timelimit=None, max_results=4):
        """
        duckduckgo-searchライブラリを使用して検索を実行します。
        
        Args:
            query (str): 検索クエリ
            search_type (str): 検索の種類。利用可能な値は以下の通り
                               "text", "images", "news", "videos"
            region (str): 地域設定 (例: "jp-jp")
            safesearch (str): セーフサーチ設定 ("off", "on", "moderate")
            timelimit (str or None): 期間指定 (例: None または "YYYY-MM-DD..YYYY-MM-DD")
            max_results (int): 取得件数
            
        Returns:
            list: 検索結果（各要素は dict）
        """
        with DDGS() as ddgs:
            search_functions = {
                "text": ddgs.text,
                "images": ddgs.images,
                "news": ddgs.news,
                "videos": ddgs.videos,
            }
            
            if search_type not in search_functions:
                raise ValueError("Invalid search_type. Choose from: " + ", ".join(search_functions.keys()))
            
            results = list(search_functions[search_type](
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results
            ))
        
        return results
    
if __name__ == "__main__":
    ddg = DuckDuckGoInstantAnswer()
    print(ddg.search("NYダウ　平均株価"))

# %%
