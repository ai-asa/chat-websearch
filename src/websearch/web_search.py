from src.webscraping.web_scraping import WebScraper

class WebSearch:
    """
    複数のWeb検索APIのラッパーを一元管理するクラス。
    各検索エンジンのAPIを統一したインターフェースで利用できます。
    """
    
    def __init__(self, default_engine="google"):
        """
        WebSearchクラスの初期化
        
        Args:
            default_engine (str): デフォルトで使用する検索エンジン
                                 "google", "bing", "duckduckgo"のいずれか
        """
        self.engines = {}
        self.default_engine = default_engine
        self._initialize_engines()
        self.scraper = WebScraper()
        
        # デフォルトエンジンが利用できない場合は、利用可能な最初のエンジンをデフォルトに設定
        if self.default_engine not in self.engines and self.engines:
            self.default_engine = next(iter(self.engines.keys()))
    
    def _initialize_engines(self):
        """利用可能な検索エンジンの初期化"""
        # Google Custom Search API
        from src.websearch.google_custom_search import get_search_response
        self.engines["google"] = {
            "instance": None,
            "search_func": get_search_response
        }
        
        # Bing Web Search API
        from src.websearch.bing_web_search import BingWebSearch
        bing_search = BingWebSearch()
        self.engines["bing"] = {
            "instance": bing_search,
            "search_func": bing_search.search
        }
        
        # DuckDuckGo Instant Answer API
        from src.websearch.duckduckgo_instant_answer import DuckDuckGoInstantAnswer
        ddg_search = DuckDuckGoInstantAnswer()
        self.engines["duckduckgo"] = {
            "instance": ddg_search,
            "search_func": ddg_search.search
        }
    
    def available_engines(self):
        """利用可能な検索エンジンのリストを返す"""
        return list(self.engines.keys())
    
    def search(self, query, engine=None, max_results=4, **kwargs):
        """
        指定された検索エンジンを使用して検索を実行
        
        Args:
            query (str): 検索クエリ
            engine (str, optional): 使用する検索エンジン。指定がない場合はデフォルトエンジンを使用
            **kwargs: 各検索エンジン固有のパラメータ
        
        Returns:
            dict or list: 検索結果（エンジンによって形式が異なる）
        
        Raises:
            ValueError: 指定されたエンジンが利用できない場合
        """
        engine = engine or self.default_engine
        
        if not self.engines:
            raise RuntimeError(f"利用可能な検索エンジンがありません。")
        
        if engine not in self.engines:
            available = ", ".join(self.available_engines())
            error_msg = f"指定されたエンジン '{engine}' は利用できません。"
            
            if available:
                error_msg += f"\n利用可能なエンジン: {available}"
            else:
                error_msg += "\n利用可能なエンジンはありません。"
                
            raise ValueError(error_msg)
        
        engine_data = self.engines[engine]
        
        if engine == "google":
            # Google検索の場合、custom_search_engine_idを渡す
            custom_search_engine_id = kwargs.pop("custom_search_engine_id", None)
            return engine_data["search_func"](query, max_results=max_results, custom_search_engine_id=custom_search_engine_id, **kwargs)
        elif engine == "bing":
            return engine_data["search_func"](query, max_results=max_results, **kwargs)
        elif engine == "duckduckgo":
            return engine_data["search_func"](query, max_results=max_results, **kwargs)
    
    def process_results(self, results, engine=None):
        """
        検索結果を処理して標準化された形式で返す
        
        Args:
            results: search()メソッドから返された検索結果
            engine (str, optional): 結果を処理する検索エンジン。指定がない場合はデフォルトエンジンを使用
        
        Returns:
            list: 標準化された検索結果のリスト。各要素は以下の形式:
            {
                "title": "タイトル",
                "link": "URL",
                "snippet": "スニペット/説明文",
                "source": "検索エンジン名"
            }
        """
        engine = engine or self.default_engine
        standardized_results = []
        
        if engine == "google":
            # Googleの結果を標準化
            for response in results:
                if "items" in response:
                    for item in response["items"]:
                        standardized_results.append({
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", "").replace("\n", " "),
                            "source": "google"
                        })
        
        elif engine == "bing":
            # Bingの結果を標準化
            if "webPages" in results and "value" in results["webPages"]:
                for item in results["webPages"]["value"]:
                    standardized_results.append({
                        "title": item.get("name", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "bing"
                    })
        
        elif engine == "duckduckgo":
            # DuckDuckGoの結果を標準化
            for item in results:
                standardized_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("href", ""),
                    "snippet": item.get("body", ""),
                    "source": "duckduckgo"
                })
        
        return standardized_results

    def search_and_standardize(self, query, engine=None, scrape_urls=False, scrape_options=None, max_results=4, **kwargs):
        """
        検索を実行し、結果を標準化された形式で返す便利なメソッド
        
        Args:
            query (str): 検索クエリ
            engine (str, optional): 使用する検索エンジン。指定がない場合はデフォルトエンジンを使用
            scrape_urls (bool): 検索結果のURLをスクレイピングするかどうか
            scrape_options (dict, optional): スクレイピングのオプション
                - output_dir (str): 保存先ディレクトリ（デフォルト: "scraped_data"）
                - save_json (bool): JSONとして保存するかどうか（デフォルト: True）
                - save_markdown (bool): Markdownとして保存するかどうか（デフォルト: True）
                - exclude_links (bool): リンクテキストを除外するかどうか（デフォルト: False）
            **kwargs: 各検索エンジン固有のパラメータ
            
        Returns:
            dict: {
                "search_results": list[dict], # 標準化された検索結果のリスト
                "scraped_data": dict, # スクレイピング結果（scrape_urls=Trueの場合）
            }
        """
        raw_results = self.search(query, engine, max_results,**kwargs)
        standardized_results = self.process_results(raw_results, engine)
        
        response = {
            "search_results": standardized_results,
            "scraped_data": None
        }
        
        if scrape_urls and standardized_results:
            # スクレイピングオプションの設定
            scrape_options = scrape_options or {}
            urls = [result["link"] for result in standardized_results]
            
            # スクレイピングの実行
            scraped_data = self.scraper.scrape_multiple_urls(
                urls=urls,
                output_dir=scrape_options.get("output_dir", "scraped_data"),
                save_json=scrape_options.get("save_json", True),
                save_markdown=scrape_options.get("save_markdown", True),
                exclude_links=scrape_options.get("exclude_links", False),
                max_depth=scrape_options.get("max_depth", 20)
            )
            
            response["scraped_data"] = scraped_data
        
        return response

