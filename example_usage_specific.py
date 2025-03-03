from src.chat.openai_adapter import OpenaiAdapter
from src.chat.get_prompt import (
    get_insurance_product_keywords_prompt,
    get_insurance_product_analysis_prompt,
    get_insurance_product_judge_prompt,
    get_insurance_product_reviews_prompt,
    get_insurance_product_sales_pitch_prompt,
    get_insurance_product_switch_pitch_prompt
)
from src.websearch.web_search import WebSearch
from src.webscraping.web_scraping import WebScraper
import json
from dotenv import load_dotenv
import os
import time
from datetime import datetime

# 開発対象
# 1. 特定の情報源から指定された形式で情報を取得する機能
#    ・検索対象のサイトを限定（Google Custom Search Engine IDを使用）
#    ・出力フォーマットを指定
#    ・必要な情報のみを抽出

# 検索結果のイメージ
# ・商品名から保険会社と正式名称、保険種別を特定する
# → 全体web検索 + AI判定　→　保険　{名前}　みたいに検索する
# ・口コミを生成する
# → 全体web検索 + AI生成
# → 口コミと一緒に強み、弱みも取得できるか

# ・競合商品の検索
# → オリコン顧客満足度ランキングで条件指定して検索
# ・販売トーク
# →これまでの情報を基にAI生成

def main():
    load_dotenv()
    custom_search_engine_id = os.getenv("GOOGLE_CSE_ID")
    # OpenAIアダプターとWebSearchのインスタンスを作成
    openai = OpenaiAdapter()
    web_search = WebSearch(default_engine="google")
    # web_search.scraper = WebScraper(verify_ssl=False)  # SSL検証を無効化
    
    print("チャットボットを起動しました。終了するには 'quit' と入力してください。")
    
    while True:
        start_time_total = time.time()
        
        # ユーザー入力を受け取る
        user_input = input("\nユーザー: ")
        
        # 終了コマンドの確認
        if user_input.lower() == 'quit':
            print("チャットボットを終了します。")
            break
        
        # 個別の保険商品に関する質問かどうかを判定
        start_time = time.time()
        judge_response = openai.openai_chat(
            openai_model="gpt-4o",
            prompt=get_insurance_product_judge_prompt() + f"\n\nユーザーの入力: {user_input}"
        )
        print(f"\n商品判定処理時間: {time.time() - start_time:.2f}秒")
        
        # 判断結果の解析
        is_insurance_product_query = False
        if "<decision>" in judge_response and "</decision>" in judge_response:
            decision_start = judge_response.find("<decision>") + len("<decision>")
            decision_end = judge_response.find("</decision>")
            decision = judge_response[decision_start:decision_end].strip()
            is_insurance_product_query = decision == "1"
            
            # 判断理由の表示（オプション）
            if "<reasoning>" in judge_response and "</reasoning>" in judge_response:
                reasoning_start = judge_response.find("<reasoning>") + len("<reasoning>")
                reasoning_end = judge_response.find("</reasoning>")
                reasoning = judge_response[reasoning_start:reasoning_end].strip()
                # print(f"\n判断理由: {reasoning}")
        
        if not is_insurance_product_query:
            print("\n個別の保険商品を入力してください。")
            continue
        
        print("\n保険商品の分析を開始します")
        
        # 検索キーワードを生成
        start_time = time.time()
        keywords_response = openai.openai_chat(
            openai_model="gpt-4o",
            prompt=get_insurance_product_keywords_prompt() + f"\n\nユーザーの質問: {user_input}"
        )
        print(f"キーワード生成処理時間: {time.time() - start_time:.2f}秒")
        
        try:
            # キーワードの抽出
            keywords_start = keywords_response.find("[")
            keywords_end = keywords_response.find("]") + 1
            if keywords_start != -1 and keywords_end != -1:
                keywords_list = json.loads(keywords_response[keywords_start:keywords_end])
                if keywords_list:
                    keyword = keywords_list[0]  # 最初のキーワードのみを使用
                    # print(f"\n生成された検索キーワード: {keyword}")
                    
                    # Web検索を実行
                    start_time = time.time()
                    scrape_options = {
                        "save_json": False,
                        "save_markdown": False,
                        "exclude_links": True # リンクを除外
                    }
                    search_result = web_search.search_and_standardize(
                        keyword,
                        scrape_urls=True,
                        scrape_options=scrape_options,
                        max_results=10,
                        custom_search_engine_id=custom_search_engine_id
                    )
                    print(f"Web検索処理時間: {time.time() - start_time:.2f}秒")
                    
                    # 検索結果の整理
                    research_content = f"検索キーワード: {keyword}\n\n"
                    if search_result.get("scraped_data"):
                        for url, data in search_result["scraped_data"].items():
                            if data and "markdown_data" in data:
                                research_content += f"\n---\nURL: {url}\n{data['markdown_data']}\n"
                    
                    # 保険商品の分析
                    start_time = time.time()
                    analysis_response = openai.openai_chat(
                        openai_model="gpt-4o",
                        prompt=get_insurance_product_analysis_prompt() + f"\n\n{research_content}"
                    )
                    print(f"商品分析処理時間: {time.time() - start_time:.2f}秒")
                    
                    # 分析結果の解析
                    if "<analysis>" in analysis_response and "</analysis>" in analysis_response:
                        analysis_start = analysis_response.find("<analysis>") + len("<analysis>")
                        analysis_end = analysis_response.find("</analysis>")
                        analysis_json = analysis_response[analysis_start:analysis_end].strip()
                        try:
                            analysis_data = json.loads(analysis_json)
                            # print("\n分析結果:")
                            # print(f"保険会社: {analysis_data.get('company', 'unknown')}")
                            # print(f"商品名: {analysis_data.get('product_name', 'unknown')}")
                            # print(f"保険種別: {analysis_data.get('category', 'unknown')}")
                            
                            # 口コミの検索と分析
                            # print("\n(口コミ情報の収集を開始します)")
                            product_name = analysis_data.get('product_name', '')
                            if product_name:
                                review_keyword = f"{product_name} 口コミ"
                                # print(f"\n生成された検索キーワード: {review_keyword}")
                                
                                # 口コミのWeb検索を実行
                                start_time = time.time()
                                review_search_result = web_search.search_and_standardize(
                                    review_keyword,
                                    scrape_urls=True,
                                    scrape_options=scrape_options,
                                    max_results=10,
                                    custom_search_engine_id=custom_search_engine_id
                                )
                                print(f"口コミ検索処理時間: {time.time() - start_time:.2f}秒")
                                
                                # 検索結果の整理
                                review_content = f"検索キーワード: {review_keyword}\n\n"
                                if review_search_result.get("scraped_data"):
                                    for url, data in review_search_result["scraped_data"].items():
                                        if data and "markdown_data" in data:
                                            review_content += f"\n---\nURL: {url}\n{data['markdown_data']}\n"
                                
                                # 口コミの分析
                                start_time = time.time()
                                reviews_response = openai.openai_chat(
                                    openai_model="gpt-4o",
                                    prompt=get_insurance_product_reviews_prompt() + f"\n\n{review_content}"
                                )
                                print(f"口コミ分析処理時間: {time.time() - start_time:.2f}秒")
                                
                                # 分析結果の解析
                                if "<analysis>" in reviews_response and "</analysis>" in reviews_response:
                                    analysis_start = reviews_response.find("<analysis>") + len("<analysis>")
                                    analysis_end = reviews_response.find("</analysis>")
                                    reviews_json = reviews_response[analysis_start:analysis_end].strip()
                                    try:
                                        reviews_data = json.loads(reviews_json)
                                        # print("\n口コミ分析結果:")
                                        # print("\n【口コミ情報】")
                                        for review in reviews_data.get("reviews", []):
                                            # print(f"・内容：{review.get('content', '')}")
                                            # print(f"  情報源：{review.get('source', '')}")
                                            # print(f"  感情：{review.get('sentiment', '')}\n")
                                            pass
                                        
                                        # print("\n【強み】")
                                        for strength in reviews_data.get("strengths", []):
                                            # print(f"・{strength}")
                                            pass
                                        
                                        # print("\n【弱み】")
                                        for weakness in reviews_data.get("weaknesses", []):
                                            # print(f"・{weakness}")
                                            pass

                                        # 販売トークの生成
                                        start_time = time.time()
                                        sales_pitch_response = openai.openai_chat(
                                            openai_model="gpt-4o",
                                            prompt=get_insurance_product_sales_pitch_prompt() + f"\n\n商品情報:\n{research_content}\n\n強み:\n" + "\n".join([f"・{s}" for s in reviews_data.get("strengths", [])])
                                        )
                                        print(f"販売トーク生成処理時間: {time.time() - start_time:.2f}秒")

                                        if "<sales_pitch>" in sales_pitch_response and "</sales_pitch>" in sales_pitch_response:
                                            pitch_start = sales_pitch_response.find("<sales_pitch>") + len("<sales_pitch>")
                                            pitch_end = sales_pitch_response.find("</sales_pitch>")
                                            sales_pitch_json = sales_pitch_response[pitch_start:pitch_end].strip()
                                            try:
                                                sales_pitch_data = json.loads(sales_pitch_json)
                                                # print("\n【販売トーク】")
                                                # print(f"\n{sales_pitch_data.get('pitch', '')}")
                                            except json.JSONDecodeError:
                                                print("\nエラー: 販売トークの解析に失敗しました。")

                                        # 乗り換えトークの生成
                                        start_time = time.time()
                                        switch_pitch_response = openai.openai_chat(
                                            openai_model="gpt-4o",
                                            prompt=get_insurance_product_switch_pitch_prompt() + f"\n\n商品情報:\n{research_content}\n\n弱み:\n" + "\n".join([f"・{w}" for w in reviews_data.get("weaknesses", [])])
                                        )
                                        print(f"乗り換えトーク生成処理時間: {time.time() - start_time:.2f}秒")

                                        if "<switch_pitch>" in switch_pitch_response and "</switch_pitch>" in switch_pitch_response:
                                            switch_start = switch_pitch_response.find("<switch_pitch>") + len("<switch_pitch>")
                                            switch_end = switch_pitch_response.find("</switch_pitch>")
                                            switch_pitch_json = switch_pitch_response[switch_start:switch_end].strip()
                                            try:
                                                switch_pitch_data = json.loads(switch_pitch_json)
                                                # print("\n【乗り換えトーク】")
                                                # print(f"\n{switch_pitch_data.get('pitch', '')}")
                                                
                                                # 結果を整形して表示
                                                print(f"\n{keyword}について検索しました。")
                                                print(f"商品名：{analysis_data.get('product_name', '不明')}")
                                                print(f"会社名：{analysis_data.get('company', '不明')}")
                                                print("\n【口コミ】")
                                                for review in reviews_data.get("reviews", []):
                                                    print(f"・内容：{review.get('content', '')}")
                                                    print(f"  情報源：{review.get('source', '')}")
                                                    print(f"  感情：{review.get('sentiment', '')}\n")
                                                
                                                print("\n【強み】")
                                                for strength in reviews_data.get("strengths", []):
                                                    print(f"・{strength}")
                                                
                                                print("\n【弱み】")
                                                for weakness in reviews_data.get("weaknesses", []):
                                                    print(f"・{weakness}")
                                                
                                                print("\n【強みを生かした販売トーク】")
                                                print(f"{sales_pitch_data.get('pitch', '')}")
                                                
                                                print("\n【弱みを使った乗換トーク】")
                                                print(f"{switch_pitch_data.get('pitch', '')}")
                                                
                                                print("\n注意書き：商品の誹謗中傷はやめましょう。")
                                                print(f"\n総処理時間: {time.time() - start_time_total:.2f}秒")
                                            except json.JSONDecodeError:
                                                print("\nエラー: 乗り換えトークの解析に失敗しました。")

                                    except json.JSONDecodeError:
                                        print("\nエラー: 口コミ分析結果の解析に失敗しました。")
                                else:
                                    print("\nエラー: 口コミ分析結果の形式が不正です。")
                            else:
                                print("\nエラー: 商品名が取得できなかったため、口コミ検索をスキップします。")
                        except json.JSONDecodeError:
                            print("\nエラー: 分析結果の解析に失敗しました。")
                    else:
                        print("\nエラー: 分析結果の形式が不正です。")
                else:
                    print("\nエラー: 検索キーワードの生成に失敗しました。")
        except json.JSONDecodeError:
            print("\nエラー: 検索キーワードの解析に失敗しました。")
        except Exception as e:
            print(f"\nエラー: 処理中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main() 