from src.chat.openai_adapter import OpenaiAdapter
from src.chat.get_prompt import (
    get_system_prompt,
    get_web_research_judge_prompt,
    get_web_research_keywords_prompt,
    get_web_research_summarize_prompt,
    get_web_research_system_prompt
)
from src.websearch.web_search import WebSearch
from src.tiktoken import count_tokens
import json
from dotenv import load_dotenv
import os
import time

# 開発対象
# 1. 特定の保険商品の情報を調べて特定の形で出力する機能
#    ・検索対象のサイトを限定する
#    ・複数商品についての比較は、下記と同様の処理
# 2. フリートークの中で、必要な情報を調べる
#    ・ウェブリサーチの必要有無を判断
#    ・ウェブリサーチのクエリを作成
#    ・ウェブリサーチの結果をもとにAIが回答を作成
# → 複数の対象を同時に調べたり、比較する必要がある場合の処理は？
#    ・クエリはリストで出力するようにする　→　無駄に複数回検索することにならないか心配
#    ・一回ごとの検索結果は一つにまとめてAIに整理させる
#    ・各検索結果、ユーザーの要求を基に、AIが回答を作成

def main():
    load_dotenv()
    custom_search_engine_id = os.getenv("GOOGLE_CSE_ID")
    # OpenAIアダプターとWebSearchのインスタンスを作成
    openai = OpenaiAdapter()
    web_search = WebSearch(default_engine="google")
    # web_search.scraper = WebScraper(verify_ssl=False)  # SSL検証を無効化
    
    # 会話履歴を保持するリスト
    conversation_history = []
    
    print("チャットボットを起動しました。終了するには 'quit' と入力してください。")
    
    while True:
        # ユーザー入力を受け取る
        user_input = input("\nユーザー: ")
        
        # 終了コマンドの確認
        if user_input.lower() == 'quit':
            print("チャットボットを終了します。")
            break
            
        start_time_total = time.time()
        
        # Webリサーチが必要かどうかを判断
        start_time = time.time()
        judge_response = openai.openai_chat(
            openai_model="gpt-4o",
            prompt=get_web_research_judge_prompt() + f"\n\nユーザーの入力: {user_input}\n\n会話履歴:\n" + 
            "\n".join([
                f"ユーザー: {turn['user']}\n" +
                f"アシスタント: {turn.get('assistant', '')}\n" +
                f"ウェブ検索使用: {'はい' if turn.get('used_web_research') else 'いいえ'}"
                for turn in conversation_history
            ])
        )
        
        # 判断結果の解析
        needs_web_research = False
        if "<decision>" in judge_response and "</decision>" in judge_response:
            decision_start = judge_response.find("<decision>") + len("<decision>")
            decision_end = judge_response.find("</decision>")
            decision = judge_response[decision_start:decision_end].strip()
            needs_web_research = decision == "1"
            
            # 判断理由の表示（オプション）
            if "<reasoning>" in judge_response and "</reasoning>" in judge_response:
                reasoning_start = judge_response.find("<reasoning>") + len("<reasoning>")
                reasoning_end = judge_response.find("</reasoning>")
                reasoning = judge_response[reasoning_start:reasoning_end].strip()
                print(f"\n判断理由: {reasoning}")
        else:
            print("\nエラー: 判断結果を正しい形式で取得できませんでした。")
            needs_web_research = False
        
        print(f"\n判断処理時間: {time.time() - start_time:.2f}秒")
        
        web_research_results = []
        
        if needs_web_research:
            print("\n(Webリサーチが必要と判断されました)")
            
            # 検索キーワードを生成
            start_time = time.time()
            keywords_response = openai.openai_chat(
                openai_model="gpt-4o",
                prompt=get_web_research_keywords_prompt() + f"\n\nユーザーの質問: {user_input}\n\n会話履歴:\n" + "\n".join([f"ユーザー: {turn['user']}\nアシスタント: {turn.get('assistant', '')}" for turn in conversation_history])
            )
            
            try:
                # キーワードの抽出
                keywords_start = keywords_response.find("[")
                keywords_end = keywords_response.find("]") + 1
                if keywords_start != -1 and keywords_end != -1:
                    keywords_list = json.loads(keywords_response[keywords_start:keywords_end])
                    print("\n生成された検索キーワード:")
                    
                    # 各キーワードで検索を実行
                    for keyword in keywords_list:
                        print(f"- {keyword}")
                        start_time = time.time()
                        scrape_options = {
                            "save_json": False,
                            "save_markdown": False,
                            "exclude_links": True, # リンクを除外
                            "max_depth": 20
                        }
                        # Web検索を実行し、Markdown形式でデータを取得
                        search_result = web_search.search_and_standardize(
                            keyword,
                            scrape_urls=True,
                            scrape_options=scrape_options,
                            max_results=5,
                            custom_search_engine_id=custom_search_engine_id
                        )
                        
                        # print(f"\n検索結果: {json.dumps(search_result['search_results'], ensure_ascii=False, indent=2)}")
                        
                        # 検索結果とスクレイピングデータを整理
                        research_content = f"検索キーワード: {keyword}\n\n"
                        current_chunk = research_content
                        intermediate_summaries = []
                        
                        # スクレイピング結果の確認
                        has_valid_content = False
                        if search_result.get("scraped_data"):
                            for url, data in search_result["scraped_data"].items():
                                if data and "markdown_data" in data:
                                    has_valid_content = True
                                    new_content = f"\n---\nURL: {url}\n{data['markdown_data']}\n"
                                    # トークン数を計算
                                    if count_tokens(current_chunk + new_content) > 30000:
                                        # 現在のチャンクを中間要約
                                        intermediate_summary = openai.openai_chat(
                                            openai_model="gpt-4o",
                                            prompt=get_web_research_summarize_prompt() + f"\n\n{current_chunk}"
                                        )
                                        intermediate_summaries.append(intermediate_summary)
                                        # 新しいチャンクを開始
                                        current_chunk = new_content
                                    else:
                                        current_chunk += new_content
                        
                        if has_valid_content:
                            # 最後のチャンクを処理
                            if current_chunk:
                                intermediate_summary = openai.openai_chat(
                                    openai_model="gpt-4o",
                                    prompt=get_web_research_summarize_prompt() + f"\n\n{current_chunk}"
                                )
                                intermediate_summaries.append(intermediate_summary)
                            
                            # すべての中間要約を結合
                            summary = "\n\n".join(intermediate_summaries)
                            print(f"検索結果整理処理時間: {time.time() - start_time:.2f}秒")
                        else:
                            summary = "情報の取得に失敗しました。"
                            
                        # print(f"summary: {summary}")
                        
                        web_research_results.append({
                            "keyword": keyword,
                            "summary": summary
                        })
                else:
                    # print("\nエラー: 検索キーワードを正しい形式で取得できませんでした。")
                    pass
            except json.JSONDecodeError:
                # print("\nエラー: 検索キーワードのJSONパースに失敗しました。")
                pass
        
        # 会話履歴を含むシステムプロンプトを生成
        if web_research_results:
            # print(f"web_research_results: {web_research_results}")
            # Web検索結果がある場合は、検索結果を含むプロンプトを使用
            system_prompt = get_web_research_system_prompt(conversation_history, web_research_results)
        else:
            # Web検索結果がない場合は、通常のプロンプトを使用
            system_prompt = get_system_prompt(conversation_history)
        
        # OpenAI APIを使用してレスポンスを生成
        start_time = time.time()
        response = openai.openai_chat(
            openai_model="gpt-4o",
            prompt=system_prompt + f"\n\nユーザー: {user_input}"
        )
        print(f"レスポンス生成処理時間: {time.time() - start_time:.2f}秒")
        
        # 会話履歴に追加
        conversation_history.append({
            "user": user_input,
            "assistant": response,
            "used_web_research": needs_web_research,  # ウェブ検索を使用したかどうかを記録
            "web_research_results": web_research_results if web_research_results else None  # 検索結果も保存
        })
        
        # レスポンスの表示
        if response:
            print(f"\nアシスタント: {response}")
            print(f"\n総処理時間: {time.time() - start_time_total:.2f}秒")
        else:
            print("\nエラー: レスポンスを取得できませんでした。")
            print(f"\n総処理時間: {time.time() - start_time_total:.2f}秒")

if __name__ == "__main__":
    main() 