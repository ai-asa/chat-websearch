import requests
from bs4 import BeautifulSoup, NavigableString, Comment
from typing import Dict, Optional, Union, Any, Tuple, List, Set
import logging
import re
from urllib.parse import urlparse, urljoin
import json
from datetime import datetime
import os
from .rate_limiter import RateLimiter
# import asyncio
# import aiohttp
import chardet
import time

class WebScraper:
    # クラス変数としてリストを定義
    UNWANTED_TAGS = ['script', 'style', 'meta', 'link', 'noscript']
    EMPTY_TAGS = ['div', 'span']
    TECHNICAL_CONTENT_PATTERNS = [
        'function', 'var ', 'const ', 'let ', '=>', 
        '{', '}', 'window.', 'document.',
        '<script', '<style', '@media', 
        'gtag', 'dataLayer', 'hbspt', 'hsVars'
    ]
    HEADING_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    PARAGRAPH_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol']
    CONTENT_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li']
    EMPTY_HEADING_MARKERS = ["#", "##", "###", "####", "#####", "######"]
    
    # 正規表現パターンを事前コンパイル（すべてクラス変数として定義）
    URL_PATH_PATTERN = re.compile(r'^https?://|^/[a-zA-Z0-9/]')
    SYMBOL_SEMICOLON_PATTERN = re.compile(r'^[^\w\s].*?[^\w\s]$')
    CONSECUTIVE_NEWLINES_PATTERN = re.compile(r'\n{3,}')
    INDENT_PATTERN = re.compile(r'^(\s*)')
    CONSECUTIVE_SPACES_PATTERN = re.compile(r' {4,}')
    HEADING_ONLY_PATTERN = re.compile(r'^#{1,6}\s*$')
    HEADING_START_PATTERN = re.compile(r'^#{1,6}')
    INVALID_FILENAME_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*\s]')
    
    # 文字化け検出用パターンもクラス変数として定義
    GARBLED_PATTERNS = [
        re.compile(r'[\uFFFD\uFFFE\uFFFF]'),  # 無効なUnicode文字
        re.compile(r'[\u0000-\u001F\u007F-\u009F]'),  # 制御文字
        re.compile(r'[\uD800-\uDFFF]'),  # サロゲートペア
        re.compile(r'ã[\\x80-\\xFF]+'),  # 典型的な日本語文字化けパターン
        re.compile(r'&#[0-9]+;'),  # 数値文字参照
        re.compile(r'%[0-9A-Fa-f]{2}'),  # URLエンコード
    ]
    JAPANESE_CHARS_PATTERN = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    
    def __init__(self, verify_ssl=True):
        """
        WebScraperクラスの初期化
        
        Args:
            verify_ssl (bool): SSLの検証を行うかどうか。デフォルトはTrue
        """
        self.verify_ssl = verify_ssl
        self.logger = logging.getLogger(__name__)
        self.exclude_links = False
        self.exclude_symbol_semicolon = False  # 記号で始まり;で終わる要素を除外
        self.exclude_garbled = False  # 文字化けした要素を除外
        self.rate_limiter = RateLimiter(default_delay=0.1)  # レート制限を追加
        
        # セッションの初期化と共通ヘッダーの設定
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
        })
        
        # リクエストの設定
        self.request_timeout = 30  # タイムアウト（秒）
        self.max_retries = 3      # 最大リトライ回数
        self.retry_delay = 0.5     # リトライ間隔（秒）

    def scrape_url(self, url: str, exclude_links: bool = False, 
                  exclude_symbol_semicolon: bool = True,
                  exclude_garbled: bool = True,
                  max_depth: int = 10) -> Optional[Dict[str, Any]]:
        """
        URLからHTMLを取得し、各形式のデータを返します。

        Args:
            url (str): スクレイピング対象のURL
            exclude_links (bool): リンクテキストを除外するかどうか
            exclude_symbol_semicolon (bool): 記号で始まり;で終わる要素を除外するかどうか
            exclude_garbled (bool): 文字化けした要素を除外するかどうか
            max_depth (int): HTMLの解析を行う最大の深さ
            
        Returns:
            Optional[Dict[str, Any]]: 以下の情報を含む辞書
                - raw_html: 取得した生のHTMLデータ
                - json_data: HTMLをJSON形式に変換したデータ
                - markdown_data: JSONをMarkdown形式に変換したデータ
                失敗時はNone
        """
        # 一時的に除外オプションの値を保存
        original_exclude_links = self.exclude_links
        original_exclude_symbol_semicolon = self.exclude_symbol_semicolon
        original_exclude_garbled = self.exclude_garbled
        
        self.exclude_links = exclude_links
        self.exclude_symbol_semicolon = exclude_symbol_semicolon
        self.exclude_garbled = exclude_garbled

        try:
            raw_html = self.fetch_html(url)
            if raw_html is None:
                return None
                
            # HTMLをJSONに変換（max_depthを渡す）
            json_data = self.html_to_json(raw_html, max_depth=max_depth)
            # JSONをMarkdownに変換
            markdown_data = self.json_to_markdown(json_data)
            
            return {
                "raw_html": raw_html,
                "json_data": json_data,
                "markdown_data": markdown_data
            }
        finally:
            # 元の値に戻す
            self.exclude_links = original_exclude_links
            self.exclude_symbol_semicolon = original_exclude_symbol_semicolon
            self.exclude_garbled = original_exclude_garbled

    # async def scrape_url_async(self, url: str, exclude_links: bool = False, 
    #               exclude_symbol_semicolon: bool = True,
    #               exclude_garbled: bool = True,
    #               max_depth: int = 10) -> Optional[Dict[str, Any]]:
    #     """
    #     URLからHTMLを非同期で取得し、各形式のデータを返します。

    #     Args:
    #         url (str): スクレイピング対象のURL
    #         exclude_links (bool): リンクテキストを除外するかどうか
    #         exclude_symbol_semicolon (bool): 記号で始まり;で終わる要素を除外するかどうか
    #         exclude_garbled (bool): 文字化けした要素を除外するかどうか
    #         max_depth (int): HTMLの解析を行う最大の深さ
            
    #     Returns:
    #         Optional[Dict[str, Any]]: 以下の情報を含む辞書
    #             - raw_html: 取得した生のHTMLデータ
    #             - json_data: HTMLをJSON形式に変換したデータ
    #             - markdown_data: JSONをMarkdown形式に変換したデータ
    #             失敗時はNone
    #     """
    #     # 一時的に除外オプションの値を保存
    #     original_exclude_links = self.exclude_links
    #     original_exclude_symbol_semicolon = self.exclude_symbol_semicolon
    #     original_exclude_garbled = self.exclude_garbled
        
    #     self.exclude_links = exclude_links
    #     self.exclude_symbol_semicolon = exclude_symbol_semicolon
    #     self.exclude_garbled = exclude_garbled

    #     try:
    #         raw_html = await self.fetch_html_async(url)
    #         if raw_html is None:
    #             return None
                
    #         # HTMLをJSONに変換（max_depthを渡す）
    #         json_data = self.html_to_json(raw_html, max_depth=max_depth)
    #         # JSONをMarkdownに変換
    #         markdown_data = self.json_to_markdown(json_data)
            
    #         return {
    #             "raw_html": raw_html,
    #             "json_data": json_data,
    #             "markdown_data": markdown_data
    #         }
    #     except Exception as e:
    #         self.logger.error(f"スクレイピング処理中にエラーが発生しました: {str(e)}")
    #         return None
    #     finally:
    #         # 元の設定に戻す
    #         self.exclude_links = original_exclude_links
    #         self.exclude_symbol_semicolon = original_exclude_symbol_semicolon
    #         self.exclude_garbled = original_exclude_garbled

    def fetch_html(self, url: str) -> Optional[str]:
        """
        指定されたURLからHTMLを取得します。
        
        Args:
            url (str): スクレイピング対象のURL
            
        Returns:
            Optional[str]: 取得したHTML。エラーの場合はNone
        """
        retries = 0
        while retries < self.max_retries:
            try:
                # リクエスト前に待機時間を確保
                self.rate_limiter.wait_if_needed(url)
                
                response = self.session.get(
                    url,
                    verify=self.verify_ssl,
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                
                # エンコーディングの処理
                encoding = None
                
                # Content-Typeヘッダーからエンコーディングを取得
                content_type = response.headers.get('content-type', '').lower()
                if 'charset=' in content_type:
                    encoding = content_type.split('charset=')[-1]
                
                # レスポンスのエンコーディングがISO-8859-1の場合、または未設定の場合
                if not encoding or response.encoding.lower() == 'iso-8859-1':
                    # chardetを使用してエンコーディングを推測
                    raw_content = response.content
                    encoding_result = chardet.detect(raw_content)
                    if encoding_result and encoding_result['encoding']:
                        encoding = encoding_result['encoding']
                
                if encoding:
                    response.encoding = encoding
                
                return response.text
                
            except requests.RequestException as e:
                retries += 1
                if retries < self.max_retries:
                    self.logger.warning(f"リトライ {retries}/{self.max_retries}: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"HTMLの取得に失敗しました: {str(e)}")
                    return None

    # async def fetch_html_async(self, url: str) -> Optional[str]:
    #     """
    #     指定されたURLからHTMLを非同期で取得します。
        
    #     Args:
    #         url (str): スクレイピング対象のURL
            
    #     Returns:
    #         Optional[str]: 取得したHTML。エラーの場合はNone
    #     """
    #     retries = 0
    #     while retries < self.max_retries:
    #         try:
    #             # リクエスト前に待機時間を確保
    #             await self.rate_limiter.wait_if_needed_async(url)
                
    #             async with aiohttp.ClientSession(headers={
    #                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    #                 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    #                 'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
    #             }) as session:
    #                 async with session.get(
    #                     url,
    #                     ssl=None if not self.verify_ssl else True,
    #                     timeout=aiohttp.ClientTimeout(total=self.request_timeout)
    #                 ) as response:
    #                     response.raise_for_status()
                        
    #                     # エンコーディングの処理
    #                     encoding = None
                        
    #                     # Content-Typeヘッダーからエンコーディングを取得
    #                     content_type = response.headers.get('content-type', '').lower()
    #                     if 'charset=' in content_type:
    #                         encoding = content_type.split('charset=')[-1]
                        
    #                     # エンコーディングが未設定の場合
    #                     if not encoding:
    #                         # テキストを取得してエンコーディングを推測
    #                         text = await response.text('iso-8859-1')
    #                         encoding_result = chardet.detect(text.encode('iso-8859-1'))
    #                         if encoding_result and encoding_result['encoding']:
    #                             encoding = encoding_result['encoding']
    #                             return text.encode('iso-8859-1').decode(encoding, errors='replace')
    #                         return text
                        
    #                     return await response.text(encoding=encoding, errors='replace')
                        
    #         except Exception as e:
    #             retries += 1
    #             if retries < self.max_retries:
    #                 self.logger.warning(f"非同期リトライ {retries}/{self.max_retries}: {str(e)}")
    #                 await asyncio.sleep(self.retry_delay)
    #             else:
    #                 self.logger.error(f"HTMLの非同期取得に失敗しました: {str(e)}")
                    return None

    def html_to_json(self, html: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        HTMLをJSON形式に変換します。
        
        Args:
            html (str): 変換対象のHTML文字列
            max_depth (int): HTMLの解析を行う最大の深さ
            
        Returns:
            Dict[str, Any]: JSON形式に変換されたHTML構造
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # 不要な要素を削除
        self._remove_unwanted_elements(soup)
        
        # html要素を取得
        html_element = soup.find('html')
        if html_element:
            return self._parse_node(html_element, max_depth=max_depth)
        return self._parse_node(soup, max_depth=max_depth)

    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """
        不要なHTML要素を削除します。
        
        Args:
            soup (BeautifulSoup): 処理対象のBeautifulSoupオブジェクト
        """
        # script, style, meta, link タグを削除
        for tag in soup.find_all(self.UNWANTED_TAGS):
            tag.decompose()
            
        # コメントを削除
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
            
        # JSON-LDを含むscriptタグを削除
        for tag in soup.find_all('script', type='application/ld+json'):
            tag.decompose()
            
        # 空のdiv, span要素を削除
        for tag in soup.find_all(self.EMPTY_TAGS):
            if not tag.get_text(strip=True):
                tag.decompose()
                
        # データ属性を含む要素を削除
        for tag in soup.find_all(lambda tag: any(attr.startswith('data-') for attr in tag.attrs)):
            if not any(child.name in self.CONTENT_TAGS for child in tag.find_all()):
                tag.decompose()
                
        # インラインスタイルを削除
        for tag in soup.find_all(style=True):
            del tag['style']

    def _is_garbled_text(self, text: str) -> bool:
        """
        文字列が文字化けしているかどうかを判定します。
        
        Args:
            text (str): 判定対象のテキスト
            
        Returns:
            bool: 文字化けしている場合はTrue
        """
        try:
            # 1. 制御文字のチェック（改行、タブ以外）
            if any(ord(c) < 32 and c not in '\n\t\r' for c in text):
                return True

            # 2. 文字化けパターンのチェック - コンパイル済みパターンを使用
            if any(pattern.search(text) for pattern in self.GARBLED_PATTERNS):
                return True

            # 3. 日本語として不自然な文字列パターンのチェック - コンパイル済みパターンを使用
            japanese_chars = len(self.JAPANESE_CHARS_PATTERN.findall(text))
            total_chars = len(text)
            
            if total_chars > 0 and japanese_chars > 0:
                # 日本語文字が含まれているが、不自然に断片化している場合
                if japanese_chars / total_chars < 0.1:  # 日本語文字の割合が10%未満
                    return True

            return False
        except UnicodeError:
            return True

    def _parse_node(self, node: Any, current_depth: int = 0, max_depth: int = 10) -> Union[Dict[str, Any], str, None]:
        """
        HTMLノードを再帰的にパースしてJSON形式に変換します。
        不要な要素は除外します。最大深度を超えた要素は削除されます。
        
        Args:
            node: パース対象のノード
            current_depth (int): 現在の再帰の深さ
            max_depth (int): 最大再帰深度
            
        Returns:
            Union[Dict[str, Any], str, None]: パースされたノードの構造、または深度超過時はNone
        """
        # 最大深度に達した場合、Noneを返して要素を削除
        if current_depth >= max_depth:
            return None

        # テキストノードの場合
        if isinstance(node, NavigableString):
            if isinstance(node, Comment):
                return ""
                
            text = str(node).strip()
            
            # 技術的なコンテンツを含む文字列を除外
            if any(pattern in text.lower() for pattern in self.TECHNICAL_CONTENT_PATTERNS):
                return ""
                
            # URLやパスのみの文字列を除外
            if self.URL_PATH_PATTERN.match(text):
                return ""
                
            # 記号で始まり記号で終わる要素を除外
            if self.exclude_symbol_semicolon and self.SYMBOL_SEMICOLON_PATTERN.match(text):
                return ""
                
            # 文字化けした要素を除外
            if self.exclude_garbled and self._is_garbled_text(text):
                return ""
                
            return text

        # 要素ノードの場合
        # リンク除外オプションが有効で、aタグの場合はスキップ
        if self.exclude_links and node.name == "a":
            return ""
            
        # 不要なタグの場合はスキップ
        if node.name in self.UNWANTED_TAGS:
            return ""

        attrs = dict(node.attrs) if node.attrs else {}
        # class属性をリストから文字列に変換
        if "class" in attrs and isinstance(attrs["class"], list):
            attrs["class"] = " ".join(attrs["class"])

        result = {
            "tag": node.name,
            "attributes": attrs,
            "children": []
        }

        # 子ノードを再帰的にパース（深度を増加させて）
        for child in node.children:
            child_result = self._parse_node(child, current_depth + 1, max_depth)
            if child_result:  # 空文字列や None の場合は追加しない
                if isinstance(child_result, str) and child_result.strip():
                    result["children"].append(child_result.strip())
                elif isinstance(child_result, dict):
                    result["children"].append(child_result)

        # 子要素が空の場合はNoneを返す
        if not result["children"] and not result["attributes"]:
            return None

        return result

    def json_to_markdown(self, json_data: Dict[str, Any], level: int = 0) -> str:
        """
        JSON形式のHTML構造をMarkdown形式に変換します。
        
        Args:
            json_data (Dict[str, Any]): 変換対象のJSON形式データ
            level (int): 現在の階層レベル（インデント用）

        Returns:
            str: Markdown形式の文字列
        """
        # 文字列の場合はそのまま返す
        if isinstance(json_data, str):
            return json_data

        result = []
        tag = json_data["tag"]
        attrs = json_data["attributes"]
        children = json_data["children"]

        # 特定のタグに応じたMarkdown要素を生成
        if tag == "h1":
            prefix = "# "
        elif tag == "h2":
            prefix = "## "
        elif tag == "h3":
            prefix = "### "
        elif tag == "h4":
            prefix = "#### "
        elif tag == "h5":
            prefix = "##### "
        elif tag == "h6":
            prefix = "###### "
        elif tag == "p":
            prefix = ""
        elif tag == "a":
            href = attrs.get("href", "")
            # リンクの子要素を処理
            child_texts = [
                text for text in (self.json_to_markdown(child, level + 1) for child in children)
                if text.strip()
            ]
            child_text = " ".join(child_texts)
            return f"[{child_text}]({href})" if child_text else ""
        elif tag == "ul":
            prefix = ""
        elif tag == "ol":
            prefix = ""
        elif tag == "li":
            prefix = "- "
        elif tag == "strong" or tag == "b":
            child_texts = [
                text for text in (self.json_to_markdown(child, level) for child in children)
                if text.strip()
            ]
            child_text = " ".join(child_texts)
            return f"**{child_text}**" if child_text else ""
        elif tag == "em" or tag == "i":
            child_texts = [
                text for text in (self.json_to_markdown(child, level) for child in children)
                if text.strip()
            ]
            child_text = " ".join(child_texts)
            return f"*{child_text}*" if child_text else ""
        elif tag == "code":
            child_texts = [
                text for text in (self.json_to_markdown(child, level) for child in children)
                if text.strip()
            ]
            child_text = " ".join(child_texts)
            return f"`{child_text}`" if child_text else ""
        elif tag == "pre":
            child_texts = [
                text for text in (self.json_to_markdown(child, level) for child in children)
                if text.strip()
            ]
            child_text = " ".join(child_texts)
            return f"```\n{child_text}\n```" if child_text else ""
        elif tag == "br":
            return "\n"
        else:
            prefix = ""

        # 子要素を処理
        for child in children:
            child_text = self.json_to_markdown(child, level + 1)
            if child_text:
                if prefix and not child_text.startswith(prefix):
                    result.append(prefix + child_text)
                else:
                    result.append(child_text)

        # 結果を結合
        markdown = "\n".join(result)

        # リストアイテムの場合、インデントを追加
        if tag in ["li"]:
            markdown = "  " * level + markdown

        # 段落やヘッダーの後に空行を追加
        if tag in self.PARAGRAPH_TAGS:
            markdown += "\n"

        # 見出しの場合、内容が空でないことを確認
        if tag in self.HEADING_TAGS:
            content = "".join(result).strip()
            if not content or content in self.EMPTY_HEADING_MARKERS:
                return ""

        return markdown

    def _clean_markdown(self, markdown: str) -> str:
        """
        Markdownテキストを整形します。
        
        Args:
            markdown (str): 整形対象のMarkdownテキスト
            
        Returns:
            str: 整形されたMarkdownテキスト
        """
        # 連続する改行を1つの改行に置換
        markdown = self.CONSECUTIVE_NEWLINES_PATTERN.sub('\n\n', markdown)
        
        # 行ごとに処理
        lines = markdown.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            # 連続する空白を4つまでに制限
            # 行頭のインデントは保持
            indent_match = self.INDENT_PATTERN.match(line)
            indent = indent_match.group(1) if indent_match else ''
            content = line[len(indent):]
            
            # インデントは8スペースまで許可（タブ2個相当）
            if len(indent) > 4:
                indent = indent[:4]
                
            # 行の内容の連続空白を4つまでに制限
            content = self.CONSECUTIVE_SPACES_PATTERN.sub('    ', content)
            line = indent + content
            
            # 空白のみの行をスキップ
            if not line.strip():
                # 前後の行をチェックして、必要な場合のみ空行を保持
                prev_line = cleaned_lines[-1] if cleaned_lines else ""
                next_line = lines[i+1] if i+1 < len(lines) else ""
                
                # 段落区切りとして必要な場合のみ空行を追加
                # 前の行が見出しや段落で、次の行にも内容がある場合
                if (prev_line.strip().startswith('#') or 
                    prev_line.strip()) and next_line.strip():
                    cleaned_lines.append("")
                continue
                
            # 見出し行の場合
            if self.HEADING_ONLY_PATTERN.match(line.strip()):
                # 次の非空行までチェック
                next_non_empty = None
                for next_line in lines[i+1:]:
                    if next_line.strip():
                        next_non_empty = next_line
                        break
                
                # 次の非空行が見出しの場合、現在の見出しをスキップ
                if next_non_empty and self.HEADING_START_PATTERN.match(next_non_empty.strip()):
                    continue
            
            cleaned_lines.append(line)
        
        # 最後の空行を削除
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
            
        return '\n'.join(cleaned_lines)

    def scrape_multiple_urls(
        self,
        urls: List[str],
        output_dir: str = "scraped_data",
        save_json: bool = True,
        save_markdown: bool = True,
        exclude_links: bool = False,
        max_depth: int = 20
    ) -> Dict[str, Dict[str, Union[Dict[str, Any], str, None]]]:
        """
        複数のURLをスクレイピングし、結果を保存します。

        Args:
            urls (List[str]): スクレイピング対象のURLリスト
            output_dir (str): 保存先ディレクトリ
            save_json (bool): JSONとして保存するかどうか
            save_markdown (bool): Markdownとして保存するかどうか
            exclude_links (bool): リンクテキストを除外するかどうか
            max_depth (int): HTMLの解析を行う最大の深さ
        Returns:
            Dict[str, Dict[str, Union[Dict[str, Any], str, None]]]: 
                URLをキーとし、以下の情報を含む辞書:
                - raw_html: 取得した生のHTMLデータ
                - json_data: スクレイピングしたJSONデータ
                - markdown_data: 変換したMarkdownデータ
                - json_file: 保存したJSONファイルのパス（保存した場合）
                - markdown_file: 保存したMarkdownファイルのパス（保存した場合）
        """
        # ファイルを保存する場合のみディレクトリを作成
        if save_json or save_markdown:
            os.makedirs(output_dir, exist_ok=True)
        results = {}

        for url in urls:
            self.logger.info(f"スクレイピング開始: {url}")
            result = self.scrape_url(url, exclude_links, max_depth=max_depth)
            
            if result:
                # ファイルに保存
                json_file, md_file = self.save_results(
                    result["json_data"],
                    url,
                    output_dir,
                    save_json=save_json,
                    save_markdown=save_markdown
                )
                
                results[url] = {
                    **result,
                    "json_file": json_file,
                    "markdown_file": md_file
                }
            else:
                self.logger.error(f"スクレイピング失敗: {url}")
                results[url] = {
                    "raw_html": None,
                    "json_data": None,
                    "markdown_data": None,
                    "json_file": None,
                    "markdown_file": None
                }

        return results

    # async def scrape_multiple_urls_async(
    #     self,
    #     urls: List[str],
    #     output_dir: str = "scraped_data",
    #     save_json: bool = True,
    #     save_markdown: bool = True,
    #     exclude_links: bool = False,
    #     max_depth: int = 20
    # ) -> Dict[str, Dict[str, Union[Dict[str, Any], str, None]]]:
    #     """
    #     複数のURLを非同期でスクレイピングし、結果を保存します。

    #     Args:
    #         urls (List[str]): スクレイピング対象のURLリスト
    #         output_dir (str): 保存先ディレクトリ
    #         save_json (bool): JSONとして保存するかどうか
    #         save_markdown (bool): Markdownとして保存するかどうか
    #         exclude_links (bool): リンクテキストを除外するかどうか
    #         max_depth (int): HTMLの解析を行う最大の深さ
    #     Returns:
    #         Dict[str, Dict[str, Union[Dict[str, Any], str, None]]]: 
    #             URLをキーとし、以下の情報を含む辞書:
    #             - raw_html: 取得した生のHTMLデータ
    #             - json_data: スクレイピングしたJSONデータ
    #             - markdown_data: 変換したMarkdownデータ
    #             - json_file: 保存したJSONファイルのパス（保存した場合）
    #             - markdown_file: 保存したMarkdownファイルのパス（保存した場合）
    #     """
    #     # ファイルを保存する場合のみディレクトリを作成
    #     if save_json or save_markdown:
    #         os.makedirs(output_dir, exist_ok=True)
    #     results = {}

    #     # 元のexclude_links設定を保存
    #     original_exclude_links = self.exclude_links
    #     # クラス変数にexclude_links設定を適用
    #     self.exclude_links = exclude_links

    #     try:
    #         # 非同期タスクのリストを作成
    #         tasks = []
    #         for url in urls:
    #             self.logger.info(f"非同期スクレイピング開始: {url}")
    #             tasks.append(self.scrape_url_async(url, exclude_links=exclude_links, exclude_symbol_semicolon=True, exclude_garbled=True, max_depth=max_depth))
            
    #         # すべてのタスクを並行実行
    #         scraped_results = await asyncio.gather(*tasks)
            
    #         # 結果を処理
    #         for i, url in enumerate(urls):
    #             result = scraped_results[i]
    #             if result:
    #                 # ファイルに保存
    #                 json_file, md_file = await self.save_results_async(
    #                     result["json_data"],
    #                     url,
    #                     output_dir,
    #                     save_json=save_json,
    #                     save_markdown=save_markdown
    #                 )
                    
    #                 results[url] = {
    #                     **result,
    #                     "json_file": json_file,
    #                     "markdown_file": md_file
    #                 }
    #             else:
    #                 self.logger.error(f"非同期スクレイピング失敗: {url}")
    #                 results[url] = {
    #                     "raw_html": None,
    #                     "json_data": None,
    #                     "markdown_data": None,
    #                     "json_file": None,
    #                     "markdown_file": None
    #                 }
    #     finally:
    #         # 元の設定に戻す
    #         self.exclude_links = original_exclude_links

    #     return results

    def save_results(
        self,
        result: dict,
        url: str,
        output_dir: str,
        save_json: bool = True,
        save_markdown: bool = True
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        スクレイピング結果を保存します。

        Args:
            result: スクレイピング結果
            url: スクレイピング対象のURL
            output_dir: 保存先ディレクトリ
            save_json: JSONとして保存するかどうか
            save_markdown: Markdownとして保存するかどうか

        Returns:
            Tuple[Optional[str], Optional[str]]: 保存したJSONとMarkdownのファイルパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # URLを安全なファイル名に変換
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path.replace('/', '_')
        if path:
            safe_name = f"{domain}{path}"
        else:
            safe_name = domain
            
        # 不正な文字を除去
        safe_name = self.INVALID_FILENAME_CHARS_PATTERN.sub('_', safe_name)
        # 長すぎるファイル名を防ぐ
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
            
        json_filename = None
        md_filename = None

        if save_json:
            json_filename = f"{output_dir}/{safe_name}_{timestamp}.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"JSONを保存しました: {json_filename}")

        if save_markdown:
            md_filename = f"{output_dir}/{safe_name}_{timestamp}.md"
            markdown_content = self.json_to_markdown(result)
            # Markdownの整形を行う
            markdown_content = self._clean_markdown(markdown_content)
            
            with open(md_filename, "w", encoding="utf-8") as f:
                # メタデータを追加
                f.write(f"# {domain}\n\n")
                f.write(f"URL: {url}\n")
                f.write(f"取得日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n") 
                f.write(markdown_content)
            self.logger.info(f"Markdownを保存しました: {md_filename}")

        return json_filename, md_filename

    # async def save_results_async(
    #     self,
    #     result: dict,
    #     url: str,
    #     output_dir: str,
    #     save_json: bool = True,
    #     save_markdown: bool = True
    # ) -> Tuple[Optional[str], Optional[str]]:
    #     """
    #     スクレイピング結果を非同期でファイルに保存します。
        
    #     Args:
    #         result (dict): スクレイピング結果のJSONデータ
    #         url (str): スクレイピング対象のURL
    #         output_dir (str): 保存先ディレクトリ
    #         save_json (bool): JSONとして保存するかどうか
    #         save_markdown (bool): Markdownとして保存するかどうか
            
    #     Returns:
    #         Tuple[Optional[str], Optional[str]]: 保存したJSONファイルとMarkdownファイルのパス
    #     """
    #     # URLからファイル名を生成
    #     domain = urlparse(url).netloc
    #     path = urlparse(url).path
        
    #     # 無効な文字を置換
    #     filename_base = f"{domain}{path}".replace("/", "_")
    #     filename_base = self.INVALID_FILENAME_CHARS_PATTERN.sub("_", filename_base)
        
    #     # ファイル名が長すぎる場合は切り詰める
    #     if len(filename_base) > 100:
    #         filename_base = filename_base[:100]
        
    #     json_file_path = None
    #     md_file_path = None
        
    #     # JSONファイルの保存
    #     if save_json:
    #         json_file_path = os.path.join(output_dir, f"{filename_base}.json")
    #         try:
    #             # 非同期でファイル操作を行うためにループを使用
    #             loop = asyncio.get_event_loop()
    #             await loop.run_in_executor(
    #                 None,
    #                 lambda: self._save_json_file(json_file_path, result)
    #             )
    #             self.logger.info(f"JSONファイルを保存しました: {json_file_path}")
    #         except Exception as e:
    #             self.logger.error(f"JSONファイルの保存に失敗しました: {str(e)}")
    #             json_file_path = None
        
    #     # Markdownファイルの保存
    #     if save_markdown:
    #         md_file_path = os.path.join(output_dir, f"{filename_base}.md")
    #         try:
    #             # 現在のexclude_links設定を保存
    #             original_exclude_links = self.exclude_links
                
    #             # exclude_linksの設定を適用（scrape_url_asyncで設定された値を使用）
    #             # Markdownに変換
    #             markdown_data = self.json_to_markdown(result)
                
    #             # 非同期でファイル操作を行うためにループを使用
    #             loop = asyncio.get_event_loop()
    #             await loop.run_in_executor(
    #                 None,
    #                 lambda: self._save_markdown_file(md_file_path, markdown_data)
    #             )
    #             self.logger.info(f"Markdownファイルを保存しました: {md_file_path}")
    #         except Exception as e:
    #             self.logger.error(f"Markdownファイルの保存に失敗しました: {str(e)}")
    #             md_file_path = None
        
    #     return json_file_path, md_file_path

    def _save_json_file(self, file_path: str, data: dict) -> None:
        """JSONファイルを保存するヘルパーメソッド"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_markdown_file(self, file_path: str, markdown_data: str) -> None:
        """Markdownファイルを保存するヘルパーメソッド"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_data)
