import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    指定されたテキストのトークン数を計算します。

    Args:
        text (str): トークン数を計算するテキスト
        model (str): 使用するモデル名（デフォルト: "gpt-4o"）

    Returns:
        int: トークン数
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text)) 