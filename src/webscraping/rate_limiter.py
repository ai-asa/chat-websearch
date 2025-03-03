from urllib.parse import urlparse
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, default_delay=0.1):
        """
        Args:
            default_delay (float): 同じドメインへの連続リクエスト時の最小待機時間（秒）
        """
        self.last_request_time = defaultdict(float)
        self.default_delay = default_delay
        self.last_domain = None  # 直前にリクエストしたドメインを保持

    def wait_if_needed(self, url):
        """同じドメインに連続してリクエストする場合のみ、待機時間を確保する

        Args:
            url (str): リクエスト先のURL
        """
        domain = urlparse(url).netloc
        current_time = time.time()
        
        # 直前のリクエストが同じドメインだった場合のみ待機
        if domain == self.last_domain:
            elapsed_time = current_time - self.last_request_time[domain]
            if elapsed_time < self.default_delay:
                wait_time = self.default_delay - elapsed_time
                time.sleep(wait_time)
        
        # 現在の情報を記録
        self.last_request_time[domain] = current_time
        self.last_domain = domain 