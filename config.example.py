"""本地配置模板。

使用方法：
1. 将本文件复制为 config.py。
2. 只在本机的 config.py 中填写真实值。
3. 不要把 config.py 或任何真实密钥提交到 Git。
"""

# 运行命令行问答、网页服务端模式和在线对比实验时使用。
DEEPSEEK_API_KEY = ""

# 网页端共享访问码。仅当服务端已配置 DEEPSEEK_API_KEY 时生效。
ACCESS_CODE = ""

# 仅重新抓取 CC98 数据时使用；运行现有知识库无需填写。
CC98_ACCESS_TOKEN = ""
