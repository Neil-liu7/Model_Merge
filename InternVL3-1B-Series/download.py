import os
from modelscope import HubApi
from modelscope import snapshot_download

# ⚠️ 强烈建议：立刻去 ModelScope 换一个新的 Token！
YOUR_ACCESS_TOKEN = 'ms-7428e330-1582-4864-9276-39bd84eef237'

# 登录 ModelScope
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

def download_model():
    model_id = "zhou170316/InternVL3-1B-geo"
    
    # 💡 核心修改点：将本地路径指定为你想要的名称
    local_path = "./InternVL3-1B-geo"  
    
    print(f"⏳ 开始从 ModelScope 下载 {model_id} 的 checkpoint-1975 文件夹...")
    print(f"📂 目标主目录: {os.path.abspath(local_path)}")
    
    try:
        # 匹配 checkpoint-1975 文件夹及其所有子内容
        # 确保下载配置文件、Python 代码文件和权重文件
        patterns = [
            "checkpoint-1975/**", 
            "*.json", 
            "*.py", 
            "*.md", 
            "*.txt"
        ]

        # 下载时，会自动在 local_path 下创建 checkpoint-1975 文件夹
        model_dir = snapshot_download(
            model_id, 
            allow_patterns=patterns, 
            local_dir=local_path,
        )
        print(f"\n✅ 模型文件夹下载完成！")
        
        # 打印出具体的 checkpoint 文件夹绝对路径
        final_dir = os.path.abspath(os.path.join(local_path, 'checkpoint-3660'))
        print(f"🎉 权重文件已成功保存在: {final_dir}")
        
    except Exception as e:
        print(f"\n❌ 下载过程中出现错误: {e}")

if __name__ == "__main__":
    download_model()