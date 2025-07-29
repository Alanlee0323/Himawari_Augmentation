import os
import shutil

def create_directory(directory):
    """創建目錄（如果不存在）"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
def copy_file(src, dst):
    """複製文件並確保目標目錄存在"""
    dst_dir = os.path.dirname(dst)
    create_directory(dst_dir)
    shutil.copy2(src, dst)