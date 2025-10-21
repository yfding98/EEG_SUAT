import os

def remove_empty_dirs(root_dir):
    """
    递归删除 root_dir 下所有空的文件夹。
    从最深层开始删除，确保嵌套的空文件夹也能被清理。
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # topdown=False 表示从最深层开始遍历
        try:
            os.rmdir(dirpath)
            print(f"Deleted empty directory: {dirpath}")
        except OSError:
            # 如果文件夹非空（比如还有文件或子文件夹未被删除），则跳过
            pass

if __name__ == "__main__":
    # 请将下面的路径替换为你想清理的根目录
    root_directory = input("请输入要清理的根目录路径（直接回车则使用当前目录）: ").strip()
    if not root_directory:
        root_directory = "."

    if os.path.isdir(root_directory):
        print(f"开始清理空文件夹，根目录: {os.path.abspath(root_directory)}")
        remove_empty_dirs(root_directory)
        print("清理完成。")
    else:
        print("错误：指定的路径不是有效目录。")