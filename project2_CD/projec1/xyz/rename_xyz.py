#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 xyz_out/ 文件夹里的 .xyz 文件，从 1 开始依次重新编号：
    1.xyz, 2.xyz, 3.xyz, ...
原文件名会被覆盖，建议提前备份。
"""

import os
from pathlib import Path

def main():
    folder = Path("xyz_out")
    if not folder.exists():
        print("文件夹 xyz_out/ 不存在！")
        return

    files = sorted(folder.glob("*.xyz"))
    if not files:
        print("xyz_out/ 里没有找到任何 .xyz 文件")
        return

    for idx, f in enumerate(files, start=1):
        new_name = folder / f"{idx}.xyz"
        print(f"重命名: {f.name} -> {new_name.name}")
        f.rename(new_name)

if __name__ == "__main__":
    main()
