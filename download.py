"""
	download.py - Q3 (DCU)
	by Ksuserkqy(20251113620)
	Docs: https://www.ksuser.cn/dcu/
	2025-10-20
"""

import shutil
from settings import *
from pathlib import Path
from typing import Optional
from modelscope.hub.snapshot_download import snapshot_download

def download_modelscope_model(model_id=MODEL_ID, revision:Optional[str]=None, force_redownload:bool=False) -> str:
    output_dir = MODEL_DIR
    target = Path(output_dir).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and force_redownload: shutil.rmtree(target)
    if target.exists() and not force_redownload: return str(target)
    local_dir = snapshot_download(
        model_id=model_id,
        revision=revision,
        local_dir=str(target),
        local_files_only=False,
    )
    return str(local_dir)

if __name__ == "__main__":
    path = download_modelscope_model()
    print(f"模型已下载至{path}")