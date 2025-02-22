# --- 1. 导入依赖、定义选项与工作目录 ---

from pathlib import Path
import os
import subprocess
import threading
import time
import socket

# 用于HTTP请求
import urllib.request

# 选项：可根据需要启用或禁用
OPTIONS = {
    'UPDATE_COMFY_UI': True,                 # 是否更新 ComfyUI
    'INSTALL_COMFYUI_MANAGER': True,         # 是否安装 ComfyUI Manager
    'INSTALL_KOLORS': True,                  # 是否安装 KOLORS 模块
    'INSTALL_CUSTOM_NODES_DEPENDENCIES': True  # 是否安装自定义节点依赖
}

# 获取当前工作路径
current_dir = !pwd
WORKSPACE = f"{current_dir[0]}/ComfyUI"

# --- 2. 检查并克隆 ComfyUI 仓库 ---

# 切换到 /mnt/workspace/
%cd /mnt/workspace/

# 如果 ComfyUI 目录不存在，则克隆
![ ! -d "$WORKSPACE" ] && echo "-= Initial setup ComfyUI =-" && git clone https://github.com/comfyanonymous/ComfyUI

# 切换到 ComfyUI 目录
%cd "$WORKSPACE"

# 如果需要更新 ComfyUI，则执行 git pull
if OPTIONS['UPDATE_COMFY_UI']:
    print("-= Updating ComfyUI =-")
    !git pull

# --- 3. 安装/更新 ComfyUI Manager ---

if OPTIONS['INSTALL_COMFYUI_MANAGER']:
    %cd custom_nodes
    ![ ! -d ComfyUI-Manager ] && echo "-= Initial setup ComfyUI-Manager =-" && git clone https://github.com/ltdrdata/ComfyUI-Manager
    %cd ComfyUI-Manager
    !git pull

# --- 4. 安装/更新 KOLORS ---

if OPTIONS['INSTALL_KOLORS']:
    %cd ../
    ![ ! -d ComfyUI-KwaiKolorsWrapper ] && echo "-= Initial setup KOLORS =-" && git clone https://github.com/kijai/ComfyUI-KwaiKolorsWrapper.git
    %cd ComfyUI-KwaiKolorsWrapper
    !git pull

# --- 5. 安装自定义节点依赖 ---

%cd "$WORKSPACE"

if OPTIONS['INSTALL_CUSTOM_NODES_DEPENDENCIES']:
    print("-= Install custom nodes dependencies =-")
    # 注意：根据实际脚本路径做检查
    ![ -f "custom_nodes/ComfyUI-Manager/scripts/colab-dependencies.py" ] && python "custom_nodes/ComfyUI-Manager/scripts/colab-dependencies.py"

# --- 6. 安装 cloudflared ---

!wget "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/cloudflared-linux-amd64.deb"
!dpkg -i cloudflared-linux-amd64.deb

# --- 7. 下载模型文件 ---

!wget -c "https://modelscope.cn/models/Kwai-Kolors/Kolors/resolve/master/unet/diffusion_pytorch_model.fp16.safetensors" -P ./models/diffusers/Kolors/unet/
!wget -c "https://modelscope.cn/models/Kwai-Kolors/Kolors/resolve/master/unet/config.json" -P ./models/diffusers/Kolors/unet/
!modelscope download --model=ZhipuAI/chatglm3-6b-base --local_dir ./models/diffusers/Kolors/text_encoder/
!wget -c "https://modelscope.cn/models/AI-ModelScope/sdxl-vae-fp16-fix/resolve/master/sdxl.vae.safetensors" -P ./models/vae/
!wget -c "https://modelscope.cn/models/Kwai-Kolors/Kolors/resolve/master/scheduler/scheduler_config.json" -P ./models/diffusers/Kolors/scheduler/
!wget -c "https://modelscope.cn/models/Kwai-Kolors/Kolors/resolve/master/model_index.json" -P ./models/diffusers/Kolors/

# --- 8. 写入自定义 LoRA 相关的 Python 脚本 ---

lora_node = """
import torch
from peft import LoraConfig, inject_adapter_in_model

class LoadKolorsLoRA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kolors_model": ("KOLORSMODEL", ),
                "lora_path": ("STRING", {"multiline": False, "default": "",}),
                "lora_alpha": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("KOLORSMODEL",)
    RETURN_NAMES = ("kolors_model",)
    FUNCTION = "add_lora"
    CATEGORY = "KwaiKolorsWrapper"

    def convert_state_dict(self, state_dict):
        prefix_rename_dict = {
            # 根据实际需求填写
        }
        suffix_rename_dict = {
            # 根据实际需求填写
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            for prefix in prefix_rename_dict:
                if name.startswith(prefix):
                    name = name.replace(prefix, prefix_rename_dict[prefix])
            for suffix in suffix_rename_dict:
                if name.endswith(suffix):
                    name = name.replace(suffix, suffix_rename_dict[suffix])
            state_dict_[name] = param
        # 示例：这里需要根据实际的 LoRA 权重key 做修改
        # lora_rank = state_dict_["..."].shape[0]
        lora_rank = 8  # 仅示例，需自行修改
        return state_dict_, lora_rank

    def load_lora(self, model, lora_rank, lora_alpha, state_dict):
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            # 其他 LoRA 参数酌情添加
        )
        model = inject_adapter_in_model(lora_config, model)
        model.load_state_dict(state_dict, strict=False)
        return model

    def add_lora(self, kolors_model, lora_path, lora_alpha):
        state_dict = torch.load(lora_path, map_location="cpu")
        state_dict, lora_rank = self.convert_state_dict(state_dict)
        kolors_model["pipeline"].unet = self.load_lora(
            kolors_model["pipeline"].unet,
            lora_rank,
            lora_alpha,
            state_dict
        )
        return (kolors_model,)

NODE_CLASS_MAPPINGS = {
    "LoadKolorsLoRA": LoadKolorsLoRA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadKolorsLoRA": "Load Kolors LoRA",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
""".strip()

os.makedirs("/mnt/workspace/ComfyUI/custom_nodes/ComfyUI-LoRA", exist_ok=True)
with open("/mnt/workspace/ComfyUI/custom_nodes/ComfyUI-LoRA/__init__.py", "w", encoding="utf-8") as f:
    f.write(lora_node)

# --- 9. 启动 ComfyUI，并通过 cloudflared 暴露端口 ---

%cd /mnt/workspace/ComfyUI

def iframe_thread(port):
    """等待ComfyUI端口开启后，通过cloudflared创建隧道并打印访问URL。"""
    while True:
        time.sleep(0.5)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        if result == 0:
            break
        sock.close()
    print("\nComfyUI finished loading, trying to launch cloudflared...\n")
    p = subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in p.stderr:
        text = line.decode()
        if "trycloudflare.com " in text:
            print("This is the URL to access ComfyUI:", text[text.find("http"):], end='')

threading.Thread(target=iframe_thread, daemon=True, args=(8188,)).start()

!python main.py --dont-print-server
