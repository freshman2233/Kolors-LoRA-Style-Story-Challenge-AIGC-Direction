import os
import json
from PIL import Image
from tqdm import tqdm
import pandas as pd

def step_1_load_dataset_and_save():
    """
    第一步：加载 MsDataset，并将数据保存到 ./data/lora_dataset/train 目录，同时生成 metadata.jsonl。
    """
    from modelscope.msdatasets import MsDataset

    # 1. 加载数据集
    ds = MsDataset.load(
        'AI-ModelScope/lowres_anime',
        subset_name='default',
        split='train',
        # 这里可以根据实际需求调整你的 cache_dir
        cache_dir="/mnt/workspace/kolors/data"
    )

    # 2. 创建保存目录
    os.makedirs("./data/lora_dataset/train", exist_ok=True)
    os.makedirs("./data/data-juicer/input", exist_ok=True)

    # 3. 写入 metadata.jsonl
    with open("./data/data-juicer/input/metadata.jsonl", "w", encoding="utf-8") as f:
        for data_id, data in enumerate(tqdm(ds, desc="Saving raw dataset")):
            image = data["image"].convert("RGB")
            image_path = f"./data/lora_dataset/train/{data_id}.jpg"
            image.save(image_path)

            # 准备 metadata
            metadata = {
                "text": "二次元",  # 这里可以根据需求修改
                "image": [image_path]
            }
            f.write(json.dumps(metadata, ensure_ascii=False))
            f.write("\n")


def step_2_create_data_juicer_config():
    """
    第二步：创建 data_juicer_config.yaml 配置文件，用于数据过滤。
    """
    data_juicer_config = """
# global parameters
project_name: 'data-process'
dataset_path: './data/data-juicer/input/metadata.jsonl'
np: 4

text_keys: 'text'
image_key: 'image'
image_special_token: '<__dj__image>'

export_path: './data/data-juicer/output/result.jsonl'

# process schedule
process:
    - image_shape_filter:
        min_width: 1024
        min_height: 1024
        any_or_all: any
    - image_aspect_ratio_filter:
        min_ratio: 0.5
        max_ratio: 2.0
        any_or_all: any
"""
    # 生成目录及写入配置
    os.makedirs("data/data-juicer", exist_ok=True)
    with open("data/data-juicer/data_juicer_config.yaml", "w", encoding="utf-8") as file:
        file.write(data_juicer_config.strip())


def step_3_run_data_juicer():
    """
    第三步：调用 data-juicer 命令行，对数据进行过滤。
    """
    # 请确保 data-juicer 已经可以在命令行使用（例如已安装在当前环境中）
    cmd = "dj-process --config data/data-juicer/data_juicer_config.yaml"
    os.system(cmd)


def step_4_save_filtered_data():
    """
    第四步：读取 data-juicer 过滤后的 result.jsonl，并保存最终用于训练的图片和 metadata.csv。
    """
    os.makedirs("./data/lora_dataset_processed/train", exist_ok=True)

    texts, file_names = [], []

    with open("./data/data-juicer/output/result.jsonl", "r", encoding="utf-8") as file:
        lines = file.readlines()

    for data_id, line in enumerate(tqdm(lines, desc="Saving filtered dataset")):
        data = json.loads(line)
        text = data["text"]
        image_path_src = data["image"][0]

        # 打开并重新保存到新的目录
        image = Image.open(image_path_src)
        image_path_dst = f"./data/lora_dataset_processed/train/{data_id}.jpg"
        image.save(image_path_dst)

        texts.append(text)
        file_names.append(f"{data_id}.jpg")

    # 生成 metadata.csv
    data_frame = pd.DataFrame()
    data_frame["file_name"] = file_names
    data_frame["text"] = texts
    data_frame.to_csv("./data/lora_dataset_processed/train/metadata.csv", index=False, encoding="utf-8-sig")

    print("Filtered dataset saved to ./data/lora_dataset_processed/train")


def step_5_download_models():
    """
    第五步：下载 Kolors 模型和 SDXL VAE 模型。
    """
    from diffsynth import download_models
    download_models(["Kolors", "SDXL-vae-fp16-fix"])


def step_6_check_train_script_help():
    """
    第六步：可选，查看 train_kolors_lora.py 的帮助信息。
    """
    cmd = "python DiffSynth-Studio/examples/train/kolors/train_kolors_lora.py -h"
    os.system(cmd)


def step_7_run_lora_training():
    """
    第七步：调用 DiffSynth-Studio 的脚本进行 LoRA 训练。
    注意：请根据你的实际路径、训练参数自行修改。
    """
    cmd = r"""
python DiffSynth-Studio/examples/train/kolors/train_kolors_lora.py \
  --pretrained_unet_path models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors \
  --pretrained_text_encoder_path models/kolors/Kolors/text_encoder \
  --pretrained_fp16_vae_path models/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors \
  --lora_rank 16 \
  --lora_alpha 4.0 \
  --dataset_path data/lora_dataset_processed \
  --output_path ./models \
  --max_epochs 1 \
  --center_crop \
  --use_gradient_checkpointing \
  --precision "16-mixed"
"""
    os.system(cmd.strip())


def step_8_inference_with_lora():
    """
    第八步：加载训练好的 LoRA 权重进行推理并保存生成图片。
    """
    import torch
    from diffsynth import ModelManager, SDXLImagePipeline
    from peft import LoraConfig, inject_adapter_in_model

    def load_lora(model, lora_rank, lora_alpha, lora_path):
        """
        加载 LoRA 权重到 model 中
        """
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out"],
        )

        model = inject_adapter_in_model(lora_config, model)
        state_dict = torch.load(lora_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        return model

    # 1. 加载预训练模型
    model_manager = ModelManager(
        torch_dtype=torch.float16,
        device="cuda",
        file_path_list=[
            "models/kolors/Kolors/text_encoder",
            "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
            "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors"
        ]
    )

    pipe = SDXLImagePipeline.from_model_manager(model_manager)

    # 2. 加载训练好的 LoRA
    pipe.unet = load_lora(
        pipe.unet,
        lora_rank=16,  # 需与训练时一致
        lora_alpha=2.0,  # 可适当调整
        lora_path="models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt"
    )

    # 3. 推理
    torch.manual_seed(0)
    prompt = "二次元，一个紫色短发小女孩，在家中沙发上坐着，双手托着腮，很无聊，全身，粉色连衣裙"
    negative_prompt = "丑陋、变形、嘈杂、模糊、低对比度"

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        cfg_scale=4,
        num_inference_steps=50,
        height=1024,
        width=1024
    )

    # 4. 保存结果图像
    result_path = "1.jpg"
    image.save(result_path)
    print(f"推理完成，结果已保存到 {result_path}")


def main():
    """
    整合执行各步骤
    """
    # 1. 加载并保存原始数据集
    step_1_load_dataset_and_save()

    # 2. 创建 data-juicer 配置文件
    step_2_create_data_juicer_config()

    # 3. 运行 data-juicer
    step_3_run_data_juicer()

    # 4. 保存过滤后的数据
    step_4_save_filtered_data()

    # 5. 下载所需模型
    step_5_download_models()

    # 6. (可选) 查看训练脚本帮助信息
    step_6_check_train_script_help()

    # 7. 执行 LoRA 训练
    step_7_run_lora_training()

    # 8. 推理生成图片
    step_8_inference_with_lora()


if __name__ == "__main__":
    main()
