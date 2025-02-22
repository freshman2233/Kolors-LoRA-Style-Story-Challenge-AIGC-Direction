# 可图kolors-lora风格故事挑战赛-AIGC方向

**语言**

[English](./README.md)      [中文](./README_CN.md)

# 1. 项目简介

本项目是**可图kolors-lora风格故事挑战赛-AIGC方向**的参赛代码。

该项目旨在展示在通过基于扩散的模型生成艺术时，梯度检查点和无分类器指导如何显着提高图像质量和叙事连贯性。

我们提供端到端的工作流程——从数据收集和预处理，到风格专业的 LoRA（低秩适应）培训，再到最终的审美评估——参与者可以采用或修改这些工作流程，以用于自己的创意或研究目的。

[比赛链接](https://tianchi.aliyun.com/s/ce4dc8bf800db1e58d51263ff357d28f)

### 关键目标

1.高质量图像生成

通过将LoRA与Kolors文本到图像扩散模型相结合，该项目展示了大规模AI美术生成器如何在保持高视觉保真度的同时，根据特定的美术风格或故事叙述要求进行微调。

2.增强讲故事的连贯性

该项目包括一个多图像叙事管道，说明人工智能生成的视觉效果如何遵循叙事弧线——从背景到高潮再到分辨率——使用精心设计的提示和负面提示。

3.实用的端到端工作流

我们提供关键阶段的示例脚本和指导：使用data - juicer进行数据预处理，使用DiffSynth-Studio框架进行LoRA训练，以及在ComfyUI中生成故事序列的多步推理。工作流被设计为可重复、可定制，并且易于使用其他节点或模块进行扩展。

### 技术亮点

1.Kolors模型

由快手的“Ket Tu”（Kolors）团队开发，Kolors是一个大规模的潜在扩散模型，训练了数十亿对文本图像。它擅长生成视觉愉悦、语义丰富的图像，尤其是中文提示，但它也支持英文文本生成。

2.梯度检查点

该技术通过选择性地缓存某些激活来减少训练期间GPU内存的消耗，允许在不进行重大硬件升级的情况下训练更大的模型或更大的小批量。

3.Classifier-Free指导

通过调整无分类器引导尺度（CFG尺度），用户可以微调文本提示对最终图像的影响。较高的指导强调及时的保真度，而较低的指导允许更多的创造性或意想不到的输出。

4.Lora训练

低秩自适应只修改原始扩散模型层中的小的低秩矩阵。
这种方法是非常有效的参数，保留了大多数原始模型的权重，同时使其适应新的风格或领域。
微调速度和较低的内存开销使其成为定制大型扩散模型的理想解决方案。

5.提示设计和消极提示

该项目提供了详细的提示示例，说明主题描述、风格提示和负面提示（不希望的属性，如“低质量”或“裁剪”）如何一起工作，以形成高质量的、主题连贯的结果。

6.数据预处理与美学评价

Data-Juicer：一个强大的工具链，用于过滤、标记和增强图像。它可以处理诸如删除低分辨率数据或自动生成训练文本描述之类的任务。
美学评分：利用预先训练的美学预测器来排除次等图像并保证一致的风格和质量。

## 1.1成果展示

![3319ae4b77d744abb68843f079e75c8c](./assets/3319ae4b77d744abb68843f079e75c8c.png)

![0b88b3de4e584e2da38c251e264966b5](./assets/0b88b3de4e584e2da38c251e264966b5.png)



## 1.2提示词

### 1.2.1 组成部分

主体描述，细节描述，修饰词，艺术风格，艺术家

【promts】Beautiful and cute girl, smiling, 16 years old, denim jacket, gradient background, soft colors, soft lighting, cinematic edge lighting, light and dark contrast, anime, super detail, 8k

【Negative prompts】(lowres, low quality, worst quality:1.2), (text:1.2),  deformed, black and white,disfigured, low contrast, cropped, missing  fingers

### 1.2.2 主体描述

【提示】美丽可爱的女孩，微笑，16岁，牛仔夹克，渐变背景，柔和的色彩，柔和的灯光，电影边缘照明，明暗对比，动漫，超级细节，8k

【负向提示】(低分辨率，低质量，最差质量:1.2)，(文本:1.2)，变形，黑白，毁损，低对比度，裁剪，缺手指

 ```
 英勇的女勇者在古老的村庄中央，身穿银光闪闪的盔甲，手持宝剑，准备出发。她的眼神坚定，身后是朝阳初升的天空。高清细节，动态光影。高精度3D渲染。宫崎骏风格
 
 低分辨率，文字覆盖，形态扭曲，低对比度，黑白图
 
 女勇者跨越充满神秘色彩的森林，四周缭绕着迷雾和发光的蘑菇。她小心翼翼地躲避森林中的陷阱。梦幻般的色彩，柔和光线。数字水彩画。顾廷烨风格。
 
 图像模糊，颜色失真，画面噪点。
 
 在森林深处，女勇者面对一条守护龙，龙的鳞片在阳光下反射出耀眼的光芒。她举剑向前冲刺。精细的纹理，动态模糊。超现实主义。达利风格。
 
 画面静止，动作模糊，过暗
 
 发现一条隐蔽的通道通向龙的巢穴，女勇者顺着石质通道悄然前行，通道内壁刻满古老的符文。历史感，微光照明。哥特式风格。蒂姆·伯顿风格。
 
 细节缺失，色彩过饱和。
 
 女勇者到达巨龙的巢穴，洞穴内火光冲天，巨龙在岩浆中苏醒，准备迎战。火焰效果，高对比度。动画风格。新海诚风格
 
 颜色单一，画面平淡
 
 激烈的战斗展开，女勇者利用敏捷的身手躲避龙的火焰攻击，剑尖闪烁着寒光反击。温馨照明，细腻表情。古典油画。佛朗西斯科·戈雅风格
 
 表情僵硬，光线不足
 
 恶龙倒下后，女勇者解开了囚禁公主的锁链，公主穿着破旧的礼服，带着泪眼含笑感谢救命之恩。温馨照明，细腻表情。古典油画。佛朗西斯科·戈雅风格
 
 表情僵硬，光线不足
 
 女勇者和公主骑着白马返回村庄，沿途村民们欢呼庆祝，花瓣从天而降，场面喜庆。节日气氛，柔和滤镜。浪漫主义。约翰·康斯特布尔风格
 
 场景拥挤，色彩冲突
 ```



## 1.3 成果展示2

![2419c4e52f2b46768c9dec2472fd7466](./assets/2419c4e52f2b46768c9dec2472fd7466.png)

![984c36407242432d8837d7320bea9a9c](./assets/984c36407242432d8837d7320bea9a9c.png)



## 1.4 提示词

| **图片编号** | **场景描述**               | **正向提示词**                                               |
| ------------ | -------------------------- | ------------------------------------------------------------ |
| 图片1        | 女主正在上课               | 古风，水墨画风格，一位黑色长发的少女坐在古木课桌前，教室内充满光线，窗外可见远山如黛。少女穿着精致的红色长裙，上半身，专心听讲，眼神专注。 |
| 图片2        | 开始睡着了                 | 古风，水墨画风格，黑色长发的少女趴在木制的课桌上，头枕在双臂之上，面容安详，红色长裙轻柔地覆盖在背上，教室宁静，窗外柳树摇曳。 |
| 图片3        | 进入梦乡，梦到自己站在路旁 | 古风，水墨画风格，黑色长发的少女站在风景如画的古道旁，两侧是古老的松树和石灯，少女身着红色长裙，面带微笑，远处有朦胧的山影，上半身。 |
| 图片4        | 王子骑马而来               | 古风，水墨画风格，英俊的王子穿着白色长袍，骑着白马从山间小路奔来，背景是风景秀丽的山脉和流水，马蹄溅起的水花与周围环境融为一体，全身。 |
| 图片5        | 两人相谈甚欢               | 古风，水墨画风格，英俊少年和黑发少女坐在林荫下的古桥上，桥下清澈的溪流潺潺，两人面对面，眼神交流充满情感，身着传统服饰，少女红裙，少年白袍，上半身。 |
| 图片6        | 一起坐在马背上             | 古风，水墨画风格，英俊的少年和长发的少女一同骑马，穿行在满是樱花的古道上，两人微笑交谈，少年白袍飘飘，少女红裙摇曳，全身。 |
| 图片7        | 下课了，梦醒了             | 古风，水墨画风格，黑色长发的少女坐在课桌旁，惊讶地睁大眼睛，从梦中醒来，教室内同学们的活动渐渐增多，背景有教室内的其它桌椅和学习用品，上半身。 |
| 图片8        | 又回到了学习生活中         | 古风，水墨画风格，黑色长发少女坐在木质课桌前，专心致志地看着黑板，手中拿着书写工具，周围教室布置古朴，窗外景色宁静，上半身。 |

| **图片编号** | **场景描述**               | **正向提示词**                                               | **反向提示词**                   |
| ------------ | -------------------------- | ------------------------------------------------------------ | -------------------------------- |
| 图片1        | 女主正在上课               | 古风，水墨画，一个黑色长发少女，坐在教室里，盯着黑板，深思，上半身，红色长裙 | 丑陋，变形，嘈杂，模糊，低对比度 |
| 图片2        | 开始睡着了                 | 古风，水墨画，一个黑色长发少女，坐在教室里，趴在桌子上睡着了，上半身，红色长裙 | 丑陋，变形，嘈杂，模糊，低对比度 |
| 图片3        | 进入梦乡，梦到自己站在路旁 | 古风，水墨画，一个黑色长发少女，站在路边，上半身，红色长裙   | 丑陋，变形，嘈杂，模糊，低对比度 |
| 图片4        | 王子骑马而来               | 古风，水墨画，一个英俊少年，骑着白马，上半身，白色衬衫       | 丑陋，变形，嘈杂，模糊，低对比度 |
| 图片5        | 两人相谈甚欢               | 古风，水墨画，一个英俊少年，白色衬衫，一个黑色长发少女，红色长裙，两个人一起聊天，开心，上半身 | 丑陋，变形，嘈杂，模糊，低对比度 |
| 图片6        | 一起坐在马背上             | 古风，水墨画，一个英俊少年，白色衬衫，一个黑色长发少女，红色长裙，两个人一起骑着马，全身 | 丑陋，变形，嘈杂，模糊，低对比度 |
| 图片7        | 下课了，梦醒了             | 古风，水墨画，一个黑色长发少女，坐在教室里，下课铃声响了，同学们开始走动，从睡梦中醒来，深思，上半身，红色长裙 | 丑陋，变形，嘈杂，模糊，低对比度 |
| 图片8        | 又回到了学习生活中         | 古风，水墨画，一个黑色长发少女，坐在教室里，盯着黑板，认真上课，上半身，红色长裙 | 丑陋，变形，嘈杂，模糊，低对比度 |

# 2.介绍

## 2.1可图大模型

可图大模型是由快手可图团队开发的基于潜在扩散的大规模文本到图像生成模型。

Kolors  在数十亿图文对下进行训练，在视觉质量、复杂语义理解、文字生成（中英文字符）等方面，相比于开源/闭源模型，都展示出了巨大的优势。同时，Kolors 支持中英双语，在中文特色内容理解方面更具竞争力。

更多的实验结果和细节请查看[技术报告](https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/Kolors_paper.pdf)。

## 2.2 AI生图

AI生图（AI生成图片，AI绘画）是指利用人工智能技术，根据输入的文本、草图或图片生成符合描述的视觉作品。这项技术结合了**深度学习、计算机视觉和自然语言处理**，能够创造高质量的图像，广泛应用于**艺术创作、广告设计、游戏开发、影视制作**等领域。

**GANs（生成对抗网络）**
通过两个神经网络（生成器和判别器）相互竞争，不断优化图像质量。例如：

- **StyleGAN**（由NVIDIA开发，能生成逼真的人脸）
- **BigGAN**（生成高分辨率的多类别图像）

**扩散模型（Diffusion Models）**
通过逐步去噪的方式生成高质量图像，目前主流的AI绘画模型多采用这一技术：

- **Stable Diffusion**（开源，可本地部署，应用广泛）
- **DALL·E**（由OpenAI开发，可根据文本生成创意图像）
- **MidJourney**（流行于艺术创作领域，风格化强）

**变分自编码器（VAE, Variational Autoencoders）**
用于降维和图像生成，如**VQ-VAE**可生成高清图片。

**CLIP（Contrastive Language-Image Pretraining）**
由OpenAI开发，可以理解文本和图像的关系，帮助AI根据文本生成更精准的图片。

 常见AI生图工具

| 工具名称             | 主要特点                     | 适用场景           |
| -------------------- | ---------------------------- | ------------------ |
| **Stable Diffusion** | 开源，支持本地部署，插件丰富 | 个人创作、商业设计 |
| **DALL·E 2**         | 由OpenAI开发，擅长细节和创意 | 广告、插画         |
| **MidJourney**       | 风格化强，社区驱动           | 艺术作品、概念设计 |
| **Runway ML**        | Web端易用，适合视频创作      | 短视频、影视制作   |
| **Deep Dream**       | Google开发，梦幻风格         | 抽象艺术           |
| **Artbreeder**       | 基因混合式AI绘画             | 头像设计、角色创作 |

### 2.2.1 难点和挑战

手：解决这个问题的相关技术，如：给图片里的人手打上标记，像把手掌、拇指、食指啥的，都给清楚地标出来；

**AI生图模型**获得图片生成能力主要是通过 学习 **图片描述** 以及 **图片特征**，尝试将这两者进行一一对应，存储在自己的记忆里。

通过输入的文字，复现出来对应的图片特征，从而生成了我们需要的图片。

关于AI是否真正理解了图片背后所代表的世界的特征，是否理解了图片的含义，这个一直是科研界和产业界存在争议的话题，我们唯一可以确定的是——由于每个模型用于训练的数据是有限的且不一定相同的，它们能匹配的描述和特征也是有限的，所以在风格、具体事物上，不同的模型会有很大的生成差异，且可能存在诸多与现实不符的情况。

AI味：AI生成的图片和实际生活场景/艺术家创作的绘画/摄影/三维作品 相比，存在强烈的违和感，或是细节处理，或是画面逻辑性存在问题，一言就能被看出是“AI出品”

- **观察图片的细节**。仔细检查人物的面部特征，尤其是眼睛和嘴巴
- **检查光线和阴影**。分析图片中的光源是否一致，阴影的方向是否与光源相符，是否存在不自然的光线或阴影
- **分析像素**。放大图片，寻找是否有模糊或像素化的部分。
- **注意背景**。检查背景中是否有不协调的元素，比如物体边缘是否平滑，背景中是否有不自然的重复模式。



### 2.2.2 生成技术介绍与分析

1.基础文生图模型优化
DALLE-2
DALLE-2 is an advanced AI model developed by OpenAI for generating images from textual descriptions. It's part of a broader class of models known as "large language models" (LLMs) that have been extended to understand and generate visual content based on natural language inputs. DALLE-2 uses deep learning techniques to create images that match detailed and complex prompts, producing high-quality and creative visuals. This capability has various applications, from helping artists and designers come up with new ideas to generating unique images for advertising and storytelling.

Stable Diffusion
Stable Diffusion is a type of deep learning model that generates high-quality images from textual descriptions. It's an open-source tool powered by a type of neural network called a latent diffusion model. This technology allows for generating detailed and diverse images based on user inputs, making it a powerful tool for artists, designers, and content creators.One of the key features of Stable Diffusion is its ability to fine-tune the results according to specific styles or requirements. Users can guide the image generation process through detailed prompts or by adjusting the model's parameters, making it highly flexible for creative purposes. Additionally, because it's open-source, it has a rapidly growing community of users who contribute to its development and use it in innovative ways.

Diffusion Transformer
The Diffusion Transformer is a type of machine learning model that combines elements of diffusion models and transformers to generate high-quality images from textual descriptions. This approach marries the generative capabilities of diffusion models, which gradually convert random noise into coherent images, with the powerful language understanding and sequence modeling capabilities of transformers.

2.可控生成与图像编辑

ControlNet

T21-Adapter

Dreambooth

3.生成采样加速

DDIM

Consistency model

Rectified Flow

## 2.3 文生图的工作流平台工具 ComfyUI

### 2.3.1 什么是ComfyUI

GUI 是 “Graphical User Interface”（图形用户界面）的缩写

简单来说，GUI 就是你在电脑屏幕上看到的那种有**图标、按钮和菜单**的交互方式。

**ComfyUI** 是GUI的一种，用于操作图像的生成技术，将AIGC模块化，类似思维导图的流程图一样，控制图像生成。

### 2.3.2 ComfyUI核心模块

1.**模型加载器**：加载基础模型文件，Model：模型文件



2.**提示词管理器**：CLIP(Contrastive Language–Image Pre-training)

将文字转化为模型可以理解的隐空间嵌入，

关于什么是隐空间呢？说白了就是从高维转化为的保留有重要特征信息的**低维信息**。



3.**采样器**：用于控制模型生成图像，不同的采样取值会影响最终输出图像的质量和多样性。

采样器可以调节生成过程的速度和质量之间的平衡。

Stable Diffusion的基本原理是通过**降噪**的方式（如完全的噪声图像），

将一个原本的噪声信号变为无噪声的信号（如人可以理解的图像）。



**模型是如何训练的呢？**

在训练过程中，模型学习最有效的方法来逆转扩散过程。

它本质上是学习图像是如何逐渐变得有噪声的，然后利用这些知识做相反的事情：

从噪声开始，一步一步地减少它，直到一个连贯的图像出现。



其中的降噪过程涉及到多次的采样。采样的系数在KSampler中配置：

- seed：控制噪声产生的随机种子
- control_after_generate：控制seed在每次生成后的变化
- steps：降噪的迭代步数，越多则信号越精准，相对的生成时间也越长
- cfg：classifier free guidance决定了prompt对于最终生成图像的影响有多大。更高的值代表更多地展现prompt中的描述。
- denoise: 多少内容会被噪声覆盖 sampler_name、scheduler：降噪参数。

4.**解码器**：VAE解码器

变分自编码器（Variational Autoencoder, VAE）是一种生成模型，

它在自编码器（Autoencoder, AE）的基础上引入了概率建模的思想，

使得模型不仅可以学习数据的低维表示，还可以生成新样本。

VAE的核心思想是学习一个潜在变量（Latent Variable）的概率分布，并从该分布中采样来生成数据。

VAE 由两个部分组成：

- **编码器（Encoder）**：将输入数据映射到一个潜在空间（Latent Space），并学习其均值（μ）和方差（σ2）。
- **解码器（Decoder）**：从潜在空间的样本重构原始数据。



## 2.4Lora微调

### 2.4.1 Lora简介

LoRA (Low-Rank Adaptation)  微调是一种用于在预训练模型上进行高效微调的技术。

它可以通过高效且灵活的方式实现模型的个性化调整，使其能够适应特定的任务或领域，同时保持良好的泛化能力和较低的资源消耗。

### 2.4.2 Lora微调的原理

LoRA通过在预训练模型的关键层中添加低秩矩阵来实现。

这些低秩矩阵通常被设计成具有较低维度的参数空间，这样它们就可以在不改变模型整体结构的情况下进行微调。

在训练过程中，只有这些新增的低秩矩阵被更新，而原始模型的大部分权重保持不变。

### 2.4.2 Lora微调的优势

快速适应**新任务**

 在特定领域有**少量标注数据**的情况下，也可以有效地对模型进行**个性化调整**，可以迅速适应新的领域或特定任务。

 保持**泛化能力**

 LoRA通过微调模型的一部分，有助于保持模型在未见过的数据上的泛化能力，同时还能学习到特定任务的知识。

 **资源效率**

 LoRA旨在通过仅微调模型的部分权重，而不是整个模型，从而减少所需的计算资源和存储空间。

### 2.4.3 Lora详解

| **参数名称**                   | **参数值**                                                   | **说明**                                         |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| `pretrained_unet_path`         | models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors | 指定预训练**UNet**模型的路径                     |
| `pretrained_text_encoder_path` | models/kolors/Kolors/text_encoder                            | 指定预训练**文本编码器**的路径                   |
| `pretrained_fp16_vae_path`     | models/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors | 指定预训练**VAE**模型的路径                      |
| `lora_rank`                    | 16                                                           | 设置LoRA的秩（rank），影响模型的复杂度和性能     |
| `lora_alpha`                   | 4                                                            | 设置LoRA的alpha值，控制微调的强度                |
| `dataset_path`                 | data/lora_dataset_processed                                  | 指定用于训练的数据集路径                         |
| `output_path`                  | ./models                                                     | 指定训练完成后保存模型的路径                     |
| `max_epochs`                   | 1                                                            | 设置最大训练轮数为1                              |
| `center_crop`                  |                                                              | 启用中心裁剪，用于图像预处理                     |
| `use_gradient_checkpointing`   |                                                              | 启用梯度检查点，节省显存                         |
| `precision`                    | “16-mixed”                                                   | 设置训练时的精度为混合16位精度（half precision） |

## 2.5 准备数据集

### 2.5.1明确你的需求和目标

**应用场景**：艺术风格转换、产品图像生成、医疗影像合成等

**数据类型**：你需要什么样的图片？比如是真实世界的照片还是合成图像？是黑白的还是彩色的？是高分辨率还是低分辨率？

**数据量**：考虑你的任务应该需要多少图片来支持训练和验证。

### 2.5.2 数据集来源整理

| **来源类型**          | 推荐                                                         |
| --------------------- | ------------------------------------------------------------ |
| **公开的数据平台**    | 魔搭社区内开放了近3000个数据集，涉及文本、图像、音频、视频和多模态等多种场景，左侧有标签栏帮助快速导览，大家可以看看有没有自己需要的数据集。https://www.modelscope.cn/datasets?Tags=object-tracking&dataType=video&page=1 ；其他数据平台推荐：[ImageNet](http://image-net.org/)：包含数百万张图片，广泛用于分类任务，也可以用于生成任务。[Open Images](https://storage.googleapis.com/openimages/web/index.html)：由Google维护，包含数千万张带有标签的图片。[Flickr](https://github.com/haltakov/flickr30k-dataset)：特别是Flickr30kK和Flickr8K数据集，常用于图像描述任务。[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)：专注于人脸图像的数据集。[LSUN](https://www.yf.io/p/lsun) (Large-scale Scene Understanding)：包含各种场景类别的大规模数据集。 |
| **使用API或爬虫获取** | 如果需要特定类型的内容，可以利用API从图库网站抓取图片，如Unsplash、Pexels等。使用网络爬虫技术从互联网上抓取图片，但需要注意版权问题。 |
| **数据合成**          | 利用现有的图形引擎（如Unity、Unreal Engine）或特定软件生成合成数据，这在训练某些类型的模型时非常有用。最近Datawhale联合阿里云天池，做了一整套多模态大模型数据合成的学习，欢迎大家一起交流。[从零入门多模态大模型数据合成](https://datawhaler.feishu.cn/wiki/PVmkwDClQiKbmOk1e7scYo2Pndd?fromScene=spaceOverview) |
| **数据增强**          | 对于较小的数据集，可以通过旋转、翻转、缩放、颜色变换等方式进行数据增强。 |
| **购买或定制**        | 如果你的应用是特定领域的，比如医学影像、卫星图像等，建议从靠谱的渠道购买一些数据集。 |

# 3.比赛详情

1. 参赛者需在可图Kolors 模型的基础上训练LoRA 模型，生成无限风格，如水墨画风格、水彩风格、赛博朋克风格、日漫风格......  

2. 基于LoRA模型生成 8 张图片组成连贯故事，故事内容可自定义；

3. 基于8图故事，评估LoRA风格的美感度及连贯性

   ##  样例：偶像少女养成日记

| 一位小女孩坐在沙发上，感到无聊                               | 小女孩来到演唱会现场，被偶像姐姐吸引                         | 小女孩的脸上露出了憧憬的神情                                 | 演唱会结束后，成为偶像的梦想在少女心中发芽                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![1-2.jpeg](./assets/57a54baa-7731-4dfe-9a84-9bf4be46488b.jpeg) | ![2-2.jpeg](./assets/9fe20985-16c3-482d-8fbc-1c9dc685d8b0.jpeg) | ![3-2.jpeg](./assets/e183fd34-7836-45e5-bf61-564a39cba340.jpeg) | ![4.jpeg](./assets/e574bee2-64fe-45a4-b7a3-3e0d2a1df3de.jpeg) |

| 为了成为偶像，少女开始日复一日的歌唱练习                     | 春去秋来，少女已然成长，成为自己心中憧憬的样子               | 终于，少女迎来了第一次登台演出，换上了黑色礼服，内心忐忑     | 在聚光灯的映衬下，少女一展歌喉，闪闪发光的梦想成为现实       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![5.jpeg](./assets/dd5ad0c6-e654-4847-a45e-357eec359ca6.jpeg) | ![6.jpeg](./assets/46ea90e7-ce72-486f-9622-aefdce57b46e.jpeg) | ![7.jpeg](./assets/455bfb03-1d2d-456d-8d46-8eb1dc2539e4.jpeg) | ![8.jpeg](./assets/40fc4001-ba63-4133-9b4f-3906d239ba61.jpeg) |



## 3.1训练及推理工具

我们为参赛选手提供了必要的 LoRA 训练及推理支持，训练框架由开源项目 DiffSynth-Studio 提供支持。

> [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 是开源的 Diffusion 引擎，旨在将基于 Diffusion 的丰富图像及视频模型集中到一起，实现开源生态互联，从而充分发挥其强大的生成能力，目前 DiffSynth-Studio 已经支持了 Stable Diffusion、ControlNet、AnimateDiff、IP-Adapter、混元 DiT、Kolors、ExVideo 等模型，且提供了简洁易用的训练框架，该框架可以轻松地实现 Kolors 等模型的 LoRA 训练。

为了方便参赛选手处理训练数据，我们在 notebook 文件中提供了使用 Data-Juicer 处理数据的样例代码，选手可以参考该部分代码，对图片数据集进行精细地处理。

> [Data-Juicer](https://github.com/modelscope/data-juicer) 是专为大模型训练设计的一站式多模态数据处理系统，在图像处理上，Data-Juicer 提供了丰富的算子，包括图像自动化标注、图像有害内容识别与过滤、水印检测、图像美学评估等。用户可以方便地使用 Data-Juicer 处理和过滤图像，获得高质量的训练数据，从而实现惊艳的图像生成效果。

在训练模型所需的计算资源上，参赛选手可以使用魔搭社区提供的免费计算资源，选手可以在“魔搭社区官网 - 我的 notebook”启动 GPU 实例，选择预装镜像 ubuntu22.04-cuda12.1.0-py310-torch2.3.0-tf2.16.1-1.16.1 后直接在线运行我们提供的 notebook。

在推理阶段，我们也提供了 ComfyUI 支持，参赛选手可以自行利用 ComfyUI 实现工作流搭建。

在模型训练方法上，参赛选手可以从多个角度入手提升模型生成的图像质量，我们提供了一些训练经验，供选手参考：

- 使用高质量数据集训练模型，例如
  - 对图像数据集进行美学评估并过滤掉低美感图像
  - 为图像数据生成对应的描述文本，提高训练数据中的文图相关性
- 调整训练脚本中的参数，例如
  - 对于特定风格的 LoRA 训练，LoRA rank 调大后（例如 16）通常效果稍好
  - 对于特定实体的 LoRA 训练，LoRA rank 调小后（例如 4）通常效果更稳定
- 搭建更强大的工作流，例如
  - 采用低分辨率生成加高分辨率润色的方式，分两阶段生成高清大图

文件说明：

- baseline.ipynb：LoRA 训练脚本
- ComfyUI.ipynb：ComfyUI 脚本
- kolors_example.json：ComfyUI 工作流（不带 LoRA）   
- kolors_with_lora_example.json：ComfyUI 工作流（带 LoRA）



## 3.3 赛程说明

 报名与组队：即日起—2024年8月31日 23:59

1.参赛者通过天池平台进行报名，确保报名信息准确有效，否则会被取消参赛资格及激励；

2.参赛组织可以单人或者多人自由组队,每队限最多5人，每人只能参加1支队伍；如果是多人团队，则需指定1名队长，负责沟通事宜；

3.本赛道所有选手需在8月31日23:59前完成实名认证（**实名认证入口：天池网站-个人中心-认证-支付宝实名认证**），未按要求完成实名认证队伍，将被取消参赛资格。

**特别提醒：**

- 每个队伍中，队长与队员都需要进行实名验证才符合“队伍完成实名认证”的需求。
- 组队成功后，点击左侧「我的团队」即可查看本队伍实名情况。

初赛：报名后-2024年8月31日23:59

决赛入围通知：2024年9月2日，从初赛作品中结合专业技术评委+人气投票筛选30组选手入围决赛（客观评分+评委主观评分）

决赛：2024年9月9日14:00-17:00 答辩展示，线上决出一二三等奖（评委主观评分）

颁奖：2024年9月19日~9月21日，线下云栖大会颁奖及展示



## 3.4作品提交

选手在天池平台后，需在魔搭平台上提交作品。步骤如下：

1、将训练好的LoRA 模型上传到[魔搭模型库](https://modelscope.cn/aigc/models?name=Kolors-LoRA-example-anime&page=1)

- LoRA模型命名为：队伍名称-可图Kolors训练-xxxxxx
- LoRA 上传地址：https://modelscope.cn/models/create?template=text-to-image-lora

2、作品发布在比赛品牌馆讨论区，https://modelscope.cn/brand/view/Kolors?branch=0&tree=11

- 发布标题格式为：天池平台报名队伍名称+LoRA模型链接地址+作品图（8图故事）

## 3.5评分标准

本次比赛通过客观评分判断选手提交作品的有效性，但最终评分以主观评分为准。

### 1、主观评分

由评委对参赛作品进行投票，评审标准可以从技术运用（40%）、组图风格连贯性（30%）、整体视觉效果（30%）几方面进行评判投票。

### 2、客观评分

美学分数仅作评价提交是否有效的标准，其中美学分数小于6（阈值可能根据比赛的实际情况调整，解释权归主办方所有）的提交被视为无效提交，无法参与主观评分。

此外，我们会核实选手上传的模型文件，赛选手需提交训练的LoRA 模型文件、LORA 模型的介绍、以及使用该模型生成的至少8张图片和对应 prompt，以便我们能够复现生成效果，对于生成效果明显无法复现的，取消获奖资格。

### 3、美学评分

```python
pip install simple-aesthetics-predictor

import torch, os
from PIL import Image
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV2Linear
from modelscope import snapshot_download


model_id = snapshot_download('AI-ModelScope/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE', cache_dir="models/")
predictor = AestheticsPredictorV2Linear.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
device = "cuda"
predictor = predictor.to(device)


def get_aesthetics_score(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = predictor(**inputs)
    prediction = outputs.logits
    return prediction.tolist()[0][0]


def evaluate(folder):
    scores = []
    for file_name in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file_name)):
            image = Image.open(os.path.join(folder, file_name))
            scores.append(get_aesthetics_score(image))
    if len(scores) == 0:
        return 0
    else:
        return sum(scores) / len(scores)


score = evaluate("./images")
print(score)
```



# 4.项目仓库

1. 环境准备
2. 数据集加载和预处理
3. 数据清洗和过滤
4. CLIP模型评估
5. Pytorch数据集与数据加载器
6. StableDiffusion图像生成





# 5.项目运行

本项目在阿里云PAI-DSW云平台环境下运行，需要魔搭社区授权。

**环境要求**：

- 已安装 `modelscope`、`data-juicer`、`diffsynth`、`peft`、`torch`、`tqdm`、`pandas`、`PIL` 等依赖。
- `dj-process` 命令可以被当前 Python 环境正确识别执行。
- 如果是容器环境，需要确保容器能访问外部网络下载模型。

**数据大小**：如果数据量比较大，`data-juicer` 的过滤过程可能需要较长时间或较大硬盘空间做缓存。

**超参调整**：LoRA 训练时的 `--lora_rank`、`--lora_alpha`、`--precision`、`--max_epochs`、`--dataset_path`、`--output_path` 等，需要根据实际需求和资源限制进行调整。

**推理时**：请注意加载的 checkpoint 路径，以及 LoRA rank、alpha 等参数是否与训练一致。

## 5.1可图培训数据集下载

```python
git lfs install
git clone https://www.modelscope.cn/datasets/maochase/kolors.git
```

## 5.2 Python环境

```python
!pip install simple-aesthetics-predictor
!pip install -v -e data-juicer
!pip uninstall pytorch-lightning -y
!pip install peft lightning pandas torchvision
!pip install -e DiffSynth-Studio
```

## 5.3上传结果

mkdir /mnt/workspace/kolors/output & cd

cp /mnt/workspace/kolors/models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt /mnt/workspace/kolors/output/

cp /mnt/workspace/kolors/1.jpg /mnt/workspace/kolors/output/

队伍名称-可图Kolors训练-xxxxxx

freshman2233-可图Kolors训练-勇士杀巨龙救公主



# 6.ComfyUI运行

```python
# Installs Git Large File Storage (LFS)
git lfs install

# Clones a Git repository from the provided URL. 
git clone https://www.modelscope.cn/datasets/maochase/kolors_test_comfyui.git

# Moves all the contents from the cloned repository directory 'kolors_test_comfyui' into the current directory.
mv kolors_test_comfyui/* ./

# Removes the now-empty directory 'kolors_test_comfyui' after its contents have been moved out.
rm -rf kolors_test_comfyui/

# Creates a new directory structure '/mnt/workspace/models/lightning_logs/version_0/checkpoints/'. 
mkdir -p /mnt/workspace/models/lightning_logs/version_0/checkpoints/

# Moves a specific model checkpoint file 'epoch=0-step=500.ckpt' to the newly created directory. 
mv epoch=0-step=500.ckpt /mnt/workspace/models/lightning_logs/version_0/checkpoints/

```



# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# 参考资料

```
[1] https://github.com/Kwai-Kolors/Kolors/tree/master
[2] Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis
[3] 最新图像生成技术研究方向-介绍与分析 https://www.bilibili.com/video/BV1vT421k7Qc/
[4] https://www.modelscope.cn/headlines/article/537?from=blibli_video
```

