# 项目介绍
    本项目是基于“Adaptive Activation Steering: A Tuning-Free LLM Truthfulness Improvement Method for Diverse Hallucinations Categories”论文进行的改进项目
    主要功能是使用引导向量将LLM幻觉引导至真实空间，降低大模型的幻觉生成概率

# 环境依赖
    conda create -n your_env_name python=3.9
    pip install -r requirements.txt
    mkdir activations directions models

# 目录结构
    AT/
    ├── TruthfulQA/                  # TruthfulQA文件夹
    │   ├── data/                    # 数据
    │       ├── v0/
    │           ├── mc_task.json     # 使用的数据
    │   └── truthfulqa/              # 工具函数文件夹
    │       ├── configs.py
    │       ├── evaluate.py
    │       ├── metrics.py
    │       ├── models.py
    │       ├── presets.py
    │       └── utilities.py
    ├── activations/                 # 激活向量文件夹
    ├── direactions/                 # 引导向量文件夹
    ├── models/                 # 引导向量文件夹
    ├── requirements.txt             # 依赖列表
    ├── gen_directions.py            # 获取引导向量函数
    ├── gen_activationss.py          # 获取激活向量函数
    ├── utils.py                     # 工具
    ├── valid_method.py              # 测试主函数
    └── README.md                    # 项目说明

# 使用说明
    安装环境后，在HuggingFace下载模型，将model_path改为对应的路径后，运行gen_activationss.py、gen_directions.py与valid_method.py
