# SECNEURON-LLM-Advanced-CN



本项目用于在小型 LLM（如 `facebook/opt-125m` / `facebook/opt-350m`）上近似复现 SECNEURON 的关键实验逻辑，包括任务级能力控制、数据级滥用防护、多任务动态权限控制、代码生成能力抑制，以及任务特异神经元选择与朴素剪枝对比。

> 注意：本项目不是 SECNEURON 官方实现，也不能直接复现论文中 OPT-6.7B、OPT-30B、Gemma-2-27B 等大模型实验数值。它主要用于理解论文方法、验证实验流程和构建后续研究基础。

---

## 1. 项目结构

```text
SECNEURON-LLM-Advanced-CN/
│
├── README.md                         # GitHub 项目说明文档
├── requirements.txt                  # Python 依赖
├── advanced_secneuron_llm.py          # 主实验代码
│
└── docs/
    ├── experiment_design.md           # 实验设计说明
    ├── symbols.md                     # 主要符号说明
    ├── troubleshooting.md             # 常见问题
    └── code_comments.md               # 代码注释规范
```

---

## 2. 环境安装

建议使用 `.venv` 虚拟环境。

```powershell
cd D:\SECNEURON\SECNEURON-LLM-Advanced-CN

python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

如果 HuggingFace 下载较慢，可以使用镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

---

## 3. 快速运行

### 3.1 快速 demo 模式

```powershell
python advanced_secneuron_llm.py ^
  --device cpu ^
  --model facebook/opt-125m ^
  --train_samples 20 ^
  --eval_samples 10 ^
  --k_per_task 40 ^
  --allow_fallback_demo ^
  --out_dir results_quick
```

### 3.2 真实数据模式

```powershell
python advanced_secneuron_llm.py ^
  --device cpu ^
  --model facebook/opt-125m ^
  --train_samples 100 ^
  --eval_samples 50 ^
  --k_per_task 500 ^
  --lambda_penalty 0.3 ^
  --out_dir results_real
```

### 3.3 加入梯度敏感性

```powershell
python advanced_secneuron_llm.py ^
  --device cpu ^
  --model facebook/opt-125m ^
  --use_gradient ^
  --grad_max_batches 4 ^
  --grad_weight 0.35 ^
  --train_samples 60 ^
  --eval_samples 30 ^
  --k_per_task 300 ^
  --out_dir results_grad
```

### 3.4 运行 HumanEval 代码生成实验

默认只计算 `compile@k`，不执行生成代码。

```powershell
python advanced_secneuron_llm.py ^
  --device cpu ^
  --model facebook/opt-125m ^
  --run_humaneval ^
  --code_limit 5 ^
  --code_k 1 ^
  --out_dir results_humaneval
```

若要真实计算 `pass@k`，需要额外添加：

```powershell
--allow_code_execution
```

该参数会执行模型生成代码，建议只在隔离沙箱或虚拟机中运行。

---

## 4. 实验内容

| 实验 | 输出文件 | 说明 |
|---|---|---|
| 实验一：任务级能力控制 | `exp1_task_control.png` / `.csv` | 对比 baseline、admin、task-only、passive |
| 实验二：数据级滥用防护 | `exp2_real_pii_extraction.png` / `exp2_real_mia_loss_auroc.png` | PII extraction 和 loss-based MIA |
| 实验三：多任务动态权限控制 | `exp3_multitask_permissions_heatmap.png` | 多权限组合热力图 |
| 实验四：代码生成能力抑制 | `exp4_humaneval_compile_at_k.png` | HumanEval / MBPP 的 compile@k |
| 实验五：任务特异选择 vs 朴素剪枝 | `exp5_naive_vs_specific.png` | 横轴授权损失，纵轴未授权下降 |

---

## 5. 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `facebook/opt-125m` | 使用的小型 LLM |
| `--device` | `cpu` | 运行设备 |
| `--train_samples` | 60 | 每个任务用于统计神经元的重要性样本数 |
| `--eval_samples` | 30 | 每个任务用于测试的样本数 |
| `--k_per_task` | 80 | 每个任务选择的神经元数量 |
| `--lambda_penalty` | 0.8 | 跨任务重叠惩罚 |
| `--use_gradient` | False | 是否启用梯度敏感性 |
| `--grad_weight` | 0.35 | 梯度敏感性权重 |
| `--abe_backend` | `mock` | CP-ABE 后端，支持 `mock` 或 `charm` |
| `--run_humaneval` | False | 是否运行 HumanEval |
| `--run_mbpp` | False | 是否运行 MBPP |
| `--allow_code_execution` | False | 是否允许执行生成代码 |

---

## 6. 结果解读建议

如果实验效果不明显，通常不是代码错误，而可能是：

1. OPT-125M 太小；
2. `train_samples` 太少；
3. `k_per_task` 太小；
4. `lambda_penalty` 惩罚过强；
5. 任务能力主要依赖共享神经元；
6. token-level accuracy 对任务能力变化不够敏感。

建议尝试：

```powershell
--train_samples 100 --eval_samples 50 --k_per_task 500 --lambda_penalty 0.3
```

如果条件允许，再尝试：

```powershell
--model facebook/opt-350m
```

---

## 7. 与 SECNEURON 原文区别

| 项目 | SECNEURON 原文 | 本项目 |
|---|---|---|
| 模型 | OPT-6.7B、OPT-30B、Gemma-2-27B 等 | OPT-125M、OPT-350M |
| 数据集 | 多任务公开数据集 | HuggingFace 真实数据 + fallback demo |
| CP-ABE | Charm 实现 | mock / 可选 Charm |
| AES-CTR | 神经元参数加密 | 提供 roundtrip 验证 |
| 未解密神经元 | 检测 + 剪枝 | 临时剪枝模拟 |
| 实验目标 | 论文级复现 | 教学级、初步复现 |

---

## 8. 后续可扩展方向

1. 使用 LoRA 微调后再做神经元选择；
2. 将 OPT-125M 替换为 OPT-350M / OPT-1.3B；
3. 加入因果干预指标 \(C(t,n)\)；
4. 使用更大规模真实数据；
5. 用真实 Charm-Crypto 替换 mock CP-ABE；
6. 使用 HumanEval / MBPP 的真实 pass@k；
7. 设计更强 PII extraction 和 membership inference attack；
8. 将能力控制从神经元层扩展到 Adapter / LoRA 层。

## 9. Statement：
项目名称（Project Name)：SECNEURON \n
项目作者(Author)：Wenbin Wang，Zhen Chen
作者单位(Affiliation)：暨南大学网络空间安全学院(College of Cyber Security，Jinan University)
