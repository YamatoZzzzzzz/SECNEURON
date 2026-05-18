# 常见问题与解决方法

## 1. HuggingFace 下载慢或中断

设置镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

或使用 fallback demo：

```powershell
--allow_fallback_demo
```

---

## 2. `trust_remote_code is not supported anymore`

这是 datasets 库提示，不一定是错误。  
如果数据能加载，可以忽略。

---

## 3. 激活维度不匹配

如果出现：

```text
Expected activation with 3 dims, got shape=(256, 3072)
```

说明当前 transformers 版本中 MLP 激活可能是：

```text
[batch * seq_len, hidden]
```

代码中的 `align_activation_and_mask()` 已支持二维和三维激活。

---

## 4. 任务控制效果不明显

可能原因：

1. OPT-125M 太小；
2. 样本数太少；
3. `k_per_task` 太小；
4. \(\lambda\) 惩罚过强；
5. 任务能力主要依赖共享神经元。

建议尝试：

```powershell
--train_samples 100 --eval_samples 50 --k_per_task 500 --lambda_penalty 0.3
```

---

## 5. PII 图为空

可能原因：

1. Email 数据中没有被正则检测到 email / phone；
2. 模型没有记住这些 PII；
3. 模型续写没有泄露 PII。

严格的 PII 实验需要先用带 PII 样本微调模型。

---

## 6. MIA AUROC 没有下降

当前 member 样本不一定是真正被模型训练过的样本。  
严格 MIA 实验需要：

1. 先用 member 数据微调模型；
2. 再比较 member / non-member loss；
3. 计算 AUROC。

---

## 7. Charm-Crypto 安装失败

Windows 下 Charm-Crypto 安装困难。

建议：

- 日常实验使用 `--abe_backend mock`；
- 如果必须测试真实 CP-ABE，使用 WSL / Linux。
