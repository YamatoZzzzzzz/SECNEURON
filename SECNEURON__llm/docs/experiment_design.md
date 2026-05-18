# 实验设计说明

## 实验一：任务级能力控制

目标是验证：

\[
\text{授权任务保留，未授权任务下降}
\]

角色包括：

- baseline；
- admin；
- story_only；
- health_only；
- code_only；
- passive。

理想趋势：

\[
Acc_{\text{admin}}\approx Acc_{\text{baseline}}
\]

\[
Acc_{\text{task-only}}(\text{authorized}) \text{ 高}
\]

\[
Acc_{\text{task-only}}(\text{unauthorized}) \text{ 低}
\]

---

## 实验二：数据级滥用防护

包括 PII extraction 和 MIA。

### PII extraction

流程：

1. 从 Email 文本中抽取 email / phone；
2. 给模型输入 PII 前面的文本前缀；
3. 检查模型续写是否出现真实 PII。

指标：

\[
PII\ Success\ Rate=
\frac{\#\text{泄露PII样本}}{\#\text{总样本}}
\]

### MIA

流程：

1. member 文本来自用于统计的重要性样本；
2. non-member 文本来自额外 hold-out 样本；
3. 计算每个样本的语言模型 loss；
4. 用 \(-loss\) 作为成员分数；
5. 计算 AUROC。

---

## 实验三：多任务动态权限控制

测试同一个模型在不同权限组合下的任务表现。

输出为热力图。

---

## 实验四：代码生成能力抑制

使用 HumanEval / MBPP。

默认指标：

\[
compile@k
\]

可选指标：

\[
pass@k
\]

---

## 实验五：任务特异选择 vs 朴素剪枝

比较：

\[
I(t,n)
\]

和：

\[
S(T,t,n)=I(t,n)-\lambda\max_{t'\neq t}I(t',n)
\]

图中：

- 横轴：授权任务 accuracy drop，越小越好；
- 纵轴：未授权 Code accuracy drop，越大越好。
