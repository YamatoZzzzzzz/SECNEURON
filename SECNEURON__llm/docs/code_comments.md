# 代码中文注释规范

## 1. 文件头注释

每个代码文件开头应说明：

- 文件功能；
- 对应论文模块；
- 输入输出；
- 是否为官方实现。

示例：

```python
"""
本文件实现 SECNEURON 风格的 LLM 任务级能力控制实验。

主要功能：
1. 加载真实任务数据；
2. 统计任务神经元激活；
3. 计算任务特异神经元分数；
4. 模拟 CP-ABE 权限控制；
5. 通过剪枝模拟未授权神经元失效；
6. 输出任务级与数据级实验结果。

注意：
这不是 SECNEURON 官方实现，而是教学版复现实验脚手架。
"""
```

## 2. 函数注释格式

建议每个函数包含：

- 函数作用；
- 输入参数；
- 返回值；
- 和论文方法的对应关系。

## 3. 关键公式注释

建议在代码中直接写明公式。

```python
# 任务特异分数：
# S(T,t,n)=I(t,n)-lambda*max_{t'!=t}I(t',n)
scores[t][k] = I[t][k] - lambda_penalty * max_other
```

如果加入梯度项：

```python
# 加入梯度敏感性后的扩展形式：
# S=I+beta*G-lambda*O
scores[t][k] = I[t][k] + grad_weight * G[t][k] - lambda_penalty * max_other
```

## 4. 实验输出注释

```python
# 实验一：任务级能力控制
# 目标：验证授权任务保留、未授权任务下降。
run_exp1_task_control(...)
```

## 5. 安全警告注释

```python
# WARNING:
# allow_code_execution=True 会执行模型生成的代码。
# 该操作存在安全风险，只建议在沙箱或虚拟机中运行。
```
