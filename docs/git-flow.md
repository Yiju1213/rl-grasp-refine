# 1️⃣ commit message 关键词有哪些？

你提到的 feat / fix / refactor 是一套常见规范（类似 Conventional Commits）。你可以扩展成下面这套（够用了）：

## 1. 常用（建议你优先用）
- feat:      新功能（模型结构、模块新增）
- fix:       修bug（逻辑错误、数值错误）
- refactor:  重构（不改变功能）
## 2. 科研/工程强相关（你应该重点用）
- exp:       实验相关改动（非常重要）
- tune:      超参数调整
- data:      数据处理 / 数据集修改
- model:     模型结构调整（如果你不想用 feat）

👉 举例（更贴近你）：
```
exp: add contact-aware reward
tune: increase PPO clip range to 0.3
data: fix point cloud normalization scale
model: replace MLP with transformer encoder
```

## 3. 辅助类（工程整洁性）
- docs:      文档 / 注释
- style:     代码格式（不影响逻辑）
- chore:     杂事（依赖、脚本、小改动）
- test:      测试代码