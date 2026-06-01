<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="assets/logo.png" alt="crucible" width="400" />
</p>

<p align="center">
  <a href="https://github.com/dogfood-lab/crucible/actions/workflows/ci.yml"><img src="https://github.com/dogfood-lab/crucible/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg" alt="Python 3.11–3.13" />
  <img src="https://img.shields.io/badge/coverage-96%25-brightgreen.svg" alt="Coverage 96%" />
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.2.0-orange.svg" alt="Version 0.2.0" /></a>
  <a href="https://dogfood-lab.github.io/crucible/"><img src="https://img.shields.io/badge/docs-handbook-orange.svg" alt="Handbook" /></a>
</p>

<p align="center"><b>A diagnostic adversarial game for frontier LLMs — a measurement instrument that happens to be fun.</b></p>

一个 Claude 会话（**设计者**）会设计针对真实、当前存在的能力差距的谜题。另一个（**求解者**）尝试解决这些谜题。一个由策略控制的内核会进行调解，根据隐藏的基准进行评分，并通过“实验室 → 竞技场 → 回归”的生命周期来整理一个目录。谜题是基于经验信号——真实的 GitHub 问题、学术文献、观察到的现场故障——而不是人为设计的。

## 它与众不同之处

- **能力，而不是“作弊”。** Crucible 区分*优雅*和*新颖*（给予奖励）与*绕过答案*（给予惩罚）。横向思维是一种可以衡量的能力，而不是一种应该惩罚的缺点。
- **该工具会自我评估。** 提示构建是一种一级可测量的手段——内核会在“中性”/“自我参照”/“社会地位”的提示框架下运行相同的谜题，并报告其自身的提示效果，作为一种诊断。
- **一个封闭的测量边界。** 动机和测量永远不会共享同一个上下文窗口；隐藏的基准由不同的模型系列进行评估，并且隐藏了代理的推理过程。模型无法利用它无法感知的事物。
- **通过一致性实现可靠性。** `pass^k`（所有 *k* 个独立的试验都成功）、威尔逊区间和跨模型系列评审小组——旨在报告分布，而不是点估计。
- **一个不断发展的目录。** 解决的谜题会被*降级*到“回归”类别，但不会被删除，因此该目录会成为一个能力演进时间线，随着前沿的推进而变化。

## 威胁模型（摘要）

Crucible 的主要对手是**正在评估的模型**，它可能会尝试找到答案或检测并利用评估机制。它会访问：谜题提示/设置、求解器自己的受限工作目录以及模型提供商 API 调用（由操作员配置）。它不会访问：基准/答案（在单独的主机上进行评估，求解器无法访问）或动机“装饰”（排名/地位——绝不会注入到评分上下文中）。**权限：**通过运行时环境变量提供的模型提供商密钥；没有捆绑的密钥、没有遥测数据、没有自主的出站调用。完整的披露——包括在何处边界是*多层防御*，而不是硬性保证——请参见 **[SECURITY.md](SECURITY.md)**。

## 架构

Crucible 是一个**构建在 [Inspect AI](https://inspect.aisi.org.uk/)（英国 AISI）之上的轻量级策略层**，而不是从头开始构建的框架。一个 `AttemptState` 对象会从设计者传递到求解者，再到（评论者），最后到评审者，通过**一个 `generate` 瓶颈点**，因此可以观察到所有模型和工具调用。

| 模块 | 职责 |
| ------ | -------------- |
| `puzzle_loader` | 将一个谜题目录（`meta.json` / `prompt` / `setup_script`）加载到求解器可见的状态中。**绝不会访问基准。** |
| `sandbox` | 将 `exec` / `read_file` / `write_file` 限制在一个锁定的、无网络连接的容器中。 |
| `roles` | 五个角色槽（设计者 / 求解者 / 评论者 / 评审者 / 协同求解者）。只有求解者可以使用工具；评论者是界面保留的，默认关闭。 |
| `budget_governor` | 每个类别的工具调用 + 时钟预算，显示给代理，由内核强制执行；对于病态循环，会强制终止。 |
| `oracle_scorer` | 在基准之外进行评估：解决谜题**并且**没有回归，与隐藏的基准进行比较（SWE-bench 模式）。 |
| `judge_panel` | 由跨模型系列的评分者 + 简化器 (PoLL) 组成的评审小组，用于验证新颖性并检测绕过行为。 |
| `trace_writer` | 每个尝试的转录记录，采用 Inspect `EvalLog` 格式；大型数据块通过摘要进行存储。 |
| `observability` | 每个尝试 → 每个谜题 → 每个模型的汇总；`pass^k` 原生支持。 |
| `attestation` | 基于密码学的溯源（cosign + event-store），位于一个带类型的子进程边界之后。 |

封闭的边界在三个层级中运行——**第一层**是评分上下文（部署形状，提示中性），**第二层**是参与框架（在每次发布时都会检查是否存在污染），**第三层**是装饰（排名/排行榜——仅用于人机界面，绝不会出现在模型解决问题的上下文中）。完整的设计原理，包括引用，请参见 [`docs/research-grounding.md`](docs/research-grounding.md)。

## 快速入门

Crucible 使用 [`uv`](https://docs.astral.sh/uv/) 进行环境和依赖管理。Python **3.11+**。

```bash
# Create the venv and install the dev + stats extras
uv sync --extra dev --extra stats

# Run the test suite (with the coverage gate)
uv run pytest --cov=crucible --cov-report=term-missing

# Lint
uv run ruff check .

# One command: lint + tests + build + smoke
bash verify.sh
```

## 文档

- **[手册](https://dogfood-lab.github.io/crucible/)**——指南、架构和参考。
- [`docs/research-grounding.md`](docs/research-grounding.md)——设计原理，包括引用。
- [`docs/gameplan.md`](docs/gameplan.md)——路线图和未解决的问题。
- [`SECURITY.md`](SECURITY.md)——威胁模型 + 诚实的剩余风险披露。

## 许可证

[MIT](LICENSE)。公开且为 1.0 之前的版本——请参见 [CHANGELOG](CHANGELOG.md)，了解版本状态。

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
