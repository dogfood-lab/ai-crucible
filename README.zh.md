<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/dogfood-lab/ai-crucible/main/assets/logo.png" alt="ai-crucible" width="500" />
</p>

<p align="center">
  <a href="https://github.com/dogfood-lab/ai-crucible/actions/workflows/ci.yml"><img src="https://github.com/dogfood-lab/ai-crucible/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg" alt="Python 3.11–3.13" />
  <img src="https://img.shields.io/badge/coverage-94%25-brightgreen.svg" alt="Coverage 94%" />
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.4.0-orange.svg" alt="Version 0.4.0" /></a>
  <a href="https://dogfood-lab.github.io/ai-crucible/"><img src="https://img.shields.io/badge/docs-handbook-orange.svg" alt="Handbook" /></a>
</p>

<p align="center"><b>A diagnostic adversarial game for frontier LLMs — a measurement instrument that happens to be fun.</b></p>

一个 Claude 会话（**设计师**）会设计针对真实、当前存在的能力差距的难题。另一个（**求解器**）尝试解决这些难题。一个由策略控制的核心系统进行协调，根据隐藏的标准进行评分，并通过“实验室 → 竞技场 → 回归”生命周期来整理目录。难题基于经验数据——真实的 GitHub 问题、学术文献、观察到的现场故障——而不是人为生成的数据。

## 它与众不同之处

- **能力，而非“作弊”。** AI Crucible 区分*优雅性*和*新颖性*（给予奖励）与*绕过答案*（受到惩罚）。横向思维是一种可以衡量的能力，而不是应该被惩罚的缺点。
- **该工具本身进行测量。**提示构建是一个重要的衡量指标——核心系统在“中立”/“自我参照”/“社会地位”框架下运行相同的难题，并报告其自身的提示效果作为诊断结果。
- **一个封闭的测量边界。**动机和测量永远不会共享同一个上下文窗口；隐藏的标准由不同的模型家族进行评估，并且代理的推理过程是隐藏的。该模型无法利用它无法感知的事物来作弊。
- **通过一致性实现可靠性。**`pass^k`（所有 *k* 个独立试验都成功）、威尔逊区间和跨模型家族的评审小组——旨在报告分布情况，而不是点估计值。
- **一个动态且持久的目录。**运行结果会累积到各个会话中，并存储在一个基于事件源、哈希链的日志中（作为真实数据的来源）；分层状态是派生出来的投影。从“实验室”毕业到“竞技场”时，系统会考虑**弃权情况**——只有在获得可靠的跨模型家族的验证后才会进行晋升；否则，它会**升级到设计师**；饱和的难题会被**降级到“回归”，但不会被删除**（一种防止重复过程），因此该目录将成为能力演进的时间线，随着前沿不断发展。
- **差异化的回报。**对于每个难题，Claude 的解决率与跨模型家族的平均水平进行比较，从而确定差距——**Claude 特有**（价值最高）/ **LLM 通用** / **Claude 优势**——基于 Newcombe 差值区间，并具有一个重要的*不确定*类别，因此即使是小样本量的零结果也不会被误认为是发现。
- **可以衡量任何模型家族。**使用本机函数调用协议（而不仅仅是文本操作协议）的求解器属于第一类对象——返回工具调用的模型可以通过相同的受控沙盒进行求解，而不是返回文本操作。

## 威胁模型（摘要）

AI Crucible 的主要对手是**正在被测量的模型**，该模型可能会尝试找到答案或检测并利用评估机制。它会访问：难题提示/设置、求解器自己的受限工作目录以及操作员配置的模型提供商 API 调用。它不会访问：标准/答案（在单独的主机上进行评估，求解器无法访问）或动机“装饰”（排名/地位——绝不会注入到评分上下文中）。**权限：**模型提供商密钥通过运行时环境变量提供；不包含捆绑的秘密、遥测数据或任何自主的出站调用。完整的披露信息，包括哪些地方是*多层防御*而不是硬性保证，请参见 **[SECURITY.md](SECURITY.md)**。

## 架构

AI Crucible 是一个**构建在 [Inspect AI](https://inspect.aisi.org.uk/)（英国 AISI）之上的轻量级策略层**，而不是从零开始构建的框架。单个 `AttemptState` 对象会依次传递给设计师 → 求解器 →（评论者）→ 评审员，通过**一个 `generate` 瓶颈点**进行处理，因此可以观察到每个模型和工具调用。

| 模块 | 职责 |
| ------ | -------------- |
| `puzzle_loader` | 将难题目录（`meta.json`/`prompt`/`setup_script`）加载到求解器可见的状态中。**绝不会访问标准。** |
| `sandbox` | 将 `exec`/`read_file`/`write_file` 限制在一个锁定的、无网络连接的容器中。 |
| `roles` | 五个角色槽（设计师/求解器/评论者/评审员/群体求解器）。只有求解器可以使用工具；评论者是接口保留，默认关闭。 |
| `budget_governor` | 每个类别的工具调用 + 时钟预算，显示给代理，由核心系统强制执行；对于病态循环，会进行硬性终止。 |
| `oracle_scorer` | 在隐藏的标准下进行评估：解决**并且**没有回归（SWE-bench 模式）。 |
| `judge_panel` | 跨模型家族的评分小组 + 简化器 (PoLL)，用于验证新颖性和检测绕过行为。 |
| `trace_writer` | 每个尝试的记录都以 Inspect `EvalLog` 的形式存储；大型数据块通过摘要进行存储。 |
| `observability` | 每个尝试 → 每个难题 → 每个模型的汇总；本机支持 `pass^k`。 |
| `catalog` | 基于事件源的持久化存储 + “实验室 → 竞技场 → 回归”生命周期（考虑弃权情况的晋升，随时有效的饱和）+ 差异化分类。构建在 `attestation` 的哈希链日志之上。 |
| `attestation` | 通过类型化的子进程边界进行加密的溯源（cosign + 事件存储）。 |

封闭的边界运行在三个层级中——**第一层**是评分上下文（部署形状，框架中立），**第二层**是参与框架（每个发布时都会检查是否存在污染），**第三层**是装饰（排名/排行榜——仅供人类查看的用户界面，绝不会出现在模型解决问题的上下文中）。完整的设计原理，包括引用，请参见 [`docs/research-grounding.md`](docs/research-grounding.md)。

## 安装

```bash
# As a Python library + CLI (PyPI):
pip install ai-crucible          # or: uv pip install ai-crucible
ai-crucible --help

# Or zero-prerequisite via npx — downloads a verified binary, no Python needed:
npx @dogfood-lab/ai-crucible --help
```

**运行一个诊断周期**——求解器尝试在沙盒中解决难题，并在封闭的标准下进行评估，并输出 `pass^k`/威尔逊汇总结果：

```bash
# @family selects the adapter: no tag / @claude -> Claude (ANTHROPIC_API_KEY);
# any other @family -> a local Ollama model of that family (text OR native tool-calls).
ai-crucible run puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

每次运行都会**累积到持久目录中**。读取和整理它，或者运行评估感知边界探测：

```bash
ai-crucible catalog list                 # tiers + per-puzzle differential typology + health
ai-crucible catalog show <puzzle-id>     # one puzzle: runs, transition timeline, differential
ai-crucible catalog graduate             # preview Lab->Arena->Regression transitions (--apply to commit)

# Eval-awareness gate: does behaviour diverge between deploy- and test-framing?
ai-crucible probe puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

**离线工具质量测试**——不需要模型，也不需要 GPU，从已提交的运行报告开始：

```bash
# Forward-screen a less-saturated, still-defensible discriminating admission set
# from a characterization run's persisted grade matrix (the harder-set pipeline):
ai-crucible calibration curate --from-run report.json --out harder.json

# Validate a candidate human-label file before a --human-labels round (intake gate):
ai-crucible labels validate human_labels.json
```

> **研究预览（v0.4.x）。** 评审团的替代测试 ω 仍然是一个*循环模型-陪审团自举法*：验证它需要一轮**≥3名独立的标注人员**（参见[alt-test](https://arxiv.org/abs/2501.10970)），而单人工作室无法满足这一要求——因此，出于结构性限制而非疏忽，该环节暂时搁置。在位的评审员保持**临时状态**，当组成评审团的人数低于法定人数时，评审团会**升级为 Claude Designer**，并且该工具会公开这一点，而不是伪造人类参与的情况。请参阅[评分卡](SCORECARD.md)，以了解诚实、不带粉饰的评估结果。

## 快速入门（从源代码）

AI Crucible 使用 [`uv`](https://docs.astral.sh/uv/) 进行环境和依赖管理。Python **3.11+**。

```bash
# Create the venv and install the dev + stats extras
uv sync --extra dev --extra stats

# Run the test suite (with the coverage gate)
uv run pytest --cov=ai_crucible --cov-report=term-missing

# Lint
uv run ruff check .

# One command: lint + tests + build + smoke
bash verify.sh
```

## 跨家族评估

首次**公开的**跨家族评审员准入测试结果位于[`eval/RESULTS.md`](eval/RESULTS.md)（以及已提交的`eval/panel.json`和特征报告）。对七个不相交的家族进行了筛选——其中两个是本地家族（gemma4、granite4.1），五个是固定的 OpenRouter 端点（deepseek、cohere、meta-llama、qwen、nvidia），在 k=3 时，使用了 93 对校准数据：总共进行了 1,395 次付费调用，并且**没有出现任何速率限制问题**。

**诚实的结果是：**该池现在**允许 3 个不相交的家族**（从之前的仅限本地的 2 个家族增加），并且一个真正新的跨家族评审员已成功加入——但是，组成的*独立*评审团仍然只有**2 名成员**（第三名因错误冗余而被剔除，ρ≈1.0），这低于法定人数，因此评审团会**升级为 Claude Designer**，而不是自动做出决定。瓶颈最终变成了**未经验证的替代测试 ω 轴，而不是评审员质量**——有四位优秀的评审员（准确率 0.91–0.96）仅基于循环模型-陪审团 ω 进行筛选。“允许 3 个家族”是一个真正的进步；这**不是**“ω 问题已解决”。ω 仍然处于搁置状态，成员保持临时状态，最终结果仍需推迟——公开披露，而不是伪造。

## 文档

- **[手册](https://dogfood-lab.github.io/ai-crucible/)**——指南、架构和参考资料。
- [`docs/research-grounding.md`](docs/research-grounding.md)——设计原理，附带引用。
- [`docs/gameplan.md`](docs/gameplan.md)——路线图和未解决的问题。
- [`SECURITY.md`](SECURITY.md)——威胁模型 + 诚实地披露剩余风险。

## 许可

[MIT](LICENSE)。公开且为预发布版本 1.0——请参阅[CHANGELOG](CHANGELOG.md），了解版本状态。

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
