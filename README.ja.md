<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/dogfood-lab/ai-crucible/main/assets/logo.png" alt="ai-crucible" width="500" />
</p>

<p align="center">
  <a href="https://github.com/dogfood-lab/ai-crucible/actions/workflows/ci.yml"><img src="https://github.com/dogfood-lab/ai-crucible/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg" alt="Python 3.11–3.13" />
  <img src="https://img.shields.io/badge/coverage-94%25-brightgreen.svg" alt="Coverage 94%" />
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.3.0-orange.svg" alt="Version 0.3.0" /></a>
  <a href="https://dogfood-lab.github.io/ai-crucible/"><img src="https://img.shields.io/badge/docs-handbook-orange.svg" alt="Handbook" /></a>
</p>

<p align="center"><b>A diagnostic adversarial game for frontier LLMs — a measurement instrument that happens to be fun.</b></p>

ある Claude セッション（**デザイナー**）は、現実の、現在確認されている能力ギャップをターゲットとしたパズルを作成します。別のセッション（**ソルバー**）がそれらを試みます。ポリシーによって制御されるカーネルが仲介し、隠された基準に対してスコアリングを行い、`Lab → Arena → Regression` ライフサイクルを通じてカタログをキュレーションします。パズルは、経験的なシグナルに基づいています。つまり、実際の GitHub の問題、学術論文、現場で観察された失敗などであり、人工的なものではありません。

## 何が異なるのか

- **能力、ではなく「不正行為」。** AI Crucible は、*エレガントさ* と *新規性*（報酬対象）を、*答えの回避*（ペナルティ対象）と区別します。横方向思考は測定可能な能力であり、処罰すべき欠点ではありません。
- **評価ツール自体が自己評価を行う。** プロンプトの構成は、主要な測定項目です。カーネルは、同じパズルを `neutral` / `self_referential` / `social_standings` の各プロンプトで実行し、そのプロンプトの効果を診断として報告します。
- **密閉された評価境界。** モチベーションと評価は、決して同じコンテキストウィンドウを共有しません。隠された基準は、ソルバーがアクセスできない別のホスト上で、エージェントの推論を非表示にした状態で、別のモデルファミリーによって評価されます。モデルは、認識できないものを操作することはできません。
- **一貫性による信頼性。** `pass^k`（すべての *k* 回の独立した試行で成功）、ウィルソンの区間、およびクロスファミリーの審査員パネルを組み込み、分布を報告するように設計されており、点推定値ではありません。
- **進化し続ける、永続的なカタログ。** 実行結果は、イベントソース化されたハッシュチェーンログ（信頼できる情報源）にセッション間で蓄積されます。ティアの状態は、そこから派生した投影です。`Lab → Arena` への移行は、**棄権を考慮**します。つまり、クロスファミリーの確かな評価に基づいてのみ昇格し、そうでない場合は **デザイナーにエスカレーション** されます。飽和状態になったパズルは **`Regression` に降格され、削除されることはありません**（不正行為防止のためのプロセス）。そのため、カタログは能力進化のタイムラインとなり、最先端が変化します。
- **差分ペイオフ。** パズルごとに、Claude の解答率とクロスファミリーコホートを比較し、ギャップを分類します。**Claude 固有**（最も高い価値）/ **LLM 全般** / **Claude の強み** となります。これは、一次的な *不確実* クラスを持つニューカムの差分区間に基づいており、少数のデータから得られた結果が誤って肯定的に解釈されることはありません。
- **あらゆるモデルファミリーを評価可能。** ネイティブの関数呼び出しプロトコル（テキストアクションプロトコルだけでなく）を使用するソルバーは、主要な対象です。テキストアクションではなくツールコールを返すモデルも、同じ制御されたサンドボックスを通じて解答します。

## 脅威モデル（概要）

AI Crucible の主な敵対者は、**評価対象のモデル**であり、答えにアクセスしようとしたり、評価を検出し操作しようとする可能性があります。これは、パズルのプロンプト/設定、ソルバー自身の隔離された作業ディレクトリ、およびオペレーターが構成するモデルプロバイダー API 呼び出しに影響を与えます。ただし、以下のものには影響を与えません。オラクル/答え（別のホスト上で評価され、ソルバーはアクセスできません）、またはモチベーション「クローム」（ランク/ランキング - モデルが解答に使用するコンテキストには決して注入されません）。**権限：**実行時に環境変数を通じてモデルプロバイダーキーを付与します。バンドルされたシークレット、テレメトリ、独自の外部呼び出しはありません。完全な情報開示（境界線が *多層防御* であるか、それとも厳格な保証であるかを明記）は、**[SECURITY.md](SECURITY.md)** に記載されています。

## アーキテクチャ

AI Crucible は、**[Inspect AI](https://inspect.aisi.org.uk/)（英国 AISI）上の薄いポリシーレイヤー**であり、ゼロから構築されたフレームワークではありません。単一の `AttemptState` オブジェクトが、デザイナー → ソルバー → (批評家) → 審査員 に渡され、**1 つの `generate` チェックポイント** を通過するため、すべてのモデルとツール呼び出しを監視できます。

| モジュール | 責任 |
| ------ | -------------- |
| `puzzle_loader` | パズルディレクトリ（`meta.json` / `prompt` / `setup_script`）をソルバーがアクセスできる状態にロードします。**オラクルには決して触れません。** |
| `sandbox` | ロックされた、ネットワーク接続のないコンテナ内に `exec` / `read_file` / `write_file` チャンネルを制限します。 |
| `roles` | 5 つのロール（デザイナー/ソルバー/批評家/審査員/コホートソルバー）。ツールを使用できるのはソルバーのみです。批評家はインターフェース予約されており、デフォルトではオフになっています。 |
| `budget_governor` | クラスごとのツール呼び出し + 壁時計時間予算をエージェントに表示し、カーネル側で強制します。異常なループが発生した場合は強制終了します。 |
| `oracle_scorer` | バンド外評価：隠されたオラクルに対して、解答 **かつ** 回帰がないことを確認（SWE ベンチパターン）。 |
| `judge_panel` | 新規性の検証と回避の検出のための、クロスファミリーのモデルスコアリングパネル + リデューサー（PoLL）。 |
| `trace_writer` | 試行ごとのトランスクリプトを Inspect の `EvalLog` 形式で保存します。大きなデータはダイジェストによって保存されます。 |
| `observability` | 試行ごと → パズルごと → モデルごとの集計。`pass^k` をネイティブにサポートします。 |
| `catalog` | イベントソース化された永続的なストレージ + `Lab → Arena → Regression` ライフサイクル（棄権を考慮した昇格、いつでも有効な飽和状態）+ 差分タイポロジー。`attestation` のハッシュチェーンログに基づいています。 |
| `attestation` | 暗号化されたプロビナンス（cosign + イベントストア）を、型付きのサブプロセス境界の後ろに配置します。 |

密閉された境界は、3 つのティアで実行されます。**ティア 1：** スコアリング対象のコンテキスト（デプロイメント形状、プロンプトの影響を受けない）、**ティア 2：** エンゲージメントプロンプト（各リリースで汚染がないか確認）、**ティア 3：** クローム（ランク/リーダーボード - 人間が閲覧する UI のみ、モデルが解答に使用するコンテキストには決して含まれません）。完全な設計の根拠と引用は、[`docs/research-grounding.md`](docs/research-grounding.md) に記載されています。

## インストール

```bash
# As a Python library + CLI (PyPI):
pip install ai-crucible          # or: uv pip install ai-crucible
ai-crucible --help

# Or zero-prerequisite via npx — downloads a verified binary, no Python needed:
npx @dogfood-lab/ai-crucible --help
```

**1 つの診断サイクルを実行します。** ソルバーがサンドボックス内でパズルを試行し、密閉されたオラクルに対してバンド外で評価され、`pass^k` / ウィルソンの集計結果が出力されます。

```bash
# @family selects the adapter: no tag / @claude -> Claude (ANTHROPIC_API_KEY);
# any other @family -> a local Ollama model of that family (text OR native tool-calls).
ai-crucible run puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

各実行は **永続的なカタログに蓄積されます。** カタログを参照およびキュレーションするか、評価認識境界プローブを実行します。

```bash
ai-crucible catalog list                 # tiers + per-puzzle differential typology + health
ai-crucible catalog show <puzzle-id>     # one puzzle: runs, transition timeline, differential
ai-crucible catalog graduate             # preview Lab->Arena->Regression transitions (--apply to commit)

# Eval-awareness gate: does behaviour diverge between deploy- and test-framing?
ai-crucible probe puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

**リサーチプレビュー（v0.3.x）。** 審査員団による代替テストωは、依然として*循環モデル・陪審員ブートストラップ*であり、その妥当性を検証するには、**3人以上の独立した人間アノテーター**が必要（[alt-test](https://arxiv.org/abs/2501.10970)を参照）。単独の人間によるスタジオでは対応できないため、この段階は**構造的な制約により一時停止されており、放置されているわけではない**。審査員は**仮の状態**を保ち、構成された審査団は定足数に達しない場合、**Claude Designerに移行**し、そのツールは人間の判断を偽装するのではなく、その状況を開示する。正直で、見かけ倒しの結果ではないゲートの結果については、[スコアカード](SCORECARD.md)を参照のこと。

## クイックスタート（ソースから）

AI Crucibleは、環境および依存関係の管理に[`uv`](https://docs.astral.sh/uv/)を使用します。Python **3.11以上**が必要です。

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

## ドキュメント

- **[ハンドブック](https://dogfood-lab.github.io/ai-crucible/)** — ガイド、アーキテクチャ、およびリファレンス。
- [`docs/research-grounding.md`](docs/research-grounding.md) — 設計の根拠と参考文献。
- [`docs/gameplan.md`](docs/gameplan.md) — ロードマップと未解決の問題。
- [`SECURITY.md`](SECURITY.md) — 脅威モデル + 正直な残存リスクの開示。

## ライセンス

[MIT](LICENSE)。公開されており、バージョン1.0以前です。バージョン状況については、[CHANGELOG](CHANGELOG.md)を参照してください。

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
