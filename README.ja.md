<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

ある Claude セッション（**デザイナー**）は、現実の、現在確認されている能力のギャップをターゲットとしたパズルを作成します。別のセッション（**ソルバー**）がそれらを解こうとします。ポリシーによって制御されるカーネルが、隠された基準と照らし合わせて評価を行い、`Lab → Arena → Regression` のライフサイクルを通じてカタログを管理します。パズルは、実際の GitHub の問題、学術論文、現場で観察された失敗など、経験的なデータに基づいており、人工的なデータではありません。

## 何が違うのか

- **能力、ではなく「不正行為」。** Crucible は、*エレガントさ*と*斬新さ*（評価される）と、*答えを回避する*（ペナルティが科される）を区別します。横方向思考は、測定すべき能力であり、処罰すべき欠点ではありません。
- **測定ツール自体が測定する。** プロンプトの構成は、主要な測定項目です。カーネルは、同じパズルを `neutral` / `self_referential` / `social_standings` の異なる構成で実行し、そのプロンプトの効果を診断として報告します。
- **密閉された測定境界。** モチベーションと測定は、同じコンテキストウィンドウを共有しません。隠された基準は、別のモデルファミリーによって評価され、エージェントの推論は隠されています。モデルは、認識できないものを操作することはできません。
- **一貫性による信頼性。** `pass^k`（すべての *k* 回の独立した試行が成功）、ウィルソンの区間、およびクロスファミリーの審査員パネル。これらは、点推定値ではなく、分布を報告するように設計されています。
- **進化し続けるカタログ。** 解決されたパズルは、削除されることなく `Regression` に*降格*されるため、カタログは能力の進化のタイムラインとなり、最先端が変化します。

## 脅威モデル（概要）

Crucible の主な敵対者は、**測定対象のモデル**であり、これは答えのキーにアクセスしようとしたり、評価を検出し、それを操作しようとしたりする可能性があります。これは、パズルのプロンプト/設定、ソルバー自身の隔離された作業ディレクトリ、およびオペレーターが構成するモデルプロバイダー API 呼び出しに**アクセス**します。**アクセスしない**のは、オラクル/答えのキー（別のホストで評価され、ソルバーがアクセスできない）、またはモチベーション「クローム」（ランク/ランキング - 評価されるコンテキストには決して注入されない）です。**権限:** 実行時の環境変数を通じてモデルプロバイダーのキーが提供されます。バンドルされた秘密、テレメトリ、または独自の外部呼び出しはありません。完全な情報開示（境界が、厳密な保証ではなく、多層防御である場所を含む）は、**[SECURITY.md](SECURITY.md)** にあります。

## アーキテクチャ

Crucible は、**[Inspect AI](https://inspect.aisi.org.uk/)**（英国 AISI）上の**薄いポリシーレイヤー**であり、ゼロから構築されたフレームワークではありません。単一の `AttemptState` オブジェクトが、デザイナー → ソルバー →（批評家）→ 審査員へと、**1 つの `generate` 処理ポイント**を通じてスレッド化されるため、すべてのモデルとツールの呼び出しを監視できます。

| モジュール | 責任 |
| ------ | -------------- |
| `puzzle_loader` | パズルのディレクトリ（`meta.json` / `prompt` / `setup_script`）を、ソルバーがアクセスできる状態にロードします。**オラクルには決してアクセスしません。** |
| `sandbox` | `exec` / `read_file` / `write_file` へのアクセスを、ロックされた、ネットワークに接続されていないコンテナに制限します。 |
| `roles` | 5 つのロールスロット（デザイナー / ソルバー / 批評家 / 審査員 / コホートソルバー）。ツールを使用できるのはソルバーのみで、批評家はインターフェース予約されており、デフォルトではオフになっています。 |
| `budget_governor` | クラスごとのツール呼び出し + 壁時計による時間制限。エージェントに表示され、カーネル側で強制されます。異常なループが発生した場合は強制終了します。 |
| `oracle_scorer` | バンド外評価：隠されたオラクルに対して、解決済み**かつ**リグレッションが発生していないことを確認します（SWE-bench パターン）。 |
| `judge_panel` | 斬新さの検証と回避の検出のための、クロスファミリーのモデルスコアラー + リデューサー（PoLL）パネル。 |
| `trace_writer` | 試行ごとのトランスクリプトは、Inspect の `EvalLog` 形式で保存され、大きなデータはダイジェストによって保存されます。 |
| `observability` | 試行ごと → パズルごと → モデルごとの集計。`pass^k` をネイティブにサポートします。 |
| `attestation` | 暗号化されたプロビナンス（cosign + イベントストア）は、型付きのサブプロセス境界の背後にあります。 |

密閉された境界は、3 つの階層で実行されます。**Tier 1** は、評価されるコンテキスト（デプロイメントの形状、構成に依存しない）、**Tier 2** は、エンゲージメントの構成（各リリースで汚染がないか確認）、**Tier 3** は、クローム（ランク/リーダーボード - 人間が操作する UI のみ、モデルが解決するコンテキストには決して含まれない）です。完全な設計の根拠と引用は、[`docs/research-grounding.md`](docs/research-grounding.md) にあります。

## クイックスタート

Crucible は、環境と依存関係の管理に [`uv`](https://docs.astral.sh/uv/) を使用します。Python **3.11 以降**。

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

## ドキュメント

- **[Handbook](https://dogfood-lab.github.io/crucible/)** - ガイド、アーキテクチャ、およびリファレンス。
- [`docs/research-grounding.md`](docs/research-grounding.md) - 設計の根拠と引用。
- [`docs/gameplan.md`](docs/gameplan.md) - ロードマップと未解決の課題。
- [`SECURITY.md`](SECURITY.md) - 脅威モデル + 正直な残余リスクの開示。

## ライセンス

[MIT](LICENSE)。公開されており、バージョン 1.0 より前です。バージョンステータスについては、[CHANGELOG](CHANGELOG.md) を参照してください。

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
