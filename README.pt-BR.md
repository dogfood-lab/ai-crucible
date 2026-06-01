<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

Uma sessão do Claude (**Designer**) cria desafios que visam lacunas de capacidade reais e atualmente observadas. Outra (**Solver**) tenta resolvê-los. Um kernel, com políticas aplicadas, faz a mediação, avalia com base em um oráculo oculto e organiza um catálogo por meio de um ciclo de vida `Lab → Arena → Regression`. Os desafios são baseados em dados empíricos — problemas reais do GitHub, literatura acadêmica, falhas observadas no campo — e não em dados sintéticos.

## O que o torna diferente

- **Capacidade, não "trapaça".** O Crucible distingue *elegância* e *novidade* (recompensadas) de *desvio da resposta* (penalizado). O pensamento lateral é uma capacidade a ser medida, não um defeito a ser punido.
- **O instrumento mede a si mesmo.** A formulação do prompt é um componente de medição de primeira classe — o kernel executa o mesmo desafio sob formulações `neutra` / `autorreferencial` / `social_standings` e relata seu próprio efeito do prompt como um diagnóstico.
- **Uma fronteira de medição selada.** A motivação e a medição nunca compartilham uma janela de contexto; o oráculo oculto é avaliado fora da banda por uma família de modelos diferente, com o raciocínio do agente oculto. O modelo não pode manipular o que não consegue perceber.
- **Confiabilidade por consistência.** `pass^k` (todas as *k* tentativas independentes têm sucesso), intervalos de Wilson e painéis de avaliação inter-familiares — criados para relatar distribuições, não estimativas pontuais.
- **Um catálogo dinâmico.** Os desafios resolvidos são *rebaixados* para `Regression`, nunca excluídos, para que o catálogo se torne uma linha do tempo da evolução das capacidades à medida que a fronteira avança.

## Modelo de ameaças (resumo)

O principal adversário do Crucible é o **modelo em avaliação**, que pode tentar acessar a chave de resposta ou detectar e manipular a avaliação. Ele **acessa**: prompts/configuração do desafio, o próprio diretório de trabalho restrito do Solver e as chamadas da API do provedor do modelo que o operador configura. Ele **não** acessa: o oráculo/chave de resposta (avaliado fora da banda em um host separado que o Solver não pode acessar) ou elementos de motivação ("chrome" — classificação/posição — nunca injetados no contexto avaliado). **Permissões:** chaves do provedor do modelo por meio de variáveis de ambiente em tempo de execução; sem segredos embutidos, sem telemetria, sem chamadas externas próprias. Divulgação completa — incluindo onde uma fronteira é uma *defesa em profundidade* em vez de uma garantia absoluta — está em **[SECURITY.md](SECURITY.md)**.

## Arquitetura

O Crucible é uma **camada de política fina sobre [Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI), não um conjunto de ferramentas criado do zero. Um único objeto `AttemptState` é passado de Designer → Solver → (Critic) → Judge por meio de **um único ponto de estrangulamento `generate`**, para que cada chamada de modelo e ferramenta seja observável.

| Módulo | Responsabilidade |
| ------ | -------------- |
| `puzzle_loader` | Carrega um diretório de desafios (`meta.json` / `prompt` / `setup_script`) no estado visível ao Solver. **Nunca acessa o oráculo.** |
| `sandbox` | Canal restrito `exec` / `read_file` / `write_file` em um contêiner bloqueado e sem rede. |
| `roles` | Os cinco slots de função (Designer / Solver / Critic / Judge / CohortSolver). Apenas o Solver tem ferramentas; o Critic é reservado para a interface, desativado por padrão. |
| `budget_governor` | Chamada de ferramenta por classe + orçamentos de tempo decorrido, exibidos ao agente, aplicados no lado do kernel; interrupção forçada em loops patológicos. |
| `oracle_scorer` | Avaliação fora da banda: resolvido **e** sem regressão em relação ao oráculo oculto (padrão SWE-bench). |
| `judge_panel` | Painel inter-família de avaliadores de modelo + redutor (PoLL) para validação de novidade e detecção de desvio. |
| `trace_writer` | Transcrição por tentativa no formato Inspect `EvalLog`; grandes blocos armazenados por hash. |
| `observability` | Agregações por tentativa → por desafio → por modelo; `pass^k` nativo. |
| `attestation` | Provável criptográfica (cosign + event-store) por trás de uma fronteira de subprocesso tipada. |

A fronteira selada é executada em três níveis — **Nível 1** contexto avaliado (moldado pela implantação, neutro em termos de formulação), **Nível 2** formulação de engajamento (verificado quanto à contaminação em cada lançamento), **Nível 3** elementos visuais (classificação/tabela de classificação — interface voltada para o usuário, nunca em um contexto no qual o modelo resolve). A justificativa completa do projeto, com citações, está em [`docs/research-grounding.md`](docs/research-grounding.md).

## Início rápido

O Crucible usa [`uv`](https://docs.astral.sh/uv/) para gerenciamento de ambiente e dependências. Python **3.11+**.

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

## Documentação

- **[Handbook](https://dogfood-lab.github.io/crucible/)** — guias, arquitetura e referência.
- [`docs/research-grounding.md`](docs/research-grounding.md) — justificativa do projeto, com citações.
- [`docs/gameplan.md`](docs/gameplan.md) — roteiro e questões em aberto.
- [`SECURITY.md`](SECURITY.md) — modelo de ameaças + divulgação honesta dos riscos residuais.

## Licença

[MIT](LICENSE). Público e pré-1.0 — consulte o [CHANGELOG](CHANGELOG.md) para o status da versão.

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
