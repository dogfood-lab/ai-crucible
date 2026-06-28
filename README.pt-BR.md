<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

Uma sessão do Claude (**Designer**) cria quebra-cabeças com o objetivo de abordar lacunas de capacidade reais e atualmente observadas. Outro (**Solver**) tenta resolvê-los. Um kernel com políticas aplicadas atua como intermediário, avalia os resultados em relação a um oráculo oculto e organiza um catálogo por meio de um ciclo de vida `Lab → Arena → Regression`. Os quebra-cabeças são baseados em dados empíricos — problemas reais do GitHub, literatura acadêmica, falhas observadas no campo — e não em dados sintéticos.

## O que o torna diferente?

- **Capacidade, não "trapaça".** O AI Crucible distingue *elegância* e *novidade* (recompensadas) de *desvio da resposta* (penalizado). O pensamento lateral é uma capacidade a ser medida, não um defeito a ser punido.
- **O instrumento mede a si mesmo.** A formulação do prompt é um elemento medido de primeira classe — o kernel executa o mesmo quebra-cabeça sob formulações `neutra` / `autorreferencial` / `social_standings` e relata seu próprio efeito do prompt como um diagnóstico.
- **Uma fronteira de medição selada.** A motivação e a medição nunca compartilham uma janela de contexto; o oráculo oculto é avaliado externamente por uma família diferente de modelos, com o raciocínio do agente oculto. O modelo não pode manipular algo que não consegue perceber.
- **Confiabilidade pela consistência.** `pass^k` (todas as *k* tentativas independentes têm sucesso), intervalos de Wilson e painéis de avaliação inter-familiares — criados para relatar distribuições, não estimativas pontuais.
- **Um catálogo dinâmico e duradouro.** As execuções são acumuladas em sessões em um log com rastreamento de eventos e encadeamento por hash (a fonte da verdade); o estado da camada é uma projeção derivada. A progressão `Lab → Arena` leva em consideração a **abstenção** — promove apenas com base em um veredicto confiável entre famílias diferentes e, caso contrário, **avança para o Designer**; quebra-cabeças saturados são **rebaixados para `Regression`, nunca excluídos** (um processo eletrônico anti-repetitivo), de modo que o catálogo se torna uma linha do tempo da evolução das capacidades à medida que a fronteira avança.
- **O benefício diferencial.** Por quebra-cabeça, a taxa de resolução do Claude em comparação com a coorte inter-familiar classifica a lacuna — **específica do Claude** (valor mais alto) / **geral para LLM** / **ponto forte do Claude** — com base em um intervalo de diferença de Newcombe com uma classe *inconclusiva* de primeira classe, para que um resultado nulo pequeno nunca seja disfarçado como uma descoberta.
- **Mede qualquer família de modelos.** Os solvers que usam o protocolo nativo de chamada de função (e não apenas o protocolo de ação de texto) são considerados elementos de primeira classe — um modelo que retorna chamadas de ferramentas em vez de ações de texto resolve por meio do mesmo ambiente controlado.

## Modelo de ameaças (resumo)

O principal adversário do AI Crucible é o **modelo sob medição**, que pode tentar acessar a chave de resposta ou detectar e manipular a avaliação. Ele **acessa**: prompts/configuração dos quebra-cabeças, o próprio diretório de trabalho restrito do Solver e as chamadas da API do provedor de modelos que o operador configura. Ele **não acessa**: o oráculo/chave de resposta (avaliado externamente em um host separado ao qual o Solver não pode acessar) ou elementos motivacionais ("aparência" — classificação/posição — nunca injetados no contexto avaliado). **Permissões:** chaves do provedor de modelos por meio de variáveis de ambiente em tempo de execução; sem segredos incluídos, sem telemetria, sem chamadas externas próprias. Divulgação completa — incluindo onde uma fronteira é *defesa em profundidade* em vez de uma garantia rígida — está em **[SECURITY.md](SECURITY.md)**.

## Arquitetura

O AI Crucible é uma **camada de política fina sobre [Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI), e não um conjunto de ferramentas criado do zero. Um único objeto `AttemptState` é transmitido do Designer → Solver → (Crítico) → Avaliador por meio de **um único ponto de estrangulamento `generate`**, para que cada chamada de modelo e ferramenta seja observável.

| Módulo | Responsabilidade |
| ------ | -------------- |
| `puzzle_loader` | Carrega um diretório de quebra-cabeças (`meta.json` / `prompt` / `setup_script`) no estado visível ao Solver. **Nunca acessa o oráculo.** |
| `sandbox` | Cria um canal restrito para `exec` / `read_file` / `write_file` em um contêiner bloqueado e sem rede. |
| `roles` | Os cinco slots de função (Designer / Solver / Crítico / Avaliador / CohortSolver). Apenas o Solver tem acesso a ferramentas; o Crítico é reservado para a interface, desativado por padrão. |
| `budget_governor` | Orçamentos por classe para chamadas de ferramentas + tempo decorrido, exibidos ao agente, aplicados no nível do kernel; interrupção forçada em loops patológicos. |
| `oracle_scorer` | Avaliação externa: resolvido **e** sem regressão em relação ao oráculo oculto (padrão SWE-bench). |
| `judge_panel` | Painel inter-familiar de avaliadores de modelos + redutor (PoLL) para validação de novidade e detecção de desvio. |
| `trace_writer` | Transcrição por tentativa no formato `EvalLog` do Inspect; grandes blocos armazenados por hash. |
| `observability` | Agregações por tentativa → por quebra-cabeça → por modelo; `pass^k` nativo. |
| `catalog` | Persistência duradoura com rastreamento de eventos + o ciclo de vida `Lab → Arena → Regression` (progressão considerando a abstenção, validação em qualquer momento) + a tipologia diferencial. Baseado no log encadeado por hash do `attestation`. |
| `attestation` | Provável criptográfica (cosign + armazenamento de eventos) atrás de uma fronteira de sub-processo tipada. |

A fronteira selada é executada em três camadas — **Camada 1** contexto avaliado (moldado pela implantação, formulação neutra), **Camada 2** formulação do engajamento (verificada quanto à contaminação a cada lançamento), **Camada 3** elementos visuais (classificação/tabela de classificação — interface voltada para o usuário, nunca em um contexto no qual o modelo resolve). A justificativa completa do projeto, com citações, está em [`docs/research-grounding.md`](docs/research-grounding.md).

## Instalação

```bash
# As a Python library + CLI (PyPI):
pip install ai-crucible          # or: uv pip install ai-crucible
ai-crucible --help

# Or zero-prerequisite via npx — downloads a verified binary, no Python needed:
npx @dogfood-lab/ai-crucible --help
```

**Execute um ciclo de diagnóstico** — um Solver tenta resolver um quebra-cabeça no ambiente controlado, avaliado externamente em relação ao oráculo selado, emitindo o `pass^k` / agregação de Wilson:

```bash
# @family selects the adapter: no tag / @claude -> Claude (ANTHROPIC_API_KEY);
# any other @family -> a local Ollama model of that family (text OR native tool-calls).
ai-crucible run puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

Cada execução **é acumulada no catálogo duradouro**. Leia e organize-o ou execute a verificação da fronteira de conscientização sobre avaliação:

```bash
ai-crucible catalog list                 # tiers + per-puzzle differential typology + health
ai-crucible catalog show <puzzle-id>     # one puzzle: runs, transition timeline, differential
ai-crucible catalog graduate             # preview Lab->Arena->Regression transitions (--apply to commit)

# Eval-awareness gate: does behaviour diverge between deploy- and test-framing?
ai-crucible probe puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

**Ferramentas de qualidade para avaliação offline** – sem modelo, sem GPU, executadas a partir de um relatório de execução registrado:

```bash
# Forward-screen a less-saturated, still-defensible discriminating admission set
# from a characterization run's persisted grade matrix (the harder-set pipeline):
ai-crucible calibration curate --from-run report.json --out harder.json

# Validate a candidate human-label file before a --human-labels round (intake gate):
ai-crucible labels validate human_labels.json
```

> **Prévia da pesquisa (v0.4.x).** O teste alternativo ω do painel de avaliadores ainda é um *modelo circular de simulação de júri*: para validá-lo, é necessário realizar uma rodada com **≥3 avaliadores humanos independentes** (o [teste alternativo](https://arxiv.org/abs/2501.10970)), o que um estúdio com apenas um avaliador humano não consegue fazer — portanto, essa rodada está **suspensa por restrição estrutural, e não por negligência**. Os avaliadores permanecem em estado **provisório**, o painel formado **é expandido para incluir um Claude Designer** quando o número mínimo de participantes não é atingido, e o instrumento revela isso em vez de simular uma base humana. Consulte a [tabela de resultados](SCORECARD.md) para obter os resultados honestos e objetivos da avaliação.

## Início rápido (a partir do código-fonte)

O AI Crucible utiliza [`uv`](https://docs.astral.sh/uv/) para o gerenciamento do ambiente e das dependências. Python **3.11+**.

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

## Avaliação entre diferentes famílias de modelos

A primeira execução **publicada** de avaliação entre diferentes famílias de modelos está disponível em [`eval/RESULTS.md`](eval/RESULTS.md) (juntamente com o arquivo `eval/panel.json` e o relatório de caracterização). Sete famílias distintas – duas locais (gemma4, granite4.1) e cinco endpoints do OpenRouter selecionados – foram avaliadas em 93 pares de calibração com k=3: 1.395 chamadas pagas com **zero interrupções devido a limites de taxa**.

**O resultado honesto:** o conjunto agora **aceita 3 famílias distintas** (em vez das 2 originais, que eram apenas locais), e um novo avaliador entre diferentes famílias é integrado – mas o painel *independente* composto ainda tem **2 membros** (o terceiro foi removido para evitar redundância de erros, ρ≈1.0), o que é **inferior ao quórum**, portanto, o painel **passa a utilizar o Claude Designer** em vez de tomar uma decisão automática. O gargalo acabou sendo o **eixo ω do teste alternativo não validado, e não a qualidade dos avaliadores** – quatro avaliadores fortes (precisão de 0,91–0,96) são avaliados *exclusivamente* no modelo circular-júri ω. “3 aceitos” é um passo real; **não** é uma solução para o problema do eixo ω. O eixo ω permanece em espera, os membros permanecem provisórios e a conclusão ainda está adiada – divulgado, não simulado.

## Documentação

- **[Manual](https://dogfood-lab.github.io/ai-crucible/)** — guias, arquitetura e referências.
- [`docs/research-grounding.md`](docs/research-grounding.md) — justificativa do projeto, com citações.
- [`docs/gameplan.md`](docs/gameplan.md) — roteiro e questões em aberto.
- [`SECURITY.md`](SECURITY.md) — modelo de ameaças + divulgação honesta dos riscos residuais.

## Licença

[MIT](LICENSE). Público e anterior à versão 1.0 — consulte o [REGISTRO DE ALTERAÇÕES](CHANGELOG.md) para obter informações sobre o status da versão.

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
