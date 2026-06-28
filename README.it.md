<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

Una sessione di Claude (**Designer**) crea enigmi mirati a specifiche lacune nelle capacità attuali. Un'altra (**Solver**) tenta di risolverli. Un kernel con politiche applicate funge da mediatore, valuta i risultati rispetto a un oracolo nascosto e cura un catalogo attraverso un ciclo di vita `Lab → Arena → Regression`. Gli enigmi si basano su dati empirici: problemi reali su GitHub, letteratura accademica, fallimenti osservati sul campo, e non su elementi sintetici.

## Cosa lo rende diverso?

- **Capacità, non "imbroglio".** AI Crucible distingue l'*eleganza* e la *novità* (premiate) dall'*elusione della risposta* (penalizzata). Il pensiero laterale è una capacità da misurare, non un difetto da punire.
- **Lo strumento si auto-valuta.** La formulazione del prompt è un elemento di misurazione di primo piano: il kernel esegue lo stesso enigma con formulazioni `neutre` / `autoriferite` / basate sullo `status sociale` e riporta i propri effetti sul prompt come diagnostica.
- **Un confine di misurazione sigillato.** La motivazione e la misurazione non condividono mai una finestra di contesto; l'oracolo nascosto viene valutato esternamente da un modello diverso, con il ragionamento dell'agente mantenuto segreto. Il modello non può manipolare ciò che non percepisce.
- **Affidabilità tramite coerenza.** `pass^k` (tutti i *k* tentativi indipendenti hanno successo), intervalli di Wilson e commissioni di valutazione inter-famiglia: progettati per riportare distribuzioni, non stime puntuali.
- **Un catalogo dinamico e duraturo.** Le esecuzioni si accumulano nelle sessioni in un registro basato su eventi e concatenato tramite hash (la fonte della verità); lo stato del livello è una proiezione derivata. Il passaggio da `Lab → Arena` tiene conto dell'*astensione*: promuove solo sulla base di un verdetto inter-famiglia affidabile e, altrimenti, **passa al Designer**; gli enigmi saturi vengono **declassati a `Regression`, ma mai eliminati** (un processo elettronico anti-fluttuazione), in modo che il catalogo diventi una cronologia dell'evoluzione delle capacità man mano che i limiti si spostano.
- **Il vantaggio differenziale.** Per ogni enigma, il tasso di risoluzione di Claude rispetto alla coorte inter-famiglia classifica la differenza: **specifica per Claude** (valore più alto) / **generale per gli LLM** / **punto di forza di Claude**, basato su un intervallo di differenza di Newcombe con una classe *inconclusiva* di primo piano, in modo che un campione piccolo con risultato nullo non venga mai presentato come una scoperta.
- **Misura qualsiasi famiglia di modelli.** I solver che utilizzano il protocollo nativo delle chiamate di funzione (e non solo il protocollo delle azioni testuali) sono considerati di primo piano: un modello che restituisce chiamate a strumenti anziché azioni testuali risolve i problemi attraverso lo stesso ambiente controllato.

## Modello delle minacce (riepilogo)

Il principale avversario di AI Crucible è il **modello in fase di valutazione**, che potrebbe tentare di accedere alla chiave di risposta o di rilevare e manipolare la valutazione. Esso **accede a**: prompt/configurazione degli enigmi, la directory di lavoro confinata del Solver e le chiamate API fornite dal provider del modello configurate dall'operatore. Non accede a: l'oracolo/chiave di risposta (valutata esternamente su un host separato a cui il Solver non può accedere) o elementi motivazionali ("cromature": classifica/punteggio, mai inseriti nel contesto valutato). **Autorizzazioni:** chiavi del provider del modello tramite variabili d'ambiente in fase di esecuzione; nessun segreto incorporato, nessuna telemetria, nessuna chiamata esterna autonoma. La divulgazione completa, compresi i punti in cui un confine rappresenta una *difesa a più livelli* piuttosto che una garanzia assoluta, è disponibile in **[SECURITY.md](SECURITY.md)**.

## Architettura

AI Crucible è uno **strato di policy sottile su [Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI), non un sistema creato da zero. Un singolo oggetto `AttemptState` viene trasmesso dal Designer al Solver e, facoltativamente, a un Critico e a un Giudice attraverso **un unico punto di controllo `generate`**, in modo che ogni chiamata di modello e strumento sia osservabile.

| Modulo | Responsabilità |
| ------ | -------------- |
| `puzzle_loader` | Carica una directory di enigmi (`meta.json` / `prompt` / `setup_script`) nello stato visibile al Solver. **Non accede mai all'oracolo.** |
| `sandbox` | Restringe il canale `exec` / `read_file` / `write_file` in un contenitore isolato e senza connessione di rete. |
| `roles` | I cinque slot di ruolo (Designer / Solver / Critico / Giudice / CohortSolver). Solo il Solver ha accesso agli strumenti; l'interfaccia del Critico è riservata ed è disattivata per impostazione predefinita. |
| `budget_governor` | Budget per classe di strumento + tempo massimo, visualizzati all'agente, applicati a livello di kernel; interruzione forzata in caso di cicli patologici. |
| `oracle_scorer` | Valutazione esterna: risolto **e** senza regressione rispetto all'oracolo nascosto (modello SWE-bench). |
| `judge_panel` | Commissione inter-famiglia di valutatori + riduttore (PoLL) per la convalida della novità e il rilevamento dell'elusione. |
| `trace_writer` | Trascrizione per tentativo nel formato `EvalLog` di Inspect; i blocchi di dati di grandi dimensioni vengono archiviati tramite digest. |
| `observability` | Aggregazioni per tentativo → per enigma → per modello; `pass^k` nativo. |
| `catalog` | Persistenza duratura basata su eventi + il ciclo di vita `Lab → Arena → Regression` (promozione che tiene conto dell'astensione, validità in qualsiasi momento) + la tipologia differenziale. Si basa sul registro concatenato tramite hash di `attestation`. |
| `attestation` | Provenienza crittografica (cosign + archivio eventi) dietro un confine di sottoprocesso tipizzato. |

Il confine sigillato opera su tre livelli: **Livello 1** contesto valutato (modellato in base alla distribuzione, neutrale rispetto alla formulazione), **Livello 2** formulazione dell'interazione (analizzata per rilevare contaminazioni a ogni rilascio), **Livello 3** elementi aggiuntivi (classifica/tabella dei punteggi: solo interfaccia utente rivolta all'utente, mai in un contesto in cui il modello risolve i problemi). La giustificazione completa del progetto, con citazioni, è disponibile in [`docs/research-grounding.md`](docs/research-grounding.md).

## Installazione

```bash
# As a Python library + CLI (PyPI):
pip install ai-crucible          # or: uv pip install ai-crucible
ai-crucible --help

# Or zero-prerequisite via npx — downloads a verified binary, no Python needed:
npx @dogfood-lab/ai-crucible --help
```

**Esegui un ciclo diagnostico**: un Solver tenta di risolvere un enigma nel sandbox, i risultati vengono valutati esternamente rispetto all'oracolo sigillato, generando il valore `pass^k` / Wilson.

```bash
# @family selects the adapter: no tag / @claude -> Claude (ANTHROPIC_API_KEY);
# any other @family -> a local Ollama model of that family (text OR native tool-calls).
ai-crucible run puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

Ogni esecuzione **si accumula nel catalogo duraturo**. Leggilo e curalo oppure esegui la sonda di consapevolezza della valutazione:

```bash
ai-crucible catalog list                 # tiers + per-puzzle differential typology + health
ai-crucible catalog show <puzzle-id>     # one puzzle: runs, transition timeline, differential
ai-crucible catalog graduate             # preview Lab->Arena->Regression transitions (--apply to commit)

# Eval-awareness gate: does behaviour diverge between deploy- and test-framing?
ai-crucible probe puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

**Strumenti di controllo della qualità degli strumenti utilizzabili offline:** non richiedono modelli o GPU e funzionano partendo da un report di esecuzione consolidato:

```bash
# Forward-screen a less-saturated, still-defensible discriminating admission set
# from a characterization run's persisted grade matrix (the harder-set pipeline):
ai-crucible calibration curate --from-run report.json --out harder.json

# Validate a candidate human-label file before a --human-labels round (intake gate):
ai-crucible labels validate human_labels.json
```

> **Anteprima della ricerca (v0.4.x).** Il test alternativo ω del collegio giudicante è ancora un *modello circolare di giuria basato sul bootstrap*: per convalidarlo, è necessario un ciclo di **almeno 3 valutatori umani indipendenti** (il [test alternativo](https://arxiv.org/abs/2501.10970)), cosa che uno studio con un solo valutatore umano non può garantire; pertanto, questo ciclo è **sospeso a causa di vincoli strutturali, e non per negligenza**. I giudici in carica rimangono **provvisori**, il collegio giudicante si **amplia fino a includere un Claude Designer** quando non si raggiunge il quorum e lo strumento rivela questa situazione anziché simulare una base umana. Consultare la [scheda dei risultati](SCORECARD.md) per i risultati onesti e privi di elementi puramente estetici.

## Guida rapida (dal codice sorgente)

AI Crucible utilizza [`uv`](https://docs.astral.sh/uv/) per la gestione dell'ambiente e delle dipendenze. Python **3.11 o superiore**.

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

## Valutazione inter-famiglia

La prima **esecuzione pubblicata** per la valutazione inter-famiglia è disponibile in [`eval/RESULTS.md`](eval/RESULTS.md) (insieme al file `eval/panel.json` e al report di caratterizzazione). Sono state valutate sette famiglie distinte: due locali (gemma4, granite4.1) e cinque endpoint OpenRouter predefiniti (deepseek, cohere, meta-llama, qwen, nvidia), utilizzando 93 coppie di dati per la calibrazione con k=3: 1.395 chiamate a pagamento senza **nessun caso di superamento del limite di richieste**.

**Il risultato reale:** il pool ora **include 3 famiglie distinte** (rispetto alle 2 iniziali, che erano solo locali), e un nuovo valutatore inter-famiglia funziona correttamente; tuttavia, il gruppo *indipendente* rimane composto da **2 membri** (il terzo è stato escluso per evitare ridondanza di errori, ρ≈1.0), il che è **inferiore al quorum**, quindi il gruppo **passa all'utilizzo di Claude Designer** invece di prendere una decisione automatica. Il collo di bottiglia si è rivelato essere l'**asse ω del test alternativo non validato, e non la qualità dei valutatori**: quattro valutatori affidabili (accuratezza 0,91–0,96) sono stati valutati *esclusivamente* sul modello circolare-giuria ω. "3 accettati" rappresenta un vero passo avanti; **non** significa che il problema di ω è stato risolto. ω rimane in fase di test, i membri del gruppo rimangono provvisori e la valutazione definitiva viene rinviata: questo è trasparente, non simulato.

## Documentazione

- **[Manuale](https://dogfood-lab.github.io/ai-crucible/)** — guide, architettura e riferimenti.
- [`docs/research-grounding.md`](docs/research-grounding.md) — motivazioni alla base del progetto, con citazioni.
- [`docs/gameplan.md`](docs/gameplan.md) — tabella di marcia e questioni aperte.
- [`SECURITY.md`](SECURITY.md) — modello delle minacce + divulgazione onesta dei rischi residui.

## Licenza

[MIT](LICENSE). Pubblica e precedente alla versione 1.0; consultare il [REGISTRO DELLE MODIFICHE](CHANGELOG.md) per lo stato della versione.

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
