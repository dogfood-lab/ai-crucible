<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/dogfood-lab/ai-crucible/main/assets/logo.png" alt="ai-crucible" width="400" />
</p>

<p align="center">
  <a href="https://github.com/dogfood-lab/ai-crucible/actions/workflows/ci.yml"><img src="https://github.com/dogfood-lab/ai-crucible/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg" alt="Python 3.11–3.13" />
  <img src="https://img.shields.io/badge/coverage-93%25-brightgreen.svg" alt="Coverage 93%" />
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.2.0-orange.svg" alt="Version 0.2.0" /></a>
  <a href="https://dogfood-lab.github.io/ai-crucible/"><img src="https://img.shields.io/badge/docs-handbook-orange.svg" alt="Handbook" /></a>
</p>

<p align="center"><b>A diagnostic adversarial game for frontier LLMs — a measurement instrument that happens to be fun.</b></p>

Una sessione di Claude (**Designer**) crea enigmi mirati a specifiche lacune nelle capacità, attualmente osservate. Un'altra (**Solver**) tenta di risolverli. Un kernel, soggetto a regole, funge da mediatore, valuta le risposte rispetto a un "oracolo" nascosto e crea un catalogo attraverso un ciclo di vita `Lab → Arena → Regression`. Gli enigmi si basano su dati empirici: problemi reali su GitHub, letteratura accademica, errori osservati sul campo, e non su elementi sintetici.

## Cosa lo rende diverso

- **Capacità, non "imbroglio".** Crucible distingue l'*eleganza* e la *novità* (premiate) dall'*elusione della risposta* (penalizzata). Il pensiero laterale è una capacità da misurare, non un difetto da punire.
- **Lo strumento si auto-valuta.** La formulazione del prompt è un elemento di misurazione di primo piano: il kernel esegue lo stesso enigma con formulazioni `neutre` / `autoriferite` / `basate sulla reputazione` e riporta il proprio effetto sul prompt come elemento diagnostico.
- **Un confine di misurazione sigillato.** La motivazione e la misurazione non condividono mai lo stesso contesto; l'"oracolo" nascosto viene valutato esternamente da una famiglia di modelli diversa, con il ragionamento dell'agente nascosto. Il modello non può manipolare ciò che non può percepire.
- **Affidabilità tramite coerenza.** `pass^k` (tutti i *k* tentativi indipendenti hanno successo), intervalli di Wilson e commissioni di valutazione inter-famiglia: progettati per riportare le distribuzioni, non le stime puntuali.
- **Un catalogo in continua evoluzione.** Gli enigmi risolti vengono *declassati* a `Regression`, ma non vengono mai eliminati, quindi il catalogo diventa una cronologia dell'evoluzione delle capacità man mano che i limiti si spostano.

## Modello delle minacce (riassunto)

Il principale avversario di Crucible è il **modello in fase di valutazione**, che potrebbe tentare di accedere alla soluzione o di rilevare e manipolare la valutazione. Esso **accede a**: prompt/configurazione dell'enigma, la directory di lavoro riservata del Solver e le chiamate API del fornitore del modello configurate dall'operatore. Non **accede a**: l'"oracolo"/soluzione (valutata esternamente su un host separato a cui il Solver non può accedere) o elementi di "motivazione" (classifica/reputazione: mai inseriti nel contesto valutato). **Autorizzazioni:** chiavi del fornitore del modello tramite variabili d'ambiente in fase di esecuzione; nessun segreto incorporato, nessuna telemetria, nessuna chiamata esterna autonoma. La completa divulgazione, compreso dove un confine rappresenta una *difesa a più livelli* piuttosto che una garanzia assoluta, è disponibile in **[SECURITY.md](SECURITY.md)**.

## Architettura

Crucible è un **sottile livello di policy sopra [Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI), non un sistema creato da zero. Un singolo oggetto `AttemptState` viene trasmesso da Designer → Solver → (Critic) → Judge attraverso **un unico punto di controllo `generate`**, in modo che ogni chiamata al modello e allo strumento sia osservabile.

| Modulo | Responsabilità |
| ------ | -------------- |
| `puzzle_loader` | Carica una directory di enigmi (`meta.json` / `prompt` / `setup_script`) nello stato visibile al Solver. **Non accede mai all'"oracolo".** |
| `sandbox` | Canalizza `exec` / `read_file` / `write_file` in un contenitore isolato e senza connessione di rete. |
| `roles` | I cinque slot di ruolo (Designer / Solver / Critic / Judge / CohortSolver). Solo il Solver ha accesso agli strumenti; l'interfaccia del Critic è riservata e disattivata per impostazione predefinita. |
| `budget_governor` | Chiamate agli strumenti specifiche per classe + limiti di tempo, visualizzati all'agente, applicati a livello di kernel; interruzione forzata in caso di cicli patologici. |
| `oracle_scorer` | Valutazione esterna: risolto **e** senza regressione rispetto all'"oracolo" nascosto (modello SWE-bench). |
| `judge_panel` | Commissione inter-famiglia di valutatori di modelli + riduttore (PoLL) per la convalida della novità e il rilevamento dell'elusione. |
| `trace_writer` | Trascrizione per tentativo nel formato `EvalLog` di Inspect; i dati di grandi dimensioni vengono archiviati tramite hash. |
| `observability` | Aggregazione per tentativo → per enigma → per modello; `pass^k` nativo. |
| `attestation` | Provenienza crittografica (cosign + event-store) dietro un confine di sottoprocesso tipizzato. |

Il confine sigillato opera su tre livelli: **Livello 1** contesto valutato (modellato in base alla distribuzione, neutrale rispetto alla formulazione), **Livello 2** formulazione dell'interazione (verificata per contaminazione a ogni rilascio), **Livello 3** elementi aggiuntivi (classifica/lavagna: solo interfaccia utente rivolta all'utente, mai in un contesto in cui il modello risolve). La completa motivazione del progetto, con citazioni, è disponibile in [`docs/research-grounding.md`](docs/research-grounding.md).

## Installa

```bash
# As a Python library + CLI (PyPI):
pip install ai-crucible          # or: uv pip install ai-crucible
ai-crucible --help

# Or zero-prerequisite via npx — downloads a verified binary, no Python needed:
npx @dogfood-lab/ai-crucible --help
```

> **Anteprima di ricerca (v0.2.x).** Il test alternativo ω della giuria è ancora un *modello circolare di bootstrap con giuria simulata* finché non viene eseguita una fase di etichettatura umana, quindi i giudici presenti sono **provvisori** e la giuria così composta **si trasforma in una giuria di Claude Designer** quando non si raggiunge il quorum. Consulta la [scheda dei risultati](SCORECARD.md) per i risultati onesti e non manipolati.

## Guida rapida (partendo dal codice sorgente)

Crucible utilizza [`uv`](https://docs.astral.sh/uv/) per la gestione dell'ambiente e delle dipendenze. Python **3.11+**.

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

## Documentazione

- **[Handbook](https://dogfood-lab.github.io/crucible/)** — guide, architettura e riferimento.
- [`docs/research-grounding.md`](docs/research-grounding.md) — motivazione del progetto, con citazioni.
- [`docs/gameplan.md`](docs/gameplan.md) — tabella di marcia e domande aperte.
- [`SECURITY.md`](SECURITY.md) — modello delle minacce + divulgazione onesta dei rischi residui.

## Licenza

[MIT](LICENSE). Pubblico e pre-1.0: vedere il [CHANGELOG](CHANGELOG.md) per lo stato della versione.

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
