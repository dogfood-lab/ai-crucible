<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

Une session Claude (**Concepteur**) crée des énigmes ciblant de réels écarts de compétences actuellement observés. Une autre (**Résolveur**) tente de les résoudre. Un noyau régi par une politique assure la médiation, évalue les résultats par rapport à un oracle caché et organise un catalogue au moyen d’un cycle de vie `Lab → Arena → Regression`. Les énigmes sont basées sur des données empiriques — de réels problèmes GitHub, des publications universitaires, des défaillances observées sur le terrain — et non sur des éléments synthétiques.

## Ce qui la rend différente

- **Compétence, pas « triche ».** AI Crucible distingue l’*élégance* et la *nouveauté* (récompensées) de la *contournement de la réponse* (pénalisée). La pensée latérale est une compétence à mesurer, et non un défaut à sanctionner.
- **L’instrument se mesure lui-même.** La formulation des invites est un élément essentiel qui est mesuré : le noyau exécute la même énigme avec des formulations « neutres », « autoréférentielles » ou basées sur les « classements sociaux », et signale son propre effet de formulation à titre d’élément diagnostique.
- **Une limite de mesure hermétique.** La motivation et la mesure ne partagent jamais le même espace contextuel ; l’oracle caché est évalué en dehors du système par un modèle différent, avec le raisonnement de l’agent masqué. Le modèle ne peut pas manipuler ce qu’il ne perçoit pas.
- **Fiabilité grâce à la cohérence.** `pass^k` (toutes les *k* tentatives indépendantes réussissent), intervalles de Wilson et panels d’évaluateurs inter-modèles — conçus pour signaler des distributions, et non des estimations ponctuelles.
- **Un catalogue vivant et durable.** Les exécutions s’accumulent au fil des sessions dans un journal basé sur les événements et chaîné par hachage (la source de vérité) ; l’état du niveau est une projection dérivée. La promotion `Lab → Arena` tient compte de l’*abstention* — elle ne promeut que si le verdict inter-modèles est positif, et la *rétrograde vers le Concepteur* dans le cas contraire ; les énigmes saturées sont *rétrogradées vers « Regression », mais jamais supprimées* (un processus électronique anti-répétition), de sorte que le catalogue devient une chronologie d’évolution des compétences au fur et à mesure que la limite évolue.
- **Le gain différentiel.** Pour chaque énigme, le taux de résolution de Claude par rapport au groupe inter-modèles permet de classer l’écart : *spécifique à Claude* (valeur la plus élevée) / *général pour les LLM* / *force de Claude* — sur la base d’un intervalle de différence de Newcombe avec une classe *inconclusive* de premier ordre, de sorte qu’une petite série de résultats nuls ne soit jamais interprétée à tort comme une découverte.
- **Mesure tout type de modèle.** Les résolveurs utilisant le protocole natif d’appel de fonction (et pas seulement le protocole d’action textuelle) sont pris en compte en priorité : un modèle qui renvoie des appels d’outils au lieu d’actions textuelles résout les problèmes dans le même environnement contrôlé.

## Modèle de menace (résumé)

L’adversaire principal d’AI Crucible est le **modèle en cours de mesure**, qui peut tenter d’accéder à la clé de réponse ou de détecter et de manipuler l’évaluation. Il a accès : aux invites/configurations des énigmes, au répertoire de travail confiné du résolveur et aux appels d’API du fournisseur de modèles que l’opérateur configure. Il n’a pas accès : à l’oracle/à la clé de réponse (évaluée en dehors du système sur un hôte distinct auquel le résolveur ne peut pas accéder) ou aux éléments de motivation (« chrome » — classement/position, qui ne sont jamais injectés dans le contexte évalué). **Autorisations :** clés du fournisseur de modèles via des variables d’environnement au moment de l’exécution ; aucun secret intégré, aucune télémétrie, aucun appel sortant autonome. La divulgation complète, y compris les endroits où une limite est une *défense en profondeur* plutôt qu’une garantie absolue, se trouve dans **[SECURITY.md](SECURITY.md)**.

## Architecture

AI Crucible est une **couche de politique légère sur [Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI), et non un système conçu à partir de zéro. Un seul objet `AttemptState` est transmis du Concepteur au Résolveur, puis au (Critique) et enfin au Juge, via **un seul point d’étranglement `generate`**, de sorte que chaque appel de modèle et d’outil soit observable.

| Module | Responsabilité |
| ------ | -------------- |
| `puzzle_loader` | Charge un répertoire d’énigmes (`meta.json` / `prompt` / `setup_script`) dans l’état visible par le résolveur. **N’a jamais accès à l’oracle.** |
| `sandbox` | Limite les canaux `exec` / `read_file` / `write_file` à un conteneur verrouillé et sans connexion réseau. |
| `roles` | Les cinq emplacements de rôle (Concepteur / Résolveur / Critique / Juge / CohortSolver). Seul le résolveur a accès aux outils ; l’interface du critique est réservée et désactivée par défaut. |
| `budget_governor` | Budgets d’appels d’outils et de temps chronologique par classe, affichés à l’agent, appliqués au niveau du noyau ; arrêt brutal en cas de boucles pathologiques. |
| `oracle_scorer` | Évaluation hors bande : résolu **et** sans régression par rapport à l’oracle caché (modèle SWE-bench). |
| `judge_panel` | Panel inter-modèles d’évaluateurs de modèles + réducteur (PoLL) pour la validation de la nouveauté et la détection des contournements. |
| `trace_writer` | Transcription par tentative dans le format `EvalLog` d’Inspect ; les gros blocs sont stockés par hachage. |
| `observability` | Regroupements par tentative → par énigme → par modèle ; `pass^k` natif. |
| `catalog` | Persistance durable basée sur les événements + le cycle de vie `Lab → Arena → Regression` (promotion tenant compte de l’abstention, saturation valide à tout moment) + la typologie différentielle. S’appuie sur le journal chaîné par hachage d’« attestation ». |
| `attestation` | Preuve cryptographique (cosign + magasin d’événements) derrière une limite de sous-processus typée. |

La limite hermétique fonctionne en trois niveaux : **Niveau 1** contexte évalué (conçu pour le déploiement, formulation neutre), **Niveau 2** formulation de l’engagement (vérifiée pour détecter toute contamination à chaque version), **Niveau 3** éléments d’interface (« chrome » — classement/tableau des scores, uniquement une interface utilisateur destinée aux humains, jamais dans un contexte dans lequel le modèle résout les problèmes). La justification complète de la conception, avec des citations, se trouve dans [`docs/research-grounding.md`](docs/research-grounding.md).

## Installation

```bash
# As a Python library + CLI (PyPI):
pip install ai-crucible          # or: uv pip install ai-crucible
ai-crucible --help

# Or zero-prerequisite via npx — downloads a verified binary, no Python needed:
npx @dogfood-lab/ai-crucible --help
```

**Exécutez un cycle de diagnostic** : un résolveur tente de résoudre une énigme dans le bac à sable, et les résultats sont évalués en dehors du système par rapport à l’oracle hermétique, ce qui génère `pass^k` / Wilson.

```bash
# @family selects the adapter: no tag / @claude -> Claude (ANTHROPIC_API_KEY);
# any other @family -> a local Ollama model of that family (text OR native tool-calls).
ai-crucible run puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

Chaque exécution **s’accumule dans le catalogue durable**. Lisez-le et organisez-le, ou exécutez la sonde de sensibilisation à l’évaluation :

```bash
ai-crucible catalog list                 # tiers + per-puzzle differential typology + health
ai-crucible catalog show <puzzle-id>     # one puzzle: runs, transition timeline, differential
ai-crucible catalog graduate             # preview Lab->Arena->Regression transitions (--apply to commit)

# Eval-awareness gate: does behaviour diverge between deploy- and test-framing?
ai-crucible probe puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

> **Aperçu de la recherche (v0.3.x).** Le test alternatif ω du jury est toujours un *modèle circulaire de bootstrap avec un jury* : sa validation nécessite une série d’au moins **3 annotateurs humains indépendants** (le [test alternatif](https://arxiv.org/abs/2501.10970)), ce qu’un studio ne peut pas assurer avec un seul opérateur — cette étape est donc **suspendue en raison de contraintes structurelles, et non par négligence**. Les juges restent **provisoires**, le jury constitué **passe à un niveau supérieur pour atteindre un Claude Designer** lorsque le quorum n’est pas atteint, et l’outil révèle cela plutôt que de simuler une base humaine. Consultez la [fiche de résultats](SCORECARD.md) pour obtenir des résultats honnêtes et objectifs.

## Démarrage rapide (à partir du code source)

AI Crucible utilise [`uv`](https://docs.astral.sh/uv/) pour la gestion de l’environnement et des dépendances. Python **3.11 ou supérieur**.

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

## Documentation

- **[Manuel](https://dogfood-lab.github.io/ai-crucible/)** — guides, architecture et documentation de référence.
- [`docs/research-grounding.md`](docs/research-grounding.md) — justification de la conception, avec références.
- [`docs/gameplan.md`](docs/gameplan.md) — feuille de route et questions en suspens.
- [`SECURITY.md`](SECURITY.md) — modèle de menace + divulgation honnête des risques résiduels.

## Licence

[MIT](LICENSE). Public et version pré-1.0 — consultez le [JOURNAL DES MODIFICATIONS](CHANGELOG.md) pour connaître l’état de la version.

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
