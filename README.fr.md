<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

Une session Claude (**Concepteur**) crée des énigmes ciblant des lacunes de capacités réelles et actuellement observées. Une autre session (**Résolveur**) tente de les résoudre. Un noyau, régi par des règles, intervient, évalue les résultats par rapport à un oracle caché et crée un catalogue au moyen d’un cycle de vie `Lab → Arena → Regression`. Les énigmes sont basées sur des données empiriques — des problèmes réels sur GitHub, des publications universitaires, des défaillances observées sur le terrain — et non sur des données synthétiques.

## Ce qui le rend différent

- **Capacité, et non « triche ».** Crucible distingue l’*élégance* et la *nouveauté* (récompensées) de la *contournement de la réponse* (pénalisée). La pensée latérale est une capacité à mesurer, et non un défaut à punir.
- **L’instrument se mesure lui-même.** La formulation de la question est un élément de mesure de premier ordre : le noyau exécute la même énigme avec des formulations « neutres », « autoréférentielles » ou « basées sur la réputation », et signale son propre effet sur la réponse à titre d’indicateur.
- **Une limite de mesure hermétique.** La motivation et la mesure ne partagent jamais le même contexte ; l’oracle caché est évalué en dehors du système par un modèle différent, et le raisonnement de l’agent est masqué. Le modèle ne peut pas manipuler ce qu’il ne peut pas percevoir.
- **Fiabilité grâce à la cohérence.** `pass^k` (toutes les *k* tentatives indépendantes réussissent), intervalles de Wilson et panels d’évaluateurs de différentes familles — conçus pour signaler des distributions, et non des estimations ponctuelles.
- **Un catalogue évolutif.** Les énigmes résolues sont *rétrogradées* vers la catégorie « Regression », mais ne sont jamais supprimées, de sorte que le catalogue devient une chronologie de l’évolution des capacités au fur et à mesure que la limite est repoussée.

## Modèle de menace (résumé)

L’adversaire principal de Crucible est le **modèle en cours d’évaluation**, qui peut tenter d’accéder à la clé de réponse ou de détecter et de manipuler l’évaluation. Il **affecte** : les formulations/configurations des énigmes, le répertoire de travail restreint du Résolveur et les appels d’API du fournisseur de modèles que l’opérateur configure. Il **n’affecte pas** : l’oracle/la clé de réponse (évaluée en dehors du système sur un hôte distinct auquel le Résolveur n’a pas accès) ou les éléments de motivation (« éléments d’interface » — classement/réputation — qui ne sont jamais injectés dans le contexte évalué). **Autorisations** : clés du fournisseur de modèles via des variables d’environnement au moment de l’exécution ; aucun secret intégré, aucune télémétrie, aucun appel sortant autonome. Divulgation complète — y compris les endroits où une limite est une *défense en profondeur* plutôt qu’une garantie absolue — est disponible dans **[SECURITY.md](SECURITY.md)**.

## Architecture

Crucible est une **couche de règles légère sur [Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI), et non un système entièrement nouveau. Un seul objet `AttemptState` est transmis du Concepteur au Résolveur, puis au (Critique) et enfin au Juge, via **un seul point d’étranglement `generate`**, de sorte que chaque appel de modèle et de chaque outil est observable.

| Module | Responsabilité |
| ------ | -------------- |
| `puzzle_loader` | Charge un répertoire d’énigmes (`meta.json` / `prompt` / `setup_script`) dans l’état visible par le Résolveur. **N’affecte jamais l’oracle.** |
| `sandbox` | Limite l’accès aux commandes `exec` / `read_file` / `write_file` à un conteneur verrouillé et sans accès réseau. |
| `roles` | Les cinq rôles (Concepteur / Résolveur / Critique / Juge / CohortSolver). Seul le Résolveur a accès aux outils ; l’interface du Critique est réservée et désactivée par défaut. |
| `budget_governor` | Appels d’outils et budgets en temps réel par classe, affichés à l’agent, appliqués au niveau du noyau ; arrêt brutal en cas de boucles pathologiques. |
| `oracle_scorer` | Évaluation en dehors du système : résolution **et** absence de régression par rapport à l’oracle caché (modèle SWE-bench). |
| `judge_panel` | Panel de modèles d’évaluation de différentes familles + réducteur (PoLL) pour la validation de la nouveauté et la détection de contournement. |
| `trace_writer` | Transcription par tentative dans le format `EvalLog` d’Inspect ; les gros blocs de données sont stockés par hachage. |
| `observability` | Regroupements par tentative → par énigme → par modèle ; `pass^k` natif. |
| `attestation` | Provenance cryptographique (cosign + event-store) derrière une limite de sous-processus typée. |

La limite hermétique fonctionne en trois niveaux : **Niveau 1** : contexte évalué (conforme au déploiement, formulation neutre), **Niveau 2** : formulation de l’interaction (vérifiée pour détecter toute contamination à chaque version), **Niveau 3** : éléments d’interface (classement/tableau de bord — uniquement pour l’interface utilisateur, jamais dans un contexte dans lequel le modèle résout un problème). La justification complète de la conception, avec des citations, est disponible dans [`docs/research-grounding.md`](docs/research-grounding.md).

## Démarrage rapide

Crucible utilise [`uv`](https://docs.astral.sh/uv/) pour la gestion de l’environnement et des dépendances. Python **3.11+**.

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

## Documentation

- **[Manuel](https://dogfood-lab.github.io/crucible/)** — guides, architecture et référence.
- [`docs/research-grounding.md`](docs/research-grounding.md) — justification de la conception, avec des citations.
- [`docs/gameplan.md`](docs/gameplan.md) — feuille de route et questions ouvertes.
- [`SECURITY.md`](SECURITY.md) — modèle de menace + divulgation honnête des risques résiduels.

## Licence

[MIT](LICENSE). Public et pré-1.0 — voir le [CHANGELOG](CHANGELOG.md) pour connaître l’état de la version.

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
