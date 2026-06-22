<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

Una sesión de Claude (**Diseñador**) crea acertijos dirigidos a deficiencias reales y actualmente observadas. Otra (**Solucionador**) intenta resolverlos. Un núcleo con políticas aplicadas actúa como mediador, evalúa los resultados en comparación con un oráculo oculto y organiza un catálogo mediante un ciclo de vida `Laboratorio → Arena → Regresión`. Los acertijos se basan en datos empíricos: problemas reales de GitHub, literatura académica, fallos observados en el campo, no en datos sintéticos.

## ¿Qué lo hace diferente?

- **Capacidad, no "trampa".** AI Crucible distingue la *elegancia* y la *novedad* (recompensadas) de la *elusión de la respuesta* (penalizada). El pensamiento lateral es una capacidad que se puede medir, no un defecto que se debe castigar.
- **El instrumento se mide a sí mismo.** La formulación del prompt es un elemento medido de primera clase: el núcleo ejecuta el mismo acertijo con diferentes enfoques (`neutral` / `autorreferencial` / `posición social`) y registra su propio efecto en el prompt como diagnóstico.
- **Un límite de medición sellado.** La motivación y la medición nunca comparten una ventana de contexto; el oráculo oculto se evalúa fuera del sistema por un modelo diferente, con el razonamiento del agente oculto. El modelo no puede manipular lo que no puede percibir.
- **Fiabilidad mediante consistencia.** `pass^k` (todos los *k* intentos independientes tienen éxito), intervalos de Wilson y paneles de evaluación intermodelos: diseñados para informar sobre distribuciones, no sobre estimaciones puntuales.
- **Un catálogo dinámico y duradero.** Las ejecuciones se acumulan en las sesiones en un registro basado en eventos y encadenado por hash (la fuente de la verdad); el estado del nivel es una proyección derivada. La transición de `Laboratorio → Arena` tiene en cuenta la **abstención**: solo promueve cuando hay un veredicto intermodelo confiable y, de lo contrario, **se escala al Diseñador**; los acertijos saturados se **reubican en `Regresión`, nunca se eliminan** (un proceso electrónico anti-repetición), por lo que el catálogo se convierte en una línea de tiempo de evolución de capacidades a medida que avanza la frontera.
- **La recompensa diferencial.** Por cada acertijo, la tasa de resolución de Claude frente a la del grupo intermodelo clasifica la diferencia: **específica de Claude** (valor más alto) / **general para LLM** / **fortaleza de Claude**, basándose en un intervalo de diferencia de Newcombe con una clase *inconclusiva* de primera categoría, por lo que un resultado nulo pequeño nunca se presenta como un hallazgo.
- **Mide cualquier familia de modelos.** Los solucionadores que utilizan el protocolo nativo de llamadas a funciones (no solo el protocolo de acción de texto) son de primera clase: un modelo que devuelve llamadas a herramientas en lugar de acciones de texto resuelve el problema a través del mismo entorno controlado.

## Modelo de amenazas (resumen)

El principal adversario de AI Crucible es el **modelo que se está midiendo**, que puede intentar acceder a la clave de respuesta o detectar y manipular la evaluación. Este modelo **accede** a: los prompts/configuración del acertijo, el directorio de trabajo confinado del Solucionador y las llamadas a la API del proveedor del modelo que configura el operador. No accede a: el oráculo/clave de respuesta (evaluado fuera del sistema en un host separado al que el Solucionador no puede acceder) ni a los elementos motivacionales ("decoración": clasificación/posiciones, nunca se inyectan en el contexto evaluado). **Permisos:** claves del proveedor del modelo a través de variables de entorno en tiempo de ejecución; sin secretos integrados, sin telemetría, sin llamadas salientes propias. La divulgación completa, incluido dónde un límite es una *defensa en profundidad* en lugar de una garantía absoluta, se encuentra en **[SECURITY.md](SECURITY.md)**.

## Arquitectura

AI Crucible es una **capa de políticas delgada sobre [Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI), no un sistema creado desde cero. Un único objeto `AttemptState` se transmite desde el Diseñador al Solucionador y luego al (Crítico) y, finalmente, al Evaluador a través de **un único punto de control (`generate`)**, por lo que cada llamada de modelo y herramienta es observable.

| Módulo | Responsabilidad |
| ------ | -------------- |
| `puzzle_loader` | Carga un directorio de acertijos (`meta.json` / `prompt` / `setup_script`) en el estado visible para el Solucionador. **Nunca accede al oráculo.** |
| `sandbox` | Canaliza estrechamente las funciones `exec` / `read_file` / `write_file` a un contenedor bloqueado y sin conexión de red. |
| `roles` | Los cinco roles (Diseñador / Solucionador / Crítico / Evaluador / Solucionador del grupo). Solo el Solucionador tiene acceso a herramientas; la interfaz del Crítico está reservada y desactivada por defecto. |
| `budget_governor` | Presupuestos de llamadas a herramientas y tiempo de reloj por clase, mostrados al agente, aplicados a nivel del núcleo; interrupción forzada en bucles patológicos. |
| `oracle_scorer` | Evaluación fuera del sistema: resuelto **y** sin regresión con respecto al oráculo oculto (patrón SWE-bench). |
| `judge_panel` | Panel intermodelo de evaluadores + reductor (PoLL) para la validación de la novedad y la detección de elusión. |
| `trace_writer` | Transcripción por intento en el formato `EvalLog` de Inspect; los datos grandes se almacenan por resumen. |
| `observability` | Resúmenes por intento → por acertijo → por modelo; `pass^k` nativo. |
| `catalog` | Persistencia duradera basada en eventos + el ciclo de vida `Laboratorio → Arena → Regresión` (graduación con conocimiento de la abstención, saturación válida en cualquier momento) + la tipología diferencial. Se basa en el registro encadenado por hash de `attestation`. |
| `attestation` | Procedencia criptográfica (cosign + almacén de eventos) detrás de un límite de subproceso tipificado. |

El límite sellado se ejecuta en tres niveles: **Nivel 1** contexto evaluado (configurado para el despliegue, neutral en cuanto al enfoque), **Nivel 2** enfoque de la interacción (analizado para detectar contaminación en cada versión), **Nivel 3** elementos adicionales (clasificación/tabla de clasificación: solo interfaz de usuario orientada al usuario, nunca en un contexto en el que el modelo resuelve el problema). La justificación completa del diseño, con citas, se encuentra en [`docs/research-grounding.md`](docs/research-grounding.md).

## Instalación

```bash
# As a Python library + CLI (PyPI):
pip install ai-crucible          # or: uv pip install ai-crucible
ai-crucible --help

# Or zero-prerequisite via npx — downloads a verified binary, no Python needed:
npx @dogfood-lab/ai-crucible --help
```

**Ejecute un ciclo de diagnóstico**: un Solucionador intenta resolver un acertijo en el entorno controlado, y los resultados se evalúan fuera del sistema con respecto al oráculo sellado, generando el resultado `pass^k` / Wilson:

```bash
# @family selects the adapter: no tag / @claude -> Claude (ANTHROPIC_API_KEY);
# any other @family -> a local Ollama model of that family (text OR native tool-calls).
ai-crucible run puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

Cada ejecución **se acumula en el catálogo duradero**. Lea y organícelo, o ejecute la prueba de conocimiento de la conciencia de la evaluación:

```bash
ai-crucible catalog list                 # tiers + per-puzzle differential typology + health
ai-crucible catalog show <puzzle-id>     # one puzzle: runs, transition timeline, differential
ai-crucible catalog graduate             # preview Lab->Arena->Regression transitions (--apply to commit)

# Eval-awareness gate: does behaviour diverge between deploy- and test-framing?
ai-crucible probe puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

> **Vista previa de la investigación (v0.3.x).** La prueba alternativa ω del jurado sigue siendo un *modelo circular de validación por bootstrapping*: para validarla, se necesita una ronda de **≥3 anotadores humanos independientes** (la [prueba alternativa](https://arxiv.org/abs/2501.10970)), lo cual no puede ser realizado por un estudio con un solo evaluador; por lo tanto, esta ronda está **temporalmente suspendida debido a una limitación estructural, no por falta de atención**. Los jueces permanecen en estado **provisional**, el jurado se **amplía hasta convertirse en un Claude Designer** cuando no se alcanza el quórum y el instrumento revela esto en lugar de simular la participación humana. Consulte la [hoja de resultados](SCORECARD.md) para obtener los resultados honestos y reales de las pruebas.

## Guía rápida (desde el código fuente)

AI Crucible utiliza [`uv`](https://docs.astral.sh/uv/) para la gestión del entorno y las dependencias. Python **3.11 o superior**.

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

## Documentación

- **[Manual](https://dogfood-lab.github.io/ai-crucible/)** — guías, arquitectura y referencia.
- [`docs/research-grounding.md`](docs/research-grounding.md) — justificación del diseño, con citas.
- [`docs/gameplan.md`](docs/gameplan.md) — hoja de ruta y preguntas pendientes.
- [`SECURITY.md`](SECURITY.md) — modelo de amenazas + divulgación honesta de los riesgos residuales.

## Licencia

[MIT](LICENSE). Pública y anterior a la versión 1.0; consulte el [REGISTRO DE CAMBIOS](CHANGELOG.md) para conocer el estado de la versión.

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
