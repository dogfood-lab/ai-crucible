<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

Una sesión de Claude (**Diseñador**) crea acertijos dirigidos a las deficiencias de capacidad reales y actualmente observadas. Otra (**Solucionador**) intenta resolverlos. Un núcleo con políticas aplicadas actúa como mediador, evalúa los resultados en comparación con un oráculo oculto y organiza un catálogo a través de un ciclo de vida de `Laboratorio → Arena → Regresión`. Los acertijos se basan en datos empíricos: problemas reales de GitHub, literatura académica, fallos observados en el campo, y no en datos sintéticos.

## ¿Qué lo hace diferente?

- **Capacidad, no "trampa".** Crucible distingue la *elegancia* y la *novedad* (recompensadas) de la *elusión de la respuesta* (penalizada). El pensamiento lateral es una capacidad que se puede medir, no un defecto que se debe castigar.
- **El instrumento se mide a sí mismo.** La formulación del problema es un componente medido de primer nivel: el núcleo ejecuta el mismo acertijo con formulaciones `neutral`, `autorreferencial` y `social`, y notifica su propio efecto en el resultado como un diagnóstico.
- **Un límite de medición sellado.** La motivación y la medición nunca comparten una ventana de contexto; el oráculo oculto se evalúa fuera de banda por una familia de modelos diferente, con el razonamiento del agente oculto. El modelo no puede manipular lo que no puede percibir.
- **Fiabilidad mediante la consistencia.** `pass^k` (todas las *k* pruebas independientes tienen éxito), intervalos de Wilson y paneles de evaluación inter-familia: diseñados para informar sobre distribuciones, no sobre estimaciones puntuales.
- **Un catálogo dinámico.** Los acertijos resueltos se *reubican* en `Regresión`, pero nunca se eliminan, por lo que el catálogo se convierte en una línea de tiempo de la evolución de las capacidades a medida que avanza la frontera.

## Modelo de amenazas (resumen)

El principal adversario de Crucible es el **modelo que se está evaluando**, que puede intentar acceder a la clave de respuesta o detectar y manipular la evaluación. Este modelo tiene acceso a: las indicaciones/configuración del acertijo, el directorio de trabajo restringido del Solucionador y las llamadas a la API del proveedor del modelo que el operador configura. No tiene acceso a: el oráculo/clave de respuesta (evaluado fuera de banda en un host separado al que el Solucionador no puede acceder) ni a los elementos de "interfaz" de la motivación (clasificación/posiciones, que nunca se inyectan en el contexto evaluado). **Permisos:** claves del proveedor del modelo a través de variables de entorno en tiempo de ejecución; sin secretos integrados, sin telemetría, sin llamadas salientes propias. La divulgación completa, incluido dónde un límite es una *defensa en profundidad* en lugar de una garantía absoluta, se encuentra en **[SECURITY.md](SECURITY.md)**.

## Arquitectura

Crucible es una **capa de políticas delgada sobre [Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI), no un sistema creado desde cero. Un único objeto `AttemptState` se transmite desde el Diseñador al Solucionador y al (Crítico) y luego al Evaluador a través de **un único punto de control `generate`**, de modo que todas las llamadas a modelos y herramientas son observables.

| Módulo | Responsabilidad |
| ------ | -------------- |
| `puzzle_loader` | Carga un directorio de acertijos (`meta.json` / `prompt` / `setup_script`) en el estado visible para el Solucionador. **Nunca accede al oráculo.** |
| `sandbox` | Canaliza `exec` / `read_file` / `write_file` en un contenedor bloqueado y sin conexión de red. |
| `roles` | Los cinco espacios de roles (Diseñador / Solucionador / Crítico / Evaluador / CohortSolver). Solo el Solucionador tiene acceso a herramientas; el Crítico tiene una interfaz reservada y está desactivada por defecto. |
| `budget_governor` | Llamadas a herramientas específicas de la clase + límites de tiempo de reloj, mostrados al agente, aplicados a nivel del núcleo; finalización forzada en bucles patológicos. |
| `oracle_scorer` | Evaluación fuera de banda: resuelto **y** sin regresión en comparación con el oráculo oculto (patrón SWE-bench). |
| `judge_panel` | Panel inter-familia de evaluadores de modelos + reductor (PoLL) para la validación de la novedad y la detección de elusión. |
| `trace_writer` | Transcripción por intento en el formato `EvalLog` de Inspect; los datos grandes se almacenan por resumen. |
| `observability` | Resúmenes por intento → por acertijo → por modelo; `pass^k` nativo. |
| `attestation` | Procedencia criptográfica (cosign + event-store) detrás de un límite de subproceso tipificado. |

El límite sellado se ejecuta en tres niveles: **Nivel 1** contexto evaluado (con la forma de la implementación, neutral en cuanto a la formulación), **Nivel 2** formulación de la interacción (analizada para detectar contaminación en cada versión), **Nivel 3** elementos de la interfaz (clasificación/tabla de posiciones, solo interfaz de usuario orientada al usuario, nunca en un contexto en el que el modelo resuelve el problema). La justificación completa del diseño, con citas, se encuentra en [`docs/research-grounding.md`](docs/research-grounding.md).

## Inicio rápido

Crucible utiliza [`uv`](https://docs.astral.sh/uv/) para la gestión del entorno y las dependencias. Python **3.11+**.

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

## Documentación

- **[Manual](https://dogfood-lab.github.io/crucible/)** — guías, arquitectura y referencia.
- [`docs/research-grounding.md`](docs/research-grounding.md) — justificación del diseño, con citas.
- [`docs/gameplan.md`](docs/gameplan.md) — hoja de ruta y preguntas pendientes.
- [`SECURITY.md`](SECURITY.md) — modelo de amenazas + divulgación honesta de los riesgos residuales.

## Licencia

[MIT](LICENSE). Público y pre-1.0: consulte el [CHANGELOG](CHANGELOG.md) para conocer el estado de la versión.

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
