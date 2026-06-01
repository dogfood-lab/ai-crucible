import type { SiteConfig } from '@mcptoolshop/site-theme';

export const config: SiteConfig = {
  title: 'crucible',
  description:
    'A diagnostic adversarial game for frontier LLMs — a measurement instrument that happens to be fun.',
  logoBadge: 'CR',
  brandName: 'crucible',
  repoUrl: 'https://github.com/dogfood-lab/crucible',
  footerText:
    'MIT Licensed — built by <a href="https://mcp-tool-shop.github.io/" style="color:var(--color-muted);text-decoration:underline">MCP Tool Shop</a>, part of the <a href="https://github.com/dogfood-lab" style="color:var(--color-muted);text-decoration:underline">dogfood-lab</a> workshop',

  hero: {
    badge: 'Diagnostic adversarial game',
    headline: 'crucible',
    headlineAccent: 'a measurement instrument that happens to be fun.',
    description:
      'One Claude session crafts puzzles targeting real, currently-observed capability gaps; another attempts them. A policy-enforced kernel scores against a hidden oracle and curates a Lab → Arena → Regression catalog — rewarding elegance and novelty, penalizing answer-bypass.',
    primaryCta: { href: 'https://github.com/dogfood-lab/crucible', label: 'View on GitHub' },
    secondaryCta: { href: 'handbook/', label: 'Read the Handbook' },
    previews: [
      { label: 'Clone', code: 'git clone https://github.com/dogfood-lab/crucible' },
      { label: 'Install', code: 'uv sync --extra dev --extra stats' },
      { label: 'Verify', code: 'bash verify.sh' },
    ],
  },

  sections: [
    {
      kind: 'features',
      id: 'features',
      title: 'What makes it different',
      subtitle: 'A diagnostic instrument, not a leaderboard.',
      features: [
        {
          title: 'Capability, not "cheating"',
          desc: 'Distinguishes elegance and novelty (rewarded) from answer-bypass (penalized). Lateral thinking is a capability to measure, not a vice to punish.',
        },
        {
          title: 'The instrument measures itself',
          desc: 'Prompt framing is a first-class measured arm — the kernel runs the same puzzle under neutral / self-referential / social framings and reports its own prompt-effect.',
        },
        {
          title: 'A sealed boundary',
          desc: 'Motivation and measurement never share a context window; the hidden oracle is graded out-of-band by a different model family with the agent’s reasoning hidden.',
        },
        {
          title: 'Reliability by consistency',
          desc: 'pass^k (all k trials succeed), Wilson intervals, and cross-family judge panels — distributions, not point estimates.',
        },
      ],
    },
    {
      kind: 'code-cards',
      id: 'usage',
      title: 'Quick start',
      cards: [
        { title: 'Set up', code: 'uv sync --extra dev --extra stats' },
        { title: 'Run the suite', code: 'uv run pytest --cov=crucible' },
        { title: 'One-command gate', code: 'bash verify.sh' },
      ],
    },
  ],
};
