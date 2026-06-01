// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://dogfood-lab.github.io',
  base: '/crucible',
  integrations: [
    starlight({
      title: 'crucible',
      description: 'A diagnostic adversarial game for frontier LLMs — a measurement instrument that happens to be fun.',
      logo: {
        src: './src/assets/logo.png',
        alt: 'crucible',
        href: '/crucible/',
        replacesTitle: false,
      },
      disable404Route: true,
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/dogfood-lab/crucible' },
      ],
      sidebar: [
        {
          label: 'Handbook',
          autogenerate: { directory: 'handbook' },
        },
      ],
      customCss: ['./src/styles/starlight-custom.css'],
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
});
