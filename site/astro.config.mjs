// @ts-check
import { defineConfig } from 'astro/config';

// Deployed to GitHub Pages at https://gaspigz.github.io/th-gaspar/
// `site` + `base` make Astro emit correct absolute URLs for a project page.
// Locally (dev/preview) BASE_URL resolves to "/th-gaspar/" too, so the
// fetch() of graph.json in index.astro uses import.meta.env.BASE_URL to stay
// correct in every environment.
// https://docs.astro.build/en/guides/deploy/github/
export default defineConfig({
  site: 'https://gaspigz.github.io',
  base: '/th-gaspar',
});
