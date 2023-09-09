#!/usr/bin/env node

/**
 * Fetch `count` random Wikipedia URLs and log them to stdout.
 */
async function fetchURLs(count) {
  for (let i = 0; i < count; i++) {
    // nb. `Special:Random` returns a 302, `fetch` follows redirects
    // automatically.
    const response = await fetch(
      "https://en.wikipedia.org/wiki/Special:Random",
    );
    if (!response.ok) {
      console.error(`Non-OK response ${response.status}`);
      continue;
    }
    console.error(`Fetched ${i + 1} of ${count} URLs`);
    console.log(response.url);
  }
}

const count = parseInt(process.argv[2]);
console.error(`Fetching ${count} random Wikipedia URLs...`);
fetchURLs(count);
