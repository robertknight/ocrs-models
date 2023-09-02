# layout-scraper

Tool that extracts text layout information from a web page, rendered at a
specific size.

## Usage

```sh
npm install
npm run build
node build/layout-scraper.js <URL>
```

The output is a JSON file containing information about the text boxes found
and their hierarchical relationship.
