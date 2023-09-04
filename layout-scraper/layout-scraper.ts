import { program } from "commander";
import puppeteer from "puppeteer";
import type { Browser } from "puppeteer";

type ScrapeOptions = {
  width?: number;
  height?: number;
};

/** [left, top, right, bottom] coordinates. */
type BoxCoords = [number, number, number, number];

type LayoutWord = {
  text: string;
  coords: BoxCoords;
};

type LayoutParagraph = {
  coords: BoxCoords;
  words: LayoutWord[];
};

type LayoutInfo = {
  url: string;
  resolution: {
    width: number;
    height: number;
  };
  paragraphs: LayoutParagraph[];
};

/**
 * Render a web page and return a JSON-serializable object containing
 * information about the layout of visible text content.
 */
async function scrapeTextLayout(
  browser: Browser,
  url: string,
  options: ScrapeOptions = {},
) {
  const { width = 1024, height = 768 } = options;

  const page = await browser.newPage();
  await page.setViewport({ width, height });
  await page.goto(url);

  // Un-comment to enable debugging via logging from the page.
  //
  // page.on("console", (msg) => console.log("PAGE LOG:", msg.text()));

  const layoutInfo = await page.evaluate(() => {
    /** Convert a DOMRect into a JSON-serializable array. */
    const coordsFromRect = (domRect: DOMRect): BoxCoords => [
      domRect.left,
      domRect.top,
      domRect.right,
      domRect.bottom,
    ];

    const isEmptyRect = (domRect: DOMRect) =>
      domRect.width <= 0 || domRect.height <= 0;

    /**
     * Return the nearest ancestor element of `node` that uses a non-inline
     * layout.
     */
    const nearestBlockAncestor = (node: Node) => {
      let parent = node instanceof Element ? node : node.parentElement;
      while (parent) {
        if (!getComputedStyle(parent).display.includes("inline")) {
          return parent;
        }
        parent = parent.parentElement;
      }
      return null;
    };

    const layoutParagraphs: LayoutParagraph[] = [];

    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
    );
    let prevBlockParent;

    const range = new Range();
    let currentNode;
    while ((currentNode = walker.nextNode())) {
      if (isEmptyRect(currentNode.parentElement!.getBoundingClientRect())) {
        // Skip over non-rendered text.
        continue;
      }
      const str = currentNode.nodeValue!;
      if (str.trim().length === 0) {
        continue;
      }

      const blockParent = nearestBlockAncestor(currentNode)!;
      if (blockParent !== prevBlockParent) {
        prevBlockParent = blockParent;

        const newParagraph: LayoutParagraph = {
          words: [],
          coords: coordsFromRect(blockParent.getBoundingClientRect()),
        };
        layoutParagraphs.push(newParagraph);
      }

      const currentPara = layoutParagraphs.at(-1)!;

      let offset = 0;
      const words = str.split(" ");
      for (const word of words) {
        range.setStart(currentNode, offset);
        range.setEnd(currentNode, offset + word.length);
        const wordRect = range.getBoundingClientRect();
        const trimmedWord = word.trim();

        if (
          trimmedWord.length > 0 &&
          wordRect.width > 0 &&
          wordRect.height > 0
        ) {
          currentPara.words.push({
            text: trimmedWord,
            coords: coordsFromRect(wordRect),
          });
        }

        offset += word.length + 1;
      }
    }
    return {
      url: document.location.href,
      resolution: { width: window.innerWidth, height: window.innerHeight },
      paragraphs: layoutParagraphs,
    };
  });

  await page.close();

  return layoutInfo;
}

program.argument("<url>", "URL to render").action(async (url: string) => {
  const browser = await puppeteer.launch({ headless: "new" });
  try {
    const layoutInfo = await scrapeTextLayout(browser, url);
    console.log(JSON.stringify(layoutInfo, null, 2));
  } finally {
    await browser.close();
  }
});
program.parse();
