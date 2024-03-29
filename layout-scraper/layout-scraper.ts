import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { InvalidArgumentError, program } from "commander";
import puppeteer from "puppeteer";
import type { Browser } from "puppeteer";

type ScrapeOptions = {
  /** Width to set viewport to. */
  width?: number;

  /** Height to set viewport to. */
  height?: number;

  /**
   * File path that a PNG screenshot of the web page will be saved to before
   * capturing layout information. If omitted, no screenshot is taken.
   */
  screenshotFile?: string;

  /**
   * Trim layout output to words that intersect viewport.
   */
  trim?: boolean;
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

  if (typeof options.screenshotFile === "string") {
    const screenshot = await page.screenshot({ encoding: "binary" });
    writeFileSync(options.screenshotFile, screenshot);
  }

  // Un-comment to enable debugging via logging from the page.
  //
  // page.on("console", (msg) => console.log("PAGE LOG:", msg.text()));

  const layoutInfo = await page.evaluate((options) => {
    /** Convert a DOMRect into a JSON-serializable array. */
    const coordsFromRect = (domRect: DOMRect): BoxCoords => [
      domRect.left,
      domRect.top,
      domRect.right,
      domRect.bottom,
    ];

    const isEmptyRect = (domRect: DOMRect) =>
      domRect.width <= 0 || domRect.height <= 0;

    const intersectsViewport = (domRect: DOMRect) =>
      domRect.top < window.innerHeight && domRect.left < window.innerWidth;

    const elementVisibleCache = new Map<Element, boolean>();

    const computeElementIsVisible = (el: Element): boolean => {
      if (el === document.body) {
        return true;
      }
      const style = getComputedStyle(el);
      if (style.display === "none" || style.visibility === "hidden") {
        return false;
      }

      // Min / max thresholds set at 1 here because some sites use boxes of
      // width/height 1 to be visible to screen readers while not being
      // technically hidden.
      const rect = el.getBoundingClientRect();
      if (rect.width <= 1 || rect.height <= 1) {
        return false;
      }

      if (!el.parentElement) {
        return false;
      }
      return computeElementIsVisible(el.parentElement);
    };

    const elementIsVisible = (el: Element): boolean => {
      let visible = elementVisibleCache.get(el);
      if (typeof visible === "boolean") {
        return visible;
      }
      visible = computeElementIsVisible(el);
      elementVisibleCache.set(el, visible);
      return visible;
    };

    /**
     * Return the nearest ancestor element of `node` that uses a non-horizontal
     * layout for its children.
     */
    const nearestBlockAncestor = (node: Node) => {
      let parent = node instanceof Element ? node : node.parentElement;
      while (parent) {
        const style = getComputedStyle(parent);
        if (
          !style.display.includes("inline") &&
          !["table-cell", "table-row"].includes(style.display)
        ) {
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
      const parentElement = currentNode.parentElement!;

      if (!elementIsVisible(parentElement)) {
        // Skip over non-rendered text.
        continue;
      }
      const boundingRect = parentElement.getBoundingClientRect();
      if (options.trim && !intersectsViewport(boundingRect)) {
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
      for (const match of str.matchAll(/\w+/g)) {
        const offset = match.index!;
        const word = match[0];

        range.setStart(currentNode, offset);
        range.setEnd(currentNode, offset + word.length);
        const wordRect = range.getBoundingClientRect();

        if (
          wordRect.width > 0 &&
          wordRect.height > 0 &&
          (!options.trim || intersectsViewport(wordRect))
        ) {
          currentPara.words.push({
            text: word,
            coords: coordsFromRect(wordRect),
          });
        }
      }
    }
    return {
      url: document.location.href,
      resolution: { width: window.innerWidth, height: window.innerHeight },
      paragraphs: layoutParagraphs,
    };
  }, options);

  await page.close();

  return layoutInfo;
}

type CLIOptions = {
  height?: number;
  incremental: boolean;
  inFile?: string;
  outDir?: string;
  screenshot: boolean;
  width?: number;
  trim: boolean;
};

/**
 * Convert a URL into a filename with no `/`s in it.
 *
 * This is a lossy transformation and URLs which are different but contain the
 * same sequence of ASCII letters/numbers may be mapped to the same URL.
 */
function filenameForURL(url: string) {
  // Strip protocol
  let filename = url.replace(/^https?:\/\//, "");

  // Replace special chars
  filename = filename.replace(/[/?:]/g, "_");

  // Trim trailing special chars
  filename = filename.replace(/_+$/, "");

  return filename;
}

function parseIntArg(val: string): number {
  if (!val.match(/^[0-9]+$/)) {
    throw new InvalidArgumentError("Must be a positive integer");
  }
  return parseInt(val, 10);
}

function countWords(li: LayoutInfo): number {
  return li.paragraphs.reduce((total, para) => total + para.words.length, 0);
}

async function processURLs(browser: Browser, urls: string[], opts: CLIOptions) {
  const outDir = opts.outDir ?? ".";
  mkdirSync(outDir, { recursive: true });

  const width = opts.width ?? 1024;
  const height = opts.height ?? 768;
  const failedURLs = [];

  for (const [i, url] of urls.entries()) {
    const outFileBase =
      outDir + "/" + filenameForURL(url) + `-${width}x${height}`;
    const layoutFile = `${outFileBase}.json`;
    if (opts.incremental && existsSync(layoutFile)) {
      continue;
    }

    try {
      const scrapeOpts: ScrapeOptions = { trim: opts.trim };
      if (opts.screenshot) {
        scrapeOpts.screenshotFile = `${outFileBase}.png`;
      }
      const layoutInfo = await scrapeTextLayout(browser, url, scrapeOpts);
      const layoutJSON = JSON.stringify(layoutInfo, null, 2);

      const nWords = countWords(layoutInfo);
      console.log(
        `Rendered ${url} (${i + 1} of ${urls.length}). ${nWords} words.`,
      );

      writeFileSync(layoutFile, layoutJSON);
    } catch (err) {
      console.error(`Processing URL "${url}" failed:`, err);
      failedURLs.push(url);
    }
  }

  if (failedURLs.length > 0) {
    console.error(`${failedURLs.length} URLs failed`);
  }
}

function isValidURL(val: string) {
  try {
    new URL(val);
    return true;
  } catch (e) {
    return false;
  }
}

program
  .description(
    "Render web pages using a headless browser and capture text layout information.",
  )
  .argument("[urls...]", "URLs to render")
  .option("-i, --in-file <file>", "Read URLs from a file")
  .option("-o, --out-dir <dir>", "Output directory")
  .option("-s, --screenshot", "Save screenshots")
  .option("-w, --width [width]", "Browser viewport width", parseIntArg)
  .option("-h, --height [height]", "Browser viewport height", parseIntArg)
  .option("-n, --incremental", "Skip URLs which have already been rendered")
  .option(
    "-t, --trim",
    "Trim layout output to words that intersect the viewport",
  )
  .action(async (urls: string[], opts: CLIOptions) => {
    const browser = await puppeteer.launch({ headless: "new" });

    if (opts.inFile) {
      const urlsFromFile = readFileSync(opts.inFile, "utf8")
        .split("\n")
        .filter((line) => {
          const trimmed = line.trim();
          return trimmed.length > 0 && !trimmed.startsWith("#");
        });
      for (const url of urlsFromFile) {
        urls.push(url);
      }
    }

    urls = urls.filter((url) => {
      if (!isValidURL(url)) {
        console.warn(`Skipping invalid URL "${url}"`);
        return false;
      }
      return true;
    });

    try {
      await processURLs(browser, urls, opts);
    } finally {
      await browser.close();
    }
  });
program.parse();
