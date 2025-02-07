import express, { Request, Response } from "express";
import Tesseract from "tesseract.js";
import sharp from "sharp";

const app = express();
app.use(express.json({ limit: "24mb" }));

interface ImageResponse {
  width: number;
  height: number;
  content: string;
  confidence: number;
  processingTimeMs: number;
}

enum Mode {
  FAST = "fast",
  ACCURATE = "accurate",
}

interface Options {
  mode?: Mode;
  words?: string[];
  autocorrect?: boolean;
  languages?: string[];
  detect_language?: boolean;
}

type ImageSource =
  | { url?: string }
  | { base64?: string }
  | { bytes?: number[] };

interface ImageMetadata {
  buffer: Buffer;
  width: number;
  height: number;
}

const ALLOWED_DOMAINS = new Set([
  "discord.mx",
]);

async function loadImage(source: ImageSource): Promise<ImageMetadata> {
  try {
    let inputBuffer: Buffer;
    
    if ("url" in source && source.url) {
      const url = new URL(source.url);
      
      if (!ALLOWED_DOMAINS.has(url.hostname)) {
        throw new Error("Domain not allowed");
      }

      const response = await fetch(source.url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const arrayBuffer = await response.arrayBuffer();
      inputBuffer = Buffer.from(arrayBuffer);
    } else if ("base64" in source && source.base64) {
      inputBuffer = Buffer.from(source.base64, "base64");
    } else if ("bytes" in source && source.bytes) {
      inputBuffer = Buffer.from(source.bytes);
    } else {
      throw new Error("Invalid image source");
    }

    const image = sharp(inputBuffer);
    const [metadata, buffer] = await Promise.all([
      image.metadata(),
      image.png().toBuffer()
    ]);

    if (!metadata.width || !metadata.height) {
      throw new Error("Failed to get image dimensions");
    }

    return {
      buffer,
      width: metadata.width,
      height: metadata.height,
    };
  } catch (error) {
    console.error("Image processing error:", error);
    throw new Error("Failed to process image");
  }
}

async function recognizeText(
  image: ImageMetadata,
  options: Options
): Promise<Omit<ImageResponse, 'processingTimeMs'>> {
  const { data } = await Tesseract.recognize(
    image.buffer,
    options.languages?.[0] || "eng",
    {
      logger: (m) => options.mode === Mode.ACCURATE ? console.log(m) : undefined,
    }
  );

  return {
    width: image.width,
    height: image.height,
    content: data.text,
    confidence: data.confidence
  };
}


const handleError = (error: unknown): string => {
  return error instanceof Error ? error.message : "An unknown error occurred";
};

app.post("/recognize", async (req: Request, res: Response) => {
  const startTime = performance.now();
  
  try {
    const source: ImageSource = req.body;
    const options: Options = req.query as Options;
    const image = await loadImage(source);
    const result = await recognizeText(image, options);

    const processingTimeMs = Math.round(performance.now() - startTime);

    res.json({ 
      success: true, 
      data: { ...result, processingTimeMs } 
    });
  } catch (error) {
    res.status(400).json({ 
      success: false, 
      error: handleError(error),
      processingTimeMs: Math.round(performance.now() - startTime)
    });
  }
});

app.post("/batch", async (req: Request, res: Response) => {
  const startTime = performance.now();
  
  try {
    const sources: ImageSource[] = req.body;
    const options: Options = req.query as Options;

    const results = await Promise.all(
      sources.map(async (source) => {
        const imageStartTime = performance.now();
        try {
          const image = await loadImage(source);
          const result = await recognizeText(image, options);
          return {
            success: true,
            data: {
              ...result,
              processingTimeMs: Math.round(performance.now() - imageStartTime)
            }
          };
        } catch (error) {
          return {
            success: false,
            error: handleError(error),
            processingTimeMs: Math.round(performance.now() - imageStartTime)
          };
        }
      })
    );

    res.json({
      success: true,
      data: results,
      totalProcessingTimeMs: Math.round(performance.now() - startTime)
    });
  } catch (error) {
    res.status(400).json({
      success: false,
      error: handleError(error),
      processingTimeMs: Math.round(performance.now() - startTime)
    });
  }
});


const PORT = 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));