/**
 * Unified LLM caller for offload local mode.
 *
 * For copilot providers: uses the official OpenAI SDK's Responses API
 * (client.responses.create) which matches what the main session uses.
 *
 * For other providers: uses Vercel AI SDK (`ai` + `@ai-sdk/openai`) with
 * "compatible" mode for standard OpenAI-compatible endpoints.
 */
import { generateText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import OpenAI from "openai";
import type { PluginLogger } from "../types.js";

const TAG = "[context-offload] [local-llm]";

export interface LlmCallerConfig {
  baseUrl: string;
  apiKey: string;
  /** Custom headers to include with every request (e.g. copilot IDE headers). */
  headers?: Record<string, string>;
  /** Use the official OpenAI SDK Responses API (client.responses.create). */
  useResponsesApi?: boolean;
  model: string;
  temperature: number;
  timeoutMs: number;
}

export interface CallLlmOpts {
  systemPrompt: string;
  userPrompt: string;
  /** Override temperature for this call */
  temperature?: number;
  /** Override timeout for this call */
  timeoutMs?: number;
  /** Label for logging (e.g. "L1", "L1.5", "L2") */
  label?: string;
}

/**
 * Call LLM with the given prompts and return the text response.
 * Throws on timeout or API errors.
 */
export async function callLlm(
  config: LlmCallerConfig,
  opts: CallLlmOpts,
  logger?: PluginLogger,
): Promise<string> {
  const startMs = Date.now();
  const label = opts.label ?? "call";
  const temperature = opts.temperature ?? config.temperature;
  const timeoutMs = opts.timeoutMs ?? config.timeoutMs;

  logger?.info?.(
    `${TAG} ${label} >>> model=${config.model}, temp=${temperature}, timeout=${timeoutMs}ms, ` +
    `systemLen=${opts.systemPrompt.length}, userLen=${opts.userPrompt.length}`,
  );

  try {
    let text: string;

    if (config.useResponsesApi) {
      // Use the official OpenAI SDK's Responses API — same format as the main session
      const client = new OpenAI({
        apiKey: config.apiKey,
        baseURL: config.baseUrl,
        defaultHeaders: config.headers,
        timeout: timeoutMs,
      });

      const stream = await client.responses.create({
        model: config.model,
        instructions: opts.systemPrompt,
        input: [{ role: "user" as const, content: opts.userPrompt }],
        temperature,
        stream: true,
      });

      // Collect streamed response text
      let chunks: string[] = [];
      for await (const event of stream as any) {
        if (event?.type === "response.output_text.delta" && event?.delta) {
          chunks.push(event.delta);
        } else if (event?.type === "response.completed" && event?.response?.output_text) {
          chunks = [event.response.output_text];
          break;
        }
      }
      text = chunks.join("").trim();
    } else {
      // Standard path: Vercel AI SDK for OpenAI-compatible endpoints
      const provider = createOpenAI({
        baseURL: config.baseUrl,
        apiKey: config.apiKey,
        compatibility: "compatible",
        headers: config.headers,
      });

      // Use Responses API for copilot (copilot routes GPT models via /v1/responses,
      // not /v1/chat/completions — the latter returns 421 Misdirected Request).
      // Non-copilot providers use standard chat completions.
      const isCopilot = config.headers?.["Copilot-Integration-Id"] != null;
      const modelInstance = isCopilot
        ? provider.responses(config.model)
        : provider.chat(config.model);

      const result = await generateText({
        model: modelInstance,
        system: opts.systemPrompt,
        prompt: opts.userPrompt,
        temperature,
        abortSignal: AbortSignal.timeout(timeoutMs),
      });

      text = result.text.trim();
    }

    const elapsedMs = Date.now() - startMs;
    logger?.info?.(
      `${TAG} ${label} <<< ${elapsedMs}ms, output=${text.length} chars`,
    );

    return text;
  } catch (err) {
    const elapsedMs = Date.now() - startMs;
    const errMsg = err instanceof Error ? err.message : String(err);
    logger?.error?.(`${TAG} ${label} FAILED (${elapsedMs}ms): ${errMsg}`);
    throw err;
  }
}
