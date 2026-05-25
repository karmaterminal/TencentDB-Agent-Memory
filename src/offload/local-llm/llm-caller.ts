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
      // Raw fetch to copilot Responses API — match main session's request shape
      // from openai-transport-stream.ts buildOpenAIResponsesParams +
      // copilot-dynamic-headers.ts buildCopilotDynamicHeaders.
      //
      // Important deltas vs the prior shape that 400'd:
      //   - System prompt goes inside `input` as the first item (role:"system"
      //     with content[].type:"input_text"), NOT as top-level `instructions`.
      //   - User prompt is a content item, not a bare string.
      //   - Add `x-initiator: user` header (Copilot routes by initiator).
      //   - Add `Accept-Encoding: identity` (avoids gzip framing surprises).
      const url = `${config.baseUrl}/v1/responses`;
      const input: Array<Record<string, unknown>> = [];
      if (opts.systemPrompt) {
        input.push({
          role: "system",
          content: [{ type: "input_text", text: opts.systemPrompt }],
        });
      }
      input.push({
        role: "user",
        content: [{ type: "input_text", text: opts.userPrompt }],
      });
      const body: Record<string, unknown> = {
        model: config.model,
        input,
        stream: true,
        store: false,
      };
      if (typeof temperature === "number") {
        body.temperature = temperature;
      }
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${config.apiKey}`,
          "Accept-Encoding": "identity",
          "x-initiator": "user",
          ...(config.headers ?? {}),
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });

      if (!res.ok) {
        const errBody = await res.text().catch(() => "");
        throw new Error(`${res.status} ${res.statusText}: ${errBody.slice(0, 400)}`);
      }

      // Collect SSE stream
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");
      const decoder = new TextDecoder();
      const chunks: string[] = [];
      let done = false;
      while (!done) {
        const { value, done: streamDone } = await reader.read();
        done = streamDone;
        if (value) {
          const lines = decoder.decode(value, { stream: true }).split("\n");
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") { done = true; break; }
            try {
              const event = JSON.parse(data);
              if (event.type === "response.output_text.delta" && event.delta) {
                chunks.push(event.delta);
              } else if (event.type === "response.completed" && event.response?.output_text) {
                chunks.length = 0;
                chunks.push(event.response.output_text);
                done = true;
              }
            } catch { /* skip non-JSON lines */ }
          }
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
