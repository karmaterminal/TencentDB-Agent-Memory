/**
 * OpenClaw-native offload LLM adapter.
 *
 * Used when local offload mode resolves an OpenClaw provider that has a gateway
 * baseUrl but no static apiKey. Calls are routed through CleanContextRunner via
 * OpenClawLLMRunnerFactory so provider auth stays owned by the OpenClaw runtime.
 */
import type { LLMRunner, LLMRunnerFactory } from "../core/types.js";
import { L1_SYSTEM_PROMPT, buildL1UserPrompt, type L1ToolPair } from "./local-llm/prompts/l1-prompt.js";
import { L15_SYSTEM_PROMPT, buildL15UserPrompt, type L15CurrentMmd, type L15MmdMeta } from "./local-llm/prompts/l15-prompt.js";
import { L2_SYSTEM_PROMPT, buildL2UserPrompt, type L2NewEntry } from "./local-llm/prompts/l2-prompt.js";
import { parseL1Response } from "./local-llm/parsers/l1-parser.js";
import { parseL15Response } from "./local-llm/parsers/l15-parser.js";
import { parseL2Response } from "./local-llm/parsers/l2-parser.js";
import type { PluginLogger } from "./types.js";
import type { L1Request, L1Response, L15Request, L15Response, L2Request, L2Response } from "./backend-client.js";

const TAG = "[context-offload] [openclaw-llm]";
const DEFAULT_TIMEOUT_MS = 120_000;
const L2_TIMEOUT_MS = 120_000;

export interface OpenClawOffloadLlmAdapterConfig {
  runnerFactory: LLMRunnerFactory;
  modelRef: string;
  timeoutMs?: number;
}

export class OpenClawOffloadLlmAdapter {
  private runner: LLMRunner;
  private modelRef: string;
  private timeoutMs: number;
  private logger?: PluginLogger;

  constructor(cfg: OpenClawOffloadLlmAdapterConfig, logger?: PluginLogger) {
    this.modelRef = cfg.modelRef;
    this.timeoutMs = cfg.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.logger = logger;
    this.runner = cfg.runnerFactory.createRunner({
      modelRef: cfg.modelRef,
      enableTools: false,
    });

    logger?.info?.(`${TAG} Initialized: model=${cfg.modelRef}, runner=openclaw-native`);
  }

  async l1Summarize(req: L1Request): Promise<L1Response> {
    const pairs: L1ToolPair[] = req.toolPairs.map((p) => ({
      toolName: p.toolName,
      toolCallId: p.toolCallId,
      params: p.params,
      result: p.result,
      timestamp: p.timestamp,
    }));

    const userPrompt = buildL1UserPrompt(req.recentMessages, pairs);
    const raw = await this.runPrompt({
      label: "L1",
      taskId: "offload-l1",
      systemPrompt: L1_SYSTEM_PROMPT,
      userPrompt,
    });

    const entries = parseL1Response(raw);
    if (entries.length === 0) {
      this.logger?.warn?.(`${TAG} L1: parsed 0 entries from LLM response (${raw.length} chars)`);
    }

    return { entries };
  }

  async l15Judge(req: L15Request): Promise<L15Response> {
    const currentMmd: L15CurrentMmd | null = req.currentMmd
      ? { filename: req.currentMmd.filename, content: req.currentMmd.content, path: req.currentMmd.path }
      : null;

    const metas: L15MmdMeta[] = req.availableMmdMetas.map((m) => ({
      filename: m.filename,
      path: m.path,
      taskGoal: m.taskGoal,
      doneCount: m.doneCount,
      doingCount: m.doingCount,
      todoCount: m.todoCount,
      updatedTime: m.updatedTime,
      nodeSummaries: m.nodeSummaries?.map((n) => ({
        nodeId: n.nodeId,
        status: n.status,
        summary: n.summary,
      })),
    }));

    const userPrompt = buildL15UserPrompt(req.recentMessages, currentMmd, metas);
    const raw = await this.runPrompt({
      label: "L1.5",
      taskId: "offload-l15",
      systemPrompt: L15_SYSTEM_PROMPT,
      userPrompt,
    });

    const result = parseL15Response(raw);
    if (!result) {
      this.logger?.warn?.(`${TAG} L1.5: failed to parse judgment from LLM response (${raw.length} chars)`);
      return {
        taskCompleted: false,
        isContinuation: false,
        isLongTask: false,
      } as L15Response;
    }

    return result as L15Response;
  }

  async l2Generate(req: L2Request): Promise<L2Response> {
    const entries: L2NewEntry[] = req.newEntries.map((e) => ({
      toolCallId: e.tool_call_id,
      toolCall: e.tool_call,
      summary: e.summary,
      timestamp: e.timestamp,
    }));

    const userPrompt = buildL2UserPrompt({
      existingMmd: req.existingMmd,
      entries,
      recentHistory: req.recentHistory,
      currentTurn: req.currentTurn,
      taskLabel: req.taskLabel,
      mmdPrefix: req.mmdPrefix,
      charCount: req.mmdCharCount,
    });

    const raw = await this.runPrompt({
      label: "L2",
      taskId: "offload-l2",
      systemPrompt: L2_SYSTEM_PROMPT,
      userPrompt,
      timeoutMs: L2_TIMEOUT_MS,
    });

    const result = parseL2Response(raw);
    if (!result) {
      this.logger?.error?.(`${TAG} L2: failed to parse response (${raw.length} chars)`);
      throw new Error("L2 response parsing failed");
    }

    return {
      fileAction: result.fileAction,
      mmdContent: result.mmdContent,
      replaceBlocks: result.replaceBlocks?.map((b) => ({
        startLine: b.startLine,
        endLine: b.endLine,
        content: b.content,
      })),
      nodeMapping: result.nodeMapping,
    };
  }

  async storeState(_payload: unknown): Promise<void> {}

  async l4Generate(_req: unknown): Promise<unknown> {
    return null;
  }

  private async runPrompt(opts: {
    label: string;
    taskId: string;
    systemPrompt: string;
    userPrompt: string;
    timeoutMs?: number;
  }): Promise<string> {
    const startMs = Date.now();
    const timeoutMs = opts.timeoutMs ?? this.timeoutMs;

    this.logger?.info?.(
      `${TAG} ${opts.label} >>> model=${this.modelRef}, timeout=${timeoutMs}ms, ` +
      `systemLen=${opts.systemPrompt.length}, userLen=${opts.userPrompt.length}`,
    );

    try {
      const raw = await this.runner.run({
        prompt: opts.userPrompt,
        systemPrompt: opts.systemPrompt,
        taskId: opts.taskId,
        timeoutMs,
      });
      const text = raw.trim();
      const elapsedMs = Date.now() - startMs;

      this.logger?.info?.(`${TAG} ${opts.label} <<< ${elapsedMs}ms, output=${text.length} chars`);
      return text;
    } catch (err) {
      const elapsedMs = Date.now() - startMs;
      const errMsg = err instanceof Error ? err.message : String(err);
      this.logger?.error?.(`${TAG} ${opts.label} FAILED (${elapsedMs}ms): ${errMsg}`);
      throw err;
    }
  }
}
