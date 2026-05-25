/**
 * OpenClawLocalLlmClient — local-mode offload LLM client backed by OpenClaw.
 *
 * Implements the same methods as LocalLlmClient, but routes calls through
 * OpenClawLLMRunnerFactory so providers such as github-copilot can use the
 * host's native token exchange instead of a raw API key.
 */
import { OpenClawLLMRunnerFactory } from "../../adapters/openclaw/llm-runner.js";
import type { OpenClawLLMRunnerFactoryOptions } from "../../adapters/openclaw/llm-runner.js";
import { L1_SYSTEM_PROMPT, buildL1UserPrompt, type L1ToolPair } from "./prompts/l1-prompt.js";
import { L15_SYSTEM_PROMPT, buildL15UserPrompt, type L15CurrentMmd, type L15MmdMeta } from "./prompts/l15-prompt.js";
import { L2_SYSTEM_PROMPT, buildL2UserPrompt, type L2NewEntry } from "./prompts/l2-prompt.js";
import { parseL1Response } from "./parsers/l1-parser.js";
import { parseL15Response } from "./parsers/l15-parser.js";
import { parseL2Response } from "./parsers/l2-parser.js";
import type { PluginLogger } from "../types.js";
import type { L1Request, L1Response, L15Request, L15Response, L2Request, L2Response } from "../backend-client.js";

const TAG = "[context-offload] [local-llm] [openclaw]";

export interface OpenClawLocalLlmClientConfig {
  config: unknown;
  agentRuntime?: OpenClawLLMRunnerFactoryOptions["agentRuntime"];
  modelRef: string;
  timeoutMs?: number;
}

export class OpenClawLocalLlmClient {
  private runnerFactory: OpenClawLLMRunnerFactory;
  private modelRef: string;
  private timeoutMs: number;
  private logger?: PluginLogger;

  constructor(cfg: OpenClawLocalLlmClientConfig, logger?: PluginLogger) {
    this.runnerFactory = new OpenClawLLMRunnerFactory({
      config: cfg.config,
      agentRuntime: cfg.agentRuntime,
      logger,
    });
    this.modelRef = cfg.modelRef;
    this.timeoutMs = cfg.timeoutMs ?? 120_000;
    this.logger = logger;
    logger?.info?.(`${TAG} Initialized: model=${cfg.modelRef}`);
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
    const raw = await this.callRunner(L1_SYSTEM_PROMPT, userPrompt, "L1");

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
    const raw = await this.callRunner(L15_SYSTEM_PROMPT, userPrompt, "L1.5");

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

    const raw = await this.callRunner(L2_SYSTEM_PROMPT, userPrompt, "L2", 120_000);
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

  private async callRunner(
    systemPrompt: string,
    userPrompt: string,
    label: string,
    timeoutMs = this.timeoutMs,
  ): Promise<string> {
    const startMs = Date.now();
    this.logger?.info?.(
      `${TAG} ${label} >>> model=${this.modelRef}, timeout=${timeoutMs}ms, ` +
      `systemLen=${systemPrompt.length}, userLen=${userPrompt.length}`,
    );

    try {
      const runner = this.runnerFactory.createRunner({
        modelRef: this.modelRef,
        enableTools: false,
      });
      const text = (await runner.run({
        systemPrompt,
        prompt: userPrompt,
        taskId: `offload-${label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
        timeoutMs,
      })).trim();
      const elapsedMs = Date.now() - startMs;
      this.logger?.info?.(`${TAG} ${label} <<< ${elapsedMs}ms, output=${text.length} chars`);
      return text;
    } catch (err) {
      const elapsedMs = Date.now() - startMs;
      const errMsg = err instanceof Error ? err.message : String(err);
      this.logger?.error?.(`${TAG} ${label} FAILED (${elapsedMs}ms): ${errMsg}`);
      throw err;
    }
  }
}
