/**
 * OpenClawOffloadAdapter — local-mode offload client backed by OpenClaw native
 * model routing instead of raw OpenAI-compatible credentials.
 */
import { OpenClawLLMRunnerFactory } from "../adapters/openclaw/llm-runner.js";
import type { LLMRunner } from "../core/types.js";
import type { EmbeddedAgentRuntimeLike } from "../utils/clean-context-runner.js";
import { L1_SYSTEM_PROMPT, buildL1UserPrompt, type L1ToolPair } from "./local-llm/prompts/l1-prompt.js";
import { L15_SYSTEM_PROMPT, buildL15UserPrompt, type L15CurrentMmd, type L15MmdMeta } from "./local-llm/prompts/l15-prompt.js";
import { L2_SYSTEM_PROMPT, buildL2UserPrompt, type L2NewEntry } from "./local-llm/prompts/l2-prompt.js";
import { parseL1Response } from "./local-llm/parsers/l1-parser.js";
import { parseL15Response } from "./local-llm/parsers/l15-parser.js";
import { parseL2Response } from "./local-llm/parsers/l2-parser.js";
import type { PluginLogger } from "./types.js";
import type {
  L1Request,
  L1Response,
  L15Request,
  L15Response,
  L2Request,
  L2Response,
  L4Request,
  L4Response,
  StoreStatePayload,
  StoreStateResponse,
} from "./backend-client.js";

const TAG = "[context-offload] [openclaw-native]";

export interface OpenClawOffloadAdapterConfig {
  config: unknown;
  modelRef: string;
  agentRuntime?: EmbeddedAgentRuntimeLike;
  logger?: PluginLogger;
  timeoutMs?: number;
}

export class OpenClawOffloadAdapter {
  private runner: LLMRunner;
  private logger?: PluginLogger;
  private modelRef: string;
  private timeoutMs: number;

  constructor(cfg: OpenClawOffloadAdapterConfig) {
    this.logger = cfg.logger;
    this.modelRef = cfg.modelRef;
    this.timeoutMs = cfg.timeoutMs ?? 120_000;

    const factory = new OpenClawLLMRunnerFactory({
      config: cfg.config,
      agentRuntime: cfg.agentRuntime,
      logger: cfg.logger,
    });
    this.runner = factory.createRunner({
      modelRef: cfg.modelRef,
      enableTools: false,
    });

    this.logger?.info?.(`${TAG} Initialized: model=${cfg.modelRef}`);
  }

  async l1Summarize(req: L1Request): Promise<L1Response> {
    const pairs: L1ToolPair[] = req.toolPairs.map((p) => ({
      toolName: p.toolName,
      toolCallId: p.toolCallId,
      params: p.params,
      result: p.result,
      timestamp: p.timestamp,
    }));

    const raw = await this.runLlm({
      systemPrompt: L1_SYSTEM_PROMPT,
      userPrompt: buildL1UserPrompt(req.recentMessages, pairs),
      taskId: "offload-l1-summarize",
      label: "L1",
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

    const raw = await this.runLlm({
      systemPrompt: L15_SYSTEM_PROMPT,
      userPrompt: buildL15UserPrompt(req.recentMessages, currentMmd, metas),
      taskId: "offload-l15-judge",
      label: "L1.5",
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

    const raw = await this.runLlm({
      systemPrompt: L2_SYSTEM_PROMPT,
      userPrompt: buildL2UserPrompt({
        existingMmd: req.existingMmd,
        entries,
        recentHistory: req.recentHistory,
        currentTurn: req.currentTurn,
        taskLabel: req.taskLabel,
        mmdPrefix: req.mmdPrefix,
        charCount: req.mmdCharCount,
      }),
      taskId: "offload-l2-generate",
      label: "L2",
      timeoutMs: 120_000,
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

  async storeState(_payload: StoreStatePayload): Promise<StoreStateResponse> {
    return {};
  }

  async l4Generate(_req: L4Request): Promise<L4Response | null> {
    return null;
  }

  private async runLlm(opts: {
    systemPrompt: string;
    userPrompt: string;
    taskId: string;
    label: string;
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
        systemPrompt: opts.systemPrompt,
        prompt: opts.userPrompt,
        taskId: opts.taskId,
        timeoutMs,
      });
      const text = raw.trim();
      this.logger?.info?.(
        `${TAG} ${opts.label} <<< ${Date.now() - startMs}ms, output=${text.length} chars`,
      );
      return text;
    } catch (err) {
      this.logger?.error?.(
        `${TAG} ${opts.label} FAILED (${Date.now() - startMs}ms): ` +
        `${err instanceof Error ? err.message : String(err)}`,
      );
      throw err;
    }
  }
}
