import { describe, expect, it } from "vitest";

import type { LLMRunParams, LLMRunner, LLMRunnerCreateOptions, LLMRunnerFactory } from "../core/types.js";
import { OpenClawOffloadLlmAdapter } from "./openclaw-llm-adapter.js";

class FakeRunnerFactory implements LLMRunnerFactory {
  createOptions: LLMRunnerCreateOptions | undefined;
  runCalls: LLMRunParams[] = [];
  response = "";

  createRunner(opts?: LLMRunnerCreateOptions): LLMRunner {
    this.createOptions = opts;
    return {
      run: async (params: LLMRunParams) => {
        this.runCalls.push(params);
        return this.response;
      },
    };
  }
}

describe("OpenClawOffloadLlmAdapter", () => {
  it("routes L1 through the runner with modelRef and configured timeout", async () => {
    const factory = new FakeRunnerFactory();
    factory.response = JSON.stringify([
      {
        tool_call: "read_file: inspect src/index.ts",
        summary: "Found the entrypoint.",
        tool_call_id: "call-1",
        timestamp: "2026-05-25T09:00:00+08:00",
        score: 8,
      },
    ]);

    const adapter = new OpenClawOffloadLlmAdapter({
      runnerFactory: factory,
      modelRef: "github-copilot/gpt-5.5",
      timeoutMs: 42_000,
    });

    const resp = await adapter.l1Summarize({
      recentMessages: "User asked to inspect the entrypoint.",
      toolPairs: [
        {
          toolName: "read_file",
          toolCallId: "call-1",
          params: { path: "src/index.ts" },
          result: "export {}",
          timestamp: "2026-05-25T09:00:00+08:00",
        },
      ],
    });

    expect(factory.createOptions).toEqual({ modelRef: "github-copilot/gpt-5.5", enableTools: false });
    expect(factory.runCalls).toHaveLength(1);
    expect(factory.runCalls[0]).toMatchObject({
      taskId: "offload-l1",
      timeoutMs: 42_000,
    });
    expect(factory.runCalls[0].systemPrompt).toContain("工具结果摘要器");
    expect(factory.runCalls[0].prompt).toContain("call-1");
    expect(resp.entries).toEqual([
      {
        tool_call: "read_file: inspect src/index.ts",
        summary: "Found the entrypoint.",
        tool_call_id: "call-1",
        timestamp: "2026-05-25T09:00:00+08:00",
        score: 8,
        node_id: null,
      },
    ]);
  });

  it("uses the LocalLlmClient-compatible L1.5 parse fallback", async () => {
    const factory = new FakeRunnerFactory();
    factory.response = "not json";
    const adapter = new OpenClawOffloadLlmAdapter({
      runnerFactory: factory,
      modelRef: "github-copilot/gpt-5.5",
    });

    const resp = await adapter.l15Judge({
      recentMessages: "hello",
      currentMmd: null,
      availableMmdMetas: [],
    });

    expect(resp).toEqual({
      taskCompleted: false,
      isContinuation: false,
      isLongTask: false,
    });
  });

  it("uses the L2 120s timeout override", async () => {
    const factory = new FakeRunnerFactory();
    factory.response = JSON.stringify({
      file_action: "write",
      mmd_content: "```mermaid\nflowchart TD\n  A[done]\n```",
      node_mapping: { "call-1": "001-N1" },
    });
    const adapter = new OpenClawOffloadLlmAdapter({
      runnerFactory: factory,
      modelRef: "github-copilot/gpt-5.5",
      timeoutMs: 1_000,
    });

    const resp = await adapter.l2Generate({
      existingMmd: null,
      newEntries: [
        {
          tool_call_id: "call-1",
          tool_call: "read_file",
          summary: "Read a file",
          timestamp: "2026-05-25T09:00:00+08:00",
        },
      ],
      recentHistory: null,
      currentTurn: null,
      taskLabel: "inspect-entrypoint",
      mmdPrefix: "001",
      mmdCharCount: 0,
    });

    expect(factory.runCalls[0]).toMatchObject({
      taskId: "offload-l2",
      timeoutMs: 120_000,
    });
    expect(resp).toEqual({
      fileAction: "write",
      mmdContent: "flowchart TD\n  A[done]",
      replaceBlocks: undefined,
      nodeMapping: { "call-1": "001-N1" },
    });
  });
});
