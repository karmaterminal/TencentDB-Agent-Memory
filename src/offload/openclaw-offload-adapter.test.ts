import { describe, it, expect, vi } from "vitest";

// Mock the CleanContextRunner to avoid needing a real OpenClaw runtime
vi.mock("../utils/clean-context-runner.js", () => ({
  CleanContextRunner: class MockRunner {
    constructor() {}
    async run(params: any) {
      // Return a mock response based on the taskId
      if (params.taskId?.includes("l1")) {
        return { text: JSON.stringify({ summary: "mocked summary", score: 0.8 }) };
      }
      if (params.taskId?.includes("l15") || params.taskId?.includes("judge")) {
        return { text: JSON.stringify({ judgment: "continue", mmd_name: null }) };
      }
      if (params.taskId?.includes("l2")) {
        return { text: "flowchart TD\n  A[Task] --> B[Done]" };
      }
      return { text: "" };
    }
  },
}));

vi.mock("../adapters/openclaw/llm-runner.js", () => ({
  OpenClawLLMRunnerFactory: class MockFactory {
    constructor() {}
    createRunner(opts?: any) {
      return {
        async run(params: any) {
          if (params.taskId?.includes("l1")) {
            return { text: JSON.stringify({ summary: "mocked summary", score: 0.8 }) };
          }
          if (params.taskId?.includes("l15") || params.taskId?.includes("judge")) {
            return { text: JSON.stringify({ judgment: "continue", mmd_name: null }) };
          }
          if (params.taskId?.includes("l2")) {
            return { text: "flowchart TD\n  A[Task] --> B[Done]" };
          }
          return { text: "" };
        },
      };
    }
  },
}));

describe("OpenClawOffloadAdapter", () => {
  it("should instantiate without error", async () => {
    const { OpenClawOffloadAdapter } = await import("./openclaw-offload-adapter.js");
    const adapter = new OpenClawOffloadAdapter({
      config: {},
      modelRef: "github-copilot/gpt-5.5",
      logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    });
    expect(adapter).toBeDefined();
  });

  it("should expose l1Summarize method", async () => {
    const { OpenClawOffloadAdapter } = await import("./openclaw-offload-adapter.js");
    const adapter = new OpenClawOffloadAdapter({
      config: {},
      modelRef: "github-copilot/gpt-5.5",
      logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    });
    expect(typeof adapter.l1Summarize).toBe("function");
  });

  it("should expose l15Judge method", async () => {
    const { OpenClawOffloadAdapter } = await import("./openclaw-offload-adapter.js");
    const adapter = new OpenClawOffloadAdapter({
      config: {},
      modelRef: "github-copilot/gpt-5.5",
      logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    });
    expect(typeof adapter.l15Judge).toBe("function");
  });

  it("should expose l2Generate method", async () => {
    const { OpenClawOffloadAdapter } = await import("./openclaw-offload-adapter.js");
    const adapter = new OpenClawOffloadAdapter({
      config: {},
      modelRef: "github-copilot/gpt-5.5",
      logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    });
    expect(typeof adapter.l2Generate).toBe("function");
  });

  it("should log initialization", async () => {
    const logFn = vi.fn();
    const { OpenClawOffloadAdapter } = await import("./openclaw-offload-adapter.js");
    new OpenClawOffloadAdapter({
      config: {},
      modelRef: "github-copilot/gpt-5.5",
      logger: { info: logFn, warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    });
    expect(logFn).toHaveBeenCalled();
  });
});
