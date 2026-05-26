/**
 * L1 Summarization Prompt — English version.
 *
 * Converts tool call/result pairs into high-density JSON summaries.
 */

// ─── System Prompt ───────────────────────────────────────────────────────────

export const L1_SYSTEM_PROMPT = `You are a "Tool Result Summarizer" supporting an AI coding assistant. Your core task is to deeply understand the current conversation context and distill verbose tool calls and execution results (one tool_call + tool_result pair per summary entry) into a high-information-density JSON array.

Before generating summaries, perform the following internal reasoning:
1. Task Alignment: Based on recent conversation, identify the user's current core goal and latest intent. If context conflicts exist, always prioritize the most recent user intent.
2. Value Filtering: Ignore redundant details about how tools work. Directly extract "what key findings were discovered", "what key actions were taken", "what specific changes were made", or "what specific errors were encountered".
3. Impact Assessment: Determine the substantive impact on the current task (e.g., confirmed a hypothesis, advanced which step, made what decision, or what error caused a blockage).

【Output Format Requirements】
You must output ONLY a valid JSON object array [{...}]. Each object MUST contain the following fields:
- "tool_call": A concise description of the tool call. Processing rules:
  · If the tool pair is marked [NEEDS_COMPRESS], compress the tool name + key parameters into one concise description (≤150 characters), keeping the tool name, operation target (file path, command intent), omitting inline scripts/large content details.
    Example: exec({"command":"python3 -c 'import csv; ...200 lines...'"}) → "exec: Run Python script analyzing sales_channels.csv data quality"
    Example: write_file({"path":"/root/app.py","content":"...5000 chars..."}) → "write_file: Write /root/app.py (Flask main app file), content is..."
  · If not marked [NEEDS_COMPRESS], briefly describe the tool and parameters (system will use original values).
- "summary": A refined summary incorporating the above reasoning (≤200 characters). Must clearly state the business value of the result and its effect on task progress/blockage.
- "tool_call_id": The original tool_call_id (must be passed through as-is).
- "timestamp": The original ISO 8601 timestamp (must be passed through as-is).
- "score" (required): Assess how well the summary can substitute for the original based on information density and task relevance, range 0-10, where 10 means the summary fully replaces the original.

【Strict Rules】
Only output a pure JSON array. Never output reasoning or explanatory text.`;

// ─── Constants ───────────────────────────────────────────────────────────────

const PARAMS_MAX_LEN = 500;
const RESULT_MAX_LEN = 2000;
const COMPRESS_THRESHOLD = 200;

// ─── Types ───────────────────────────────────────────────────────────────────

export interface L1ToolPair {
  toolName: string;
  toolCallId: string;
  params: unknown;
  result: unknown;
  timestamp: string;
}

// ─── User Prompt Builder ─────────────────────────────────────────────────────

/**
 * Build the L1 user prompt for summarization.
 */
export function buildL1UserPrompt(recentMessages: string, pairs: L1ToolPair[]): string {
  const parts: string[] = [];

  parts.push("## Recent conversation context (for understanding the current task):");
  parts.push(recentMessages);
  parts.push("\n## Tool call/result pairs to summarize:");

  for (let i = 0; i < pairs.length; i++) {
    const p = pairs[i];
    const paramsStr = truncate(stringify(p.params), PARAMS_MAX_LEN);
    const resultStr = truncate(stringify(p.result), RESULT_MAX_LEN);
    const canonical = `${p.toolName}(${stringify(p.params)})`;
    const needsCompress = canonical.length > COMPRESS_THRESHOLD;

    parts.push(`--- Tool Pair ${i + 1} ---`);
    parts.push(`tool_call_id: ${p.toolCallId}`);
    parts.push(`timestamp: ${p.timestamp}`);
    if (needsCompress) {
      parts.push(`Tool: ${p.toolName} [NEEDS_COMPRESS]`);
    } else {
      parts.push(`Tool: ${p.toolName}`);
    }
    parts.push(`Params: ${paramsStr}`);
    parts.push(`Result: ${resultStr}\n`);
  }

  parts.push("Summarize each pair into the JSON array format described.");
  return parts.join("\n");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function stringify(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen) + "...";
}
