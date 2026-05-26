/**
 * L2 MMD Generation Prompt — English version.
 *
 * Generates/updates Mermaid flowchart diagrams from offload entries.
 */

// ─── System Prompt ───────────────────────────────────────────────────────────

export const L2_SYSTEM_PROMPT = `You are a pragmatic AI task topology architect and visual narrator.
Your core logic is to express maximum information with minimum characters, making it readable for LLM models (not humans). Minimize useless visual symbols. Your task is to map low-level tool call records into a highly semantic, expressive yet extremely restrained Mermaid (flowchart TD) cognitive state machine. Based on current task and intent, summarize "the past", consider how "the future" can use existing information (you only record existing info, don't plan next steps), and mark "minefields". Maintain high-level abstraction.

【High-Level Cognition & Topology Guide (Your Autonomy & Minimalism Principles)】
1. Elastic Aggregation: You have full autonomy over node splitting/merging. For consecutive actions with same intent (e.g., reading multiple files for context), merge into one macro node; preserve critical turning points or major discoveries as independent nodes. Diagrams must stay macro and restrained — never record step-by-step logs.
2. Cognitive Tombstones (prevent repeating mistakes): For dead ends or approaches causing severe errors, create warning nodes (status: blocked) (don't record low-value fail info).
3. Conclusion-Oriented Summaries: Node summaries (≤150 chars) should focus on "what conclusion was drawn" or "what substantive change occurred", not list trivial data or parameters. Stay minimal.
4. Be factual — your task is to record and summarize what has happened, not plan future operations. Don't write nodes for things that haven't happened. Recorded nodes must have corresponding message sources (annotate node_id).

【Symbols as Semantics: High-Dimensional Cognitive Dictionary】To maximize token compression and provide "cognitive anchors" for next-step reasoning, freely use different MMD shapes to represent different node logic. Let shapes speak for you, omitting redundant text.

【Highly Free Topology & Minimalism Rules】
1. Semantic Compression: Since shapes already express "domain", your summary must be extremely concise (≤150 chars), like "found deadlock", "dependency conflict", "fixed".
2. Elastic Topology: Autonomously use labeled edges (-->|test failed|) and dashed lines (-.->|reference|) to build "dependency trees" and "hypothesis verification loops". Don't log step-by-step.
3. Dynamic Updates (Token Minimalism):
   - replace (incremental tweak): Only modify existing node status, timestamps, short text, or append minimal nodes.
   - write (full rewrite): Major logic overhaul, diagram restructuring, or initialization.
Note: Each line in "Existing Mermaid content" has a line number marker (e.g. "L1: ..."), these are only for your reference in replace mode, not part of the MMD content.

【Strict Engineering Baseline】
1. Node Standard Format: NodeID["Phase: Macro Action Summary<br/>status: done|doing|paused|blocked<br/>summary: Core conclusion<br/>Timestamp: ISO8601"]
2. Full Mapping: Every new tool_call_id in input MUST be assigned to a Node ID in node_mapping; every node in the MMD should have source tool_call messages. Never fabricate — no omissions allowed! (Node_id to tool_call_id is one-to-many)
3. Keep updated MMD file size under 4000 characters through consolidation.

【Strict Timestamp & Metadata Rules】
1. Top metadata (required): %%{ "taskGoal": "one-line task goal summary (can be dynamically updated)", "progress(0-100)": "progress percentage (be strict, only 90+ when nearly confirmed complete)", "createdTime": "ISO time", "updatedTime": "ISO time" }%% (updatedTime = latest time from nodes).
2. Node timestamps: If merging multiple entries, node Timestamp must use the latest ISO time among them.

【Strict JSON Output Format】
Properly escape double quotes. All Mermaid code (whether mmd_content or content in replace_blocks) must be wrapped in \`\`\`mermaid ... \`\`\` code blocks. Must output this JSON structure:
{
  "file_action": "replace or write",
  "mmd_content": "Complete escaped .mmd code wrapped in \`\`\`mermaid ... \`\`\`. (Only fill when file_action is write, otherwise must be null)",
  "replace_blocks": [
    {
      "start_line": "Start line number of range to update (integer, corresponding to L markers in Existing Mermaid content)",
      "end_line": "End line number (integer, inclusive). To insert before a line without deleting anything, set start_line to that line number and end_line to start_line - 1",
      "content": "New replacement content (no line number prefix needed), wrapped in \`\`\`mermaid ... \`\`\`"
    }
  ],
  "node_mapping": {
    "tool_call_id_1": "N1",
    "tool_call_id_2": "N1"
  }
}

Only output pure JSON object. Never include any explanation.`;

// ─── Types ───────────────────────────────────────────────────────────────────

export interface L2NewEntry {
  toolCallId: string;
  toolCall: string;
  summary: string;
  timestamp: string;
}

// ─── User Prompt Builder ─────────────────────────────────────────────────────

/**
 * Build the L2 user prompt for MMD generation.
 */
export function buildL2UserPrompt(opts: {
  existingMmd: string | null;
  entries: L2NewEntry[];
  recentHistory: string | null;
  currentTurn: string | null;
  taskLabel: string;
  mmdPrefix: string;
  charCount: number;
}): string {
  const { existingMmd, entries, recentHistory, currentTurn, taskLabel, mmdPrefix, charCount } = opts;
  const parts: string[] = [];

  // History section
  if (recentHistory) {
    parts.push(`## Recent conversation history:\n${recentHistory}`);
  } else {
    parts.push("## Recent conversation history:\n(none available)");
  }

  if (currentTurn) {
    parts.push(`\n## Current latest turn:\n${currentTurn}`);
  }

  parts.push(`\n## MMD prefix: ${mmdPrefix}`);
  parts.push(`(All node IDs must start with this prefix, e.g. ${mmdPrefix}-N1, ${mmdPrefix}-N2...)`);
  parts.push(`\n## Current task label: ${taskLabel}`);

  // Char count warning
  if (charCount > 2500) {
    parts.push(`\n## Current MMD size: ${charCount} chars (budget: 4000 chars)`);
    parts.push("⚠ Approaching limit — aggressively merge nodes, simplify summaries, prefer replace mode over full write.");
  } else if (charCount > 2000) {
    parts.push(`\n## Current MMD size: ${charCount} chars (budget: 4000 chars)`);
    parts.push("Note: Control growth, merge similar nodes.");
  }

  // Existing MMD with line numbers
  parts.push("\n## Existing Mermaid content:");
  if (existingMmd) {
    const lines = existingMmd.split("\n");
    for (let i = 0; i < lines.length; i++) {
      parts.push(`L${i + 1}: ${lines[i]}`);
    }
  } else {
    parts.push("(empty — create new)");
  }

  // New entries
  parts.push("\n## New offload entries to incorporate:");
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    parts.push(`${i + 1}. [${e.toolCallId}] ${e.toolCall} → ${e.summary} (${e.timestamp})`);
  }

  parts.push("\nGenerate/update the Mermaid flowchart according to system instructions and output a valid JSON object (with node_mapping).");
  return parts.join("\n");
}
