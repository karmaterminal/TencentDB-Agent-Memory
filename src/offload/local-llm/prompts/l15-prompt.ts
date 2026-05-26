/**
 * L1.5 Task Judgment Prompt — English version.
 *
 * Determines task lifecycle: completion, continuation, new task detection.
 */

// ─── System Prompt ───────────────────────────────────────────────────────────

export const L15_SYSTEM_PROMPT = `You are a "Task Lifecycle Gatekeeper" for an AI coding assistant.
Your role is to cross-analyze the three provided input sources, accurately judge task state, and output a pure JSON object.

【Input Data Analysis Guide (mandatory reasoning chain)】
1. Step 1 - Analyze recentMessages (identify intent): From current and historical conversation, extract the user's latest core request. Determine if it's "continue troubleshooting", "declare completion (e.g., it works now)", "single-turn casual Q&A", or "start a brand new requirement".
2. Step 2 - Align with currentMmd (assess baseline): Compare the user's latest intent against the full Mermaid content of currentMmd — focus on taskGoal, each node's status (done/doing/todo), and summary. If the request is completely outside the current diagram's scope or all goals are achieved (all nodes done with no follow-up), then taskCompleted = true. If still solving sub-problems in the diagram (including doing nodes or fixing bugs), then false. (If there's no currentMmd, judge based on current and historical conversation alone.)
3. Step 3 - Search availableMmds (determine continuation): If judging to start a new task (isLongTask=true and taskCompleted=true/no current task), scan availableMmds taskGoal and time info. If the new request highly overlaps with an existing old task (e.g., returning to yesterday's unfinished module), it's a continuation (isContinuation=true).

【Strict JSON Output Format】
Must output a valid pure JSON object in this format:
{
  "taskCompleted": boolean, // Whether current task has ended (if currentMmd is none, must be true)
  "isLongTask": boolean,    // Whether latest request is a complex multi-step engineering task (simple Q&A/chat = false)
  "isContinuation": boolean, // Whether continuing a historical task from availableMmds
  "continuationMmdFile": "string|null", // If continuing old task, exact filename from availableMmds (no path prefix), else null
  "newTaskLabel": "string|null" // If new long task, generate short label (≤30 chars, kebab-case, e.g. "refactor-api"), else null
}

Only output pure JSON object. Never include explanatory text.`;

// ─── Types ───────────────────────────────────────────────────────────────────

export interface L15CurrentMmd {
  filename: string;
  content: string;
  path: string;
}

export interface L15MmdMeta {
  filename: string;
  path: string;
  taskGoal: string;
  doneCount: number;
  doingCount: number;
  todoCount: number;
  updatedTime?: string | null;
  nodeSummaries?: Array<{ nodeId: string; status: string; summary: string }>;
}

// ─── User Prompt Builder ─────────────────────────────────────────────────────

/**
 * Build the L1.5 user prompt for task judgment.
 */
export function buildL15UserPrompt(
  recentMessages: string,
  currentMmd: L15CurrentMmd | null,
  metas: L15MmdMeta[],
): string {
  const parts: string[] = [];

  parts.push("## 1. Recent conversation context (last 6 messages):");
  parts.push(recentMessages);
  parts.push("\n## 2. Currently mounted task graph (Active Mermaid — full content):");

  if (currentMmd && currentMmd.filename) {
    parts.push(`**File:** ${currentMmd.filename}`);
    if (currentMmd.path) {
      parts.push(`**Path:** \`${currentMmd.path}\``);
    }
    parts.push(`\n\`\`\`mermaid\n${currentMmd.content}\n\`\`\``);
  } else {
    parts.push("(none - currently idle, no active task)");
  }

  parts.push("\n## 3. Historical available task graphs (Available Mermaid task files):");

  if (metas.length === 0) {
    parts.push("(none - no historical long tasks)");
  } else {
    for (const m of metas) {
      parts.push(`- **${m.filename}**`);
      parts.push(`  path: \`${m.path}\``);
      parts.push(`  taskGoal: ${m.taskGoal}`);
      const total = m.doneCount + m.doingCount + m.todoCount;
      parts.push(`  progress: ${m.doneCount}/${total} done, ${m.doingCount} doing, ${m.todoCount} todo`);
      if (m.updatedTime) {
        parts.push(`  lastUpdated: ${m.updatedTime}`);
      }
      if (m.nodeSummaries && m.nodeSummaries.length > 0) {
        parts.push("  recentNodes:");
        for (const n of m.nodeSummaries) {
          parts.push(`    - [${n.nodeId}] (${n.status}) ${n.summary}`);
        }
      }
      parts.push("");
    }
  }

  parts.push("Analyze according to the system instruction's three-step reasoning chain and output a valid JSON object.");
  return parts.join("\n");
}
