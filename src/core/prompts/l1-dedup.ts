/**
 * L1 Conflict Detection Prompt (Batch Mode)
 *
 * Based on Kenty's validated prototype prompt (l1_conflict_detection_prompt.md).
 * Batch-compares multiple new memories against a unified candidate pool,
 * supporting cross-type merge and multi-target operations.
 */

import type { MemoryRecord, ExtractedMemory } from "../record/l1-writer.js";

// ============================
// System Prompt
// ============================

export const CONFLICT_DETECTION_SYSTEM_PROMPT = `You are a memory conflict detector. Batch-compare multiple 【New memories】 against existing memories in the 【Unified candidate memory pool】, deciding how to handle each one.

## Core Rules

- **Cross-type merging**: Memories of different types (persona / episodic / instruction) that semantically describe the same fact/event **can be merged**.
- **Many-to-many merging**: A single new memory can simultaneously replace/merge **multiple** existing memories from the candidate pool (specified via the target_ids array).
- After merging, you must determine the best type for the new memory (merged_type).

## Decision Logic

1. **Distinguish memory nature**:
   - **State-type** (persona/instruction): Preferences, traits, long-term settings, relatively stable facts, behavioral rules
   - **Event-type** (episodic): One-time experiences, objective records with timestamps; recommend merging cause and effect of the same event

2. **Determine if same fact/event**: Same subject, consistent topic, close in time, similar scene_name

3. **Choose action**:
   - "store": Treat as new information — add the current memory.
   - "skip": Existing memory is better; new memory has no incremental value or is more vague — ignore current memory.
   - "update": Same fact/event; new memory is superior in content or timing (more specific, more recent, or corrective) — overwrite old memory with new, may retain still-correct details from old memory.
   - "merge": Same fact or same evolutionary process; multiple memories contain complementary non-contradictory information — merge into one more complete memory with minimal redundancy.

4. **Strategy tendencies**:
   - State-type: Multiple entries describing same preference/trait → tend toward merge; no incremental value → skip; clear update → update
   - Event-type: Cause/effect or different stages of same event → tend toward merge into one complete narrative; completely identical → skip
   - Cross-type example: One episodic "The user started making podcasts in 2018" + one persona "The user has podcast production experience" → can merge into one persona or episodic (depending on information emphasis)

5. **Timestamp handling**:
   - For merge / update, merged_timestamps should contain the **union of all related memories' timestamps** (deduplicated and sorted)
   - This preserves the complete timeline of events

## Output Format

Strictly output a JSON array where each element corresponds to a decision for one new memory. Do not output anything else:

[
  {
    "record_id": "The new memory's record_id",
    "action": "store|update|skip|merge",
    "target_ids": ["candidate memory record_id 1 to delete", "record_id 2"],
    "merged_content": "Memory content after merge/update (required for merge/update)",
    "merged_type": "Best type after merge: persona|episodic|instruction (required for merge/update)",
    "merged_priority": 85,
    "merged_timestamps": ["Timestamp array after merge, union of all new and old memory timestamps (required for merge/update)"]
  }
]

Field descriptions:
- target_ids: **Array** of old memory IDs to delete/replace (can be 1 or more). Omit or leave empty for store/skip.
- merged_content: Final memory text for merge/update. Omit for store/skip.
- merged_type: The type the memory should belong to after merge/update. Determine based on the essence of merged content.
- merged_priority: New priority after merge/update (integer 0-100, required for merge/update). After merging, information is more complete and certain, so priority should usually be **moderately increased** (e.g., two memories with priority 70 can be raised to 80 after merging). Reference standard: 80-100 (core traits/important events), 60-79 (general preferences/ordinary activities), <60 (secondary information).
- merged_timestamps: Timestamp array after merge. Collect timestamps from new memory + all merged old memories, deduplicate and sort.`;

// ============================
// Prompt Builder
// ============================

/**
 * Candidate search result for a single new memory.
 */
export interface CandidateMatch {
  newMemory: ExtractedMemory & { record_id: string };
  candidates: MemoryRecord[];
}

/**
 * Format the batch conflict detection prompt using a unified candidate pool.
 *
 * Format (aligned with prototype):
 * 1. Unified candidate pool: de-duplicated list of all existing candidates across all new memories
 * 2. Per new memory: content + list of related candidate IDs from the pool
 *
 * This approach lets the LLM see the global picture and handle cross-memory dedup in one pass.
 *
 * @param matches - Array of new memories with their candidate matches
 */
export function formatBatchConflictPrompt(matches: CandidateMatch[]): string {
  // Step 1: Build unified candidate pool (de-duplicate across all new memories)
  const unifiedPool = new Map<string, MemoryRecord>();
  const perMemoryCandidateIds = new Map<string, string[]>();

  for (const m of matches) {
    const candidateIds: string[] = [];
    for (const c of m.candidates) {
      if (!unifiedPool.has(c.id)) {
        unifiedPool.set(c.id, c);
      }
      candidateIds.push(c.id);
    }
    perMemoryCandidateIds.set(m.newMemory.record_id, candidateIds);
  }

  // Step 2: Format unified pool as JSON
  const poolList = Array.from(unifiedPool.values()).map((c) => ({
    record_id: c.id,
    content: c.content,
    type: c.type,
    priority: c.priority,
    scene_name: c.scene_name,
    timestamps: c.timestamps,
  }));

  let poolSection: string;
  if (poolList.length === 0) {
    poolSection = "## Unified candidate memory pool\n\n(Empty — no existing memories. All new memories should be stored directly.)";
  } else {
    const poolStr = JSON.stringify(poolList, null, 2);
    poolSection = `## Unified candidate memory pool (${poolList.length} existing memories)\n\n${poolStr}`;
  }

  // Step 3: Format each new memory with its related candidate IDs
  const memoryParts = matches.map((m, idx) => {
    const relatedIds = perMemoryCandidateIds.get(m.newMemory.record_id) ?? [];
    const relatedNote =
      relatedIds.length > 0
        ? JSON.stringify(relatedIds)
        : "[](no similar candidates — store directly)";

    const memStr = JSON.stringify(
      {
        record_id: m.newMemory.record_id,
        content: m.newMemory.content,
        type: m.newMemory.type,
        priority: m.newMemory.priority,
        scene_name: m.newMemory.scene_name,
      },
      null,
      2,
    );

    return `### New memory #${idx + 1} (record_id: ${m.newMemory.record_id})\n${memStr}\n\n【Related candidate IDs】${relatedNote}`;
  });

  const newMemoriesText = memoryParts.join(
    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n",
  );

  // Step 4: Assemble final prompt
  return `${poolSection}

${"═".repeat(50)}

## New memories to evaluate (${matches.length} total)

${newMemoriesText}

Evaluate each memory and output a decision JSON array. When a new memory's candidate list is empty, output action=store for that entry.`;
}
