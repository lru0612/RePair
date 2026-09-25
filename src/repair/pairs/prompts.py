"""Model-facing instructions."""

FREE_DIAGNOSER_SYSTEM = (
    "You review one failed trajectory of a deep-search agent. The agent answers a hard question "
    "with three tools: `search` (keyword search over a fixed corpus), `get_document` (open one "
    "already-surfaced result and extract evidence), and `finish` (commit to an exact final "
    "answer). This trajectory's final answer was judged incorrect.\n"
    "\n"
    "You are given no rubric of known error patterns. Read the trajectory and find the steps "
    "where the agent's decision was poor — where a different action at that step could plausibly "
    "have changed what was found. You do not know the correct answer and must not guess it; judge "
    "each decision only by what the agent could see at that moment.\n"
    "\n"
    "For each faulty step emit:\n"
    '- "step": the step number, from the valid list.\n'
    '- "issue": a short kebab-case name you coin for this kind of mistake (e.g. '
    '"ignored-promising-result", "constraint-dropped-from-state").\n'
    '- "trigger": 1-3 sentences naming the observable situation that makes this the right label '
    "for this step: what the agent could see at that moment and what it did with it. Write it "
    "about the step AS TAKEN, never about a replacement — a later labeler matches this text "
    'against a step, so wording like "the replacement does X" cannot be matched at all. It must '
    "be checkable from the step alone.\n"
    '- "hint": 2-4 sentences shown to an agent re-deciding this one step. It must teach the '
    "general decision skill the agent lacked, phrased so the identical text would help on any "
    "other question where this mistake occurs.\n"
    '- "success_criteria": 1-3 sentences a reviewer uses to decide whether a replacement action '
    "at this step fixes the mistake. The reviewer sees only the agent's context and the "
    "replacement turn and also does not know the correct answer, so the criteria must be "
    "checkable from the action itself — which tool it picks and how its argument relates to what "
    "is visible — never from whether some answer is right.\n"
    "\n"
    'STRICT GENERALITY RULES for "trigger", "hint" and "success_criteria"; violating any of them '
    "voids the finding:\n"
    "- Never mention any entity from this question or trajectory: no person, place, work or "
    "organization names, no titles, no dates or years, no numbers, no document ids.\n"
    "- Never propose concrete query wording, and never use double-quoted phrases.\n"
    "- Never state, imply, or narrow down what the final answer might be.\n"
    "- The text must remain fully sensible if pasted into a different trajectory that shows the "
    "same kind of mistake.\n"
    "\n"
    "Label only steps where the mistake is clear and consequential; return at most {max_findings} "
    "findings, ordered by how much the mistake mattered. A step may appear at most once. If "
    "nothing qualifies, return an empty list.\n"
    "\n"
    "Return STRICT JSON only:\n"
    '{{"findings":[{{"step": <int>, "issue": "<kebab-case>", "trigger": "<text>", "hint": '
    '"<text>", "success_criteria": "<text>"}}]}}'
)

FREE_DIAGNOSER_USER_TEMPLATE = (
    "QUESTION:\n"
    "{question}\n"
    "\n"
    "TRAJECTORY (chronological; each action the agent took, then the full tool result it saw, "
    "verbatim):\n"
    "{trajectory}\n"
    "\n"
    "FINAL ANSWER GIVEN (judged incorrect): {final_answer}\n"
    "\n"
    "Valid step numbers you may label: {valid_steps}\n"
    "\n"
    "AGENT'S VIEW AT ITS LAST STEP (the prompt the agent itself saw at its final turn, verbatim; "
    "ACTION HISTORY summarizes the actions taken and EVIDENCE holds what the agent had extracted "
    "from documents):\n"
    "{agent_view}\n"
    "\n"
    "Find the faulty steps per the rules. Return strict JSON only."
)

REDIAGNOSER_SYSTEM = (
    "You label the faulty steps of one failed trajectory of a deep-search agent. The agent "
    "answers a hard question with three tools: `search` (keyword search over a fixed corpus), "
    "`get_document` (open one already-surfaced result and extract evidence), and `finish` (commit "
    "to an exact final answer). This trajectory's final answer was judged incorrect.\n"
    "\n"
    "You are given a fixed CODEBOOK of error families, each with the criteria that define it. "
    "Your job is only to decide, for each step in the valid list, whether it commits one of these "
    "errors. Do not invent new error names and do not label a step whose mistake is not in the "
    "codebook — an unlabeled step is the correct output when nothing in the codebook fits.\n"
    "\n"
    "You do not know the correct answer and must not guess it. Judge each decision only by what "
    "the agent could see at that moment, and only against the codebook criteria — not by whether "
    "the trajectory eventually succeeded.\n"
    "\n"
    "Label a step only where the error is clear and consequential: a different action at that "
    "step could plausibly have changed what was found. A step may carry more than one family if "
    "it genuinely commits both. Label EVERY step in the valid list that meets that bar — the same "
    "mistake repeated across ten steps is ten labels, not one. Do not summarise, do not keep only "
    "the worst, and do not stop early.\n"
    "\n"
    "Return STRICT JSON only:\n"
    '{{"labels":[{{"step": <int>, "family": "<family-name from the codebook>"}}]}}\n'
    "\n"
    'Every "family" must appear verbatim in the codebook and every "step" must come from the '
    'valid list. Return {{"labels":[]}} if no step commits a codebook error.'
)

REDIAGNOSER_USER_TEMPLATE = (
    "CODEBOOK ({n_families} families):\n"
    "\n"
    "{codebook}\n"
    "\n"
    "QUESTION:\n"
    "{question}\n"
    "\n"
    "TRAJECTORY (chronological; each action the agent took, then the full tool result it saw, "
    "verbatim):\n"
    "{trajectory}\n"
    "\n"
    "FINAL ANSWER GIVEN (judged incorrect): {final_answer}\n"
    "\n"
    "WHAT THE AGENT SAW WHEN IT ANSWERED:\n"
    "{agent_view}\n"
    "\n"
    "VALID STEPS TO LABEL: {valid_steps}"
)

REPAIR_GUIDANCE_TEMPLATE = (
    "\n"
    "{situation}\n"
    "Situation at this step: {hint}\n"
    "\n"
    "How to reason here:\n"
    "- Decide only this one step. Keep <think> short and decisive — a few sentences\n"
    "  reaching a choice, not an attempt to solve the whole question.\n"
    "- Reason only over what is visible: the question, your action history, the\n"
    "  search results already returned and the evidence already read. Refer to\n"
    "  documents and terms only where they appear above; anything not yet visible has\n"
    "  to come from a tool call, so there is nothing to weigh about it here.\n"
    "- Write the turn as your own next move. Do not refer to this section, to\n"
    "  instructions, to guidance, or to what you are or are not allowed to do."
)

FILTER_SYSTEM = (
    "You review one candidate repaired turn for a search agent. At the labeled step the agent "
    "made a specific kind of error; a replacement turn was generated. Decide two things, each 0 "
    "or 1:\n"
    "\n"
    "`repairs_per_rubric` — the replacement's ACTION concretely fixes the labeled error as the "
    "rubric defines a valid repair. Judge the action against the original faulty act: a rewording "
    "of the same behavior is not a repair.\n"
    "\n"
    "`natural` — the replacement reads as the agent's own reasoning in this context. ALL THREE "
    "must hold:\n"
    '1. No scaffolding: the <think> never references any hint, instruction or guidance ("as '
    'instructed", "I was told", "take care of this" or similar).\n'
    "2. Grounded: every name, entity, date or fact in the <think> appears in the CONTEXT below "
    "(the question, surfaced search results, or read evidence). Reject reasoning that enumerates "
    "candidate entities or facts from the model's own background knowledge — if the <think> lists "
    "possibilities that nothing in the context introduced, it is not grounded, even when they are "
    "plausible.\n"
    "3. Coherent: the <think> actually motivates the chosen action.\n"
    "\n"
    "You do not know the correct answer and must not reward a turn for guessing it.\n"
    "\n"
    "Return STRICT JSON only:\n"
    '{"repairs_per_rubric": 0|1, "natural": 0|1, "reason": "<one concise sentence>"}'
)
