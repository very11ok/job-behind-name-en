interface BatchOutcome {
  /** Number of objects actually written to the judgment JSON. */
  written: number;
  /** First object id of the batch. */
  first: string;
  /** Last object id of the batch. */
  last: string;
}

const DIR = "D:\\harness office\\research\\jobtask-agent\\out\\judge_input";
const OUTDIR = "D:\\harness office\\research\\jobtask-agent\\out\\judge_output";

const batches: string[] = [];
for (let i = 0; i < 34; i += 1) batches.push(`g1_batch_${String(i).padStart(3, "0")}`);
for (let i = 0; i < 26; i += 1) batches.push(`g2_batch_${String(i).padStart(3, "0")}`);
for (let i = 0; i < 6; i += 1) batches.push(`g3_batch_${String(i).padStart(3, "0")}`);

const SYS =
  "You are the antecedent judge of an organizational-economics prediction agent, performing antecedent judgments for the three re-bundling criteria on a job—task network." +
  "You judge object by object strictly according to the judgment protocol in the header of each batch file; do not bring in information outside the file, and do not fabricate ids." +
  "If the batch file cannot be read, the format does not match the protocol description, or the tasks are self-contradictory, state so honestly and return written=0 rather than guessing a result.";

const askText = (b: string, retryNote: string) =>
  `${retryNote}Use the Read tool to read the batch file ${DIR}\\${b}.txt (the header is the judgment protocol, the body is per-object blocks).\n` +
  `Judge the three fields for each object according to the protocol:\n` +
  `- holds: whether the antecedent holds, 0 or 1;\n` +
  `- p: the probability bin for the protocol-described consequence occurring for this object in the future, taking one of 0.05/0.1/0.2/0.3/0.5/0.7/0.9 only;\n` +
  `- why: rationale in ≤30 characters.\n` +
  `After judging, use the Write tool to write a strict JSON array to ${OUTDIR}\\${b}.json (UTF-8): one element per object, in the same order as the object blocks in the file, with field names id/holds/p/why, where id is copied verbatim from the value after id= in the first-line brackets of the object block.\n` +
  `After writing the file, reply with only one JSON object: {"written": <number of objects actually written>, "first": "<first id>", "last": "<last id>"}. Do not output the full judgment array in your reply, and do not modify the input file.`;

phase("LLM judges the three criteria's antecedents object by object");
log(`${batches.length} batches in total (Criterion 1: 34, Criterion 2: 26, Criterion 3: 6), up to 60 objects per batch`);
let okCount = 0;
const outcomes: { batch: string; written: number; status: string }[] = [];
await Promise.all(
  batches.map(async (b) => {
    const judge = agent(`judge-${b}`, { system: SYS });
    try {
      const r = await judge.ask<BatchOutcome>(askText(b, ""));
      const status = r.written > 0 ? "ok" : "empty";
      if (status === "ok") okCount += 1;
      report({ batch: b, written: r.written, status });
      outcomes.push({ batch: b, written: r.written, status });
    } catch {
      report({ batch: b, written: 0, status: "failed" });
      outcomes.push({ batch: b, written: 0, status: "failed" });
    }
  }),
);

phase("Re-run failed batches");
const failed = outcomes.filter((o) => o.status !== "ok");
log(`${outcomes.length - failed.length} batches succeeded, ${failed.length} batches need a re-run`);
const retried: string[] = [];
for (const f of failed) {
  const judge = agent(`backfill-${f.batch}`, { system: SYS });
  try {
    const r = await judge.ask<BatchOutcome>(
      askText(f.batch, `The previous judgment was not successfully written to disk (status ${f.status}). Please re-run the complete process.\n`),
    );
    if (r.written > 0) {
      okCount += 1;
      retried.push(f.batch);
      report({ batch: f.batch, written: r.written, status: "ok-retry" });
    } else {
      report({ batch: f.batch, written: 0, status: "failed-retry" });
    }
  } catch {
    report({ batch: f.batch, written: 0, status: "failed-retry" });
  }
}

phase("Summarize judgment progress");
const stillBad = failed.filter((f) => !retried.includes(f.batch)).map((f) => f.batch);
await artifact.markdown(
  "summary",
  [
    `# Three-criteria antecedent judgment progress`,
    "",
    `- Planned batches: ${batches.length}; successfully written to disk: ${okCount} (including ${retried.length} re-run batches).`,
    `- Still not written: ${stillBad.length === 0 ? "none" : stillBad.join(", ")}`,
    `- Output directory: out/judge_output/ (one JSON array per batch, fields id/holds/p/why)`,
  ].join("\n"),
  { title: "Antecedent judgment write-out checklist", description: "Completion status of the 66 batches of LLM antecedent judgments." },
);

return {
  conclusion: `Three-criteria antecedent judgment: ${okCount} of ${batches.length} batches were written to out/judge_output/, ${stillBad.length} failed (${stillBad.join(", ") || "none"}).`,
  verified: ["Each batch is judged and written to JSON by an independent subagent; a batch counts as successful only when written>0"],
  notCovered: ["Consistency checks between the judgment JSONs and the object-pool ids are done by the main session's merge scripts, not inside the workflow"],
};
