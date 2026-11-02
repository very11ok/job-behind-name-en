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

// NOTE: Chinese prompt kept verbatim (LLM-facing, part of the experiment record)
const SYS =
  "你是组织经济学预测 agent 的前因判定员，为岗位—任务网络上的三再捆绑准则做前因判定。" +
  "你严格按批次文件头部的判定协议逐对象判定，不引入文件之外的信息，不编造 id。" +
  "如果批次文件读不到、格式和协议描述不符、或任务自相矛盾，如实说明并返回 written=0，而不是猜一个结果。";

// NOTE: Chinese prompt kept verbatim (LLM-facing, part of the experiment record)
// (`retryNote` is prepended to the same LLM-facing message and is therefore also kept verbatim below)
const askText = (b: string, retryNote: string) =>
  `${retryNote}用 Read 工具读取批次文件 ${DIR}\\${b}.txt（头部是判定协议，正文是逐个对象块）。\n` +
  `逐个对象按协议判定三个字段：\n` +
  `- holds：前因是否成立，0 或 1；\n` +
  `- p：该对象未来发生协议所述后果的概率档位，只取 0.05/0.1/0.2/0.3/0.5/0.7/0.9 之一；\n` +
  `- why：≤30 字中文依据。\n` +
  `判定完成后，用 Write 工具把严格 JSON 数组写入 ${OUTDIR}\\${b}.json（UTF-8）：每个对象一个元素，顺序与文件中的对象块一致，字段名为 id/holds/p/why，id 照抄对象块第一行方括号里 id= 后面的值。\n` +
  `写完文件后，回复只含一个 JSON 对象：{"written": <实际写入的对象数>, "first": "<首个id>", "last": "<末个id>"}。不要在回复里输出判定数组全文，不要修改输入文件。`;

phase("LLM judges the three criteria's antecedents object by object");
log(`${batches.length} batches in total (Criterion 1: 34, Criterion 2: 26, Criterion 3: 6), up to 60 objects per batch`);
let okCount = 0;
const outcomes: { batch: string; written: number; status: string }[] = [];
await Promise.all(
  batches.map(async (b) => {
    // Subagent name template kept verbatim ("判定员" = judge); program identifier, not translated
    const judge = agent(`判定员-${b}`, { system: SYS });
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
  // Subagent name template kept verbatim ("补标员" = backfill judge); program identifier, not translated
  const judge = agent(`补标员-${f.batch}`, { system: SYS });
  try {
    const r = await judge.ask<BatchOutcome>(
      // NOTE: Chinese retry note kept verbatim (LLM-facing, part of the experiment record)
      askText(f.batch, `上一次判定没有成功落盘（状态 ${f.status}）。请重新完整执行一遍。\n`),
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
