import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import {
    useJobs,
    useModels,
    usePlan,
    usePromote,
    useRollback,
    useStartJob,
} from "../api/hooks";
import type { Gate, Job, ModelInfo } from "../api/types";
import {
    Badge,
    Button,
    Card,
    Empty,
    ErrorBox,
    PageHeader,
    ProgressBar,
    Spinner,
    StatusBadge,
} from "../components/ui";
import { pct, time } from "../format";

/** A positive "drop" means the model got worse. */
function change(drop: number | null): string {
    if (drop === null) return "–";
    const points = -drop * 100;
    return `${points > 0 ? "+" : ""}${points.toFixed(1)} pts`;
}

function GateChecks({ gate }: { gate: Gate }) {
    return (
        <ul className="space-y-1 text-sm">
            {gate.checks.map((check) => (
                <li key={check.name} className="flex gap-2">
                    <span className={check.passed ? "text-green-600" : "text-red-600"}>
                        {check.passed ? "✓" : "✗"}
                    </span>
                    <span>{check.detail}</span>
                </li>
            ))}
        </ul>
    );
}

function GateBadge({ gate }: { gate: Gate | null }) {
    if (!gate) return <span className="text-slate-400">–</span>;
    return gate.passed ? (
        <Badge tone="green">Gate passed</Badge>
    ) : (
        <Badge tone="red">Gate failed</Badge>
    );
}

// ---- training --------------------------------------------------------------
function TrainingCard({ running }: { running: boolean }) {
    const plan = usePlan();
    const start = useStartJob();

    if (plan.isLoading) return <Spinner />;
    if (plan.error || !plan.data) return <ErrorBox error={plan.error} />;

    const { data } = plan;
    const refresh = data.refresh_categories ?? [];

    return (
        <Card
            title="Training"
            actions={
                <Button
                    variant="primary"
                    disabled={!data.can_start || running || start.isPending}
                    onClick={() => start.mutate()}
                >
                    {running ? "Training in progress…" : "Start training"}
                </Button>
            }
        >
            {data.can_start ? (
                <p className="text-sm text-slate-600">
                    The next run would use {data.n_tickets} confirmed tickets.
                </p>
            ) : (
                <p className="text-sm text-slate-600">
                    {data.reason ?? "Nothing is ready to train."}
                </p>
            )}

            {data.ready_categories.length > 0 && (
                <div className="mt-3">
                    <div className="mb-1 text-xs font-medium uppercase text-slate-500">
                        New categories ready
                    </div>
                    <div className="flex flex-wrap gap-1">
                        {data.ready_categories.map((c) => (
                            <Badge key={c.name} tone="indigo">
                                {c.name} ({c.tickets})
                            </Badge>
                        ))}
                    </div>
                </div>
            )}

            {refresh.length > 0 && (
                <div className="mt-3">
                    <div className="mb-1 text-xs font-medium uppercase text-slate-500">
                        Existing categories with enough new tickets
                    </div>
                    <div className="flex flex-wrap gap-1">
                        {refresh.map((c) => (
                            <Badge key={c.name} tone="blue">
                                {c.name} ({c.tickets})
                            </Badge>
                        ))}
                    </div>
                </div>
            )}

            {data.collecting.length > 0 && (
                <div className="mt-3 space-y-2">
                    <div className="text-xs font-medium uppercase text-slate-500">
                        Collecting examples
                    </div>
                    {data.collecting.map((c) => (
                        <div key={c.name}>
                            <div className="mb-1 flex justify-between text-sm">
                                <span>{c.name}</span>
                                <span className="text-slate-500">
                                    {c.tickets}/{c.needed}
                                </span>
                            </div>
                            <ProgressBar value={c.tickets} max={c.needed} />
                        </div>
                    ))}
                </div>
            )}

            <div className="mt-3">
                <ErrorBox error={start.error} />
            </div>
        </Card>
    );
}

function JobRow({ job }: { job: Job }) {
    const [open, setOpen] = useState(false);
    const summary = job.summary;

    return (
        <li className="rounded-md border border-slate-200 p-3">
            <div className="flex flex-wrap items-center gap-2 text-sm">
                <span className="font-medium">Job #{job.id}</span>
                <StatusBadge status={job.status} />
                {summary?.gate && <GateBadge gate={summary.gate} />}
                <span className="text-slate-500">
                    {job.base_version ?? "–"} → {job.version ?? "–"} · {job.n_tickets} tickets
                </span>
                <span className="ml-auto text-xs text-slate-500">{time(job.created_at)}</span>
                {summary && (
                    <Button variant="ghost" onClick={() => setOpen(!open)}>
                        {open ? "Hide" : "Details"}
                    </Button>
                )}
            </div>

            {job.error && (
                <div className="mt-2">
                    <ErrorBox error={new Error(job.error)} />
                </div>
            )}

            {open && summary && (
                <div className="mt-3 space-y-2 text-sm">
                    {summary.new_categories && summary.new_categories.length > 0 && (
                        <div>New categories: {summary.new_categories.join(", ")}</div>
                    )}
                    <div className="text-slate-600">
                        Existing categories: {change(summary.old_accuracy_drop)}
                        {summary.new_category_accuracy !== null &&
                            ` · new category accuracy ${pct(summary.new_category_accuracy, 0)}`}
                    </div>
                    {summary.gate && <GateChecks gate={summary.gate} />}
                </div>
            )}
        </li>
    );
}

// ---- versions --------------------------------------------------------------
function VersionRow({ model }: { model: ModelInfo }) {
    const promote = usePromote();
    const [open, setOpen] = useState(false);

    const onPromote = () => {
        const failed = model.gate !== null && !model.gate.passed;
        const message = failed
            ? `The promotion gate FAILED for ${model.name}:\n\n${model.gate?.checks
                .filter((c) => !c.passed)
                .map((c) => `• ${c.detail}`)
                .join("\n")}\n\nPromote it anyway?`
            : `Promote ${model.name}? It starts serving tickets right away.`;

        if (window.confirm(message)) {
            promote.mutate({ name: model.name, force: failed });
        }
    };

    const calibration = model.calibration;

    return (
        <>
            <tr className="border-b border-slate-100 align-top">
                <td className="px-3 py-2 font-medium">
                    {model.name} {model.active && <Badge tone="green">Active</Badge>}
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-slate-500">
                    {time(model.created_at)}
                </td>
                <td className="px-3 py-2">{model.n_categories ?? "–"}</td>
                <td className="px-3 py-2">{pct(model.accuracy)}</td>
                <td className="px-3 py-2">{change(model.old_accuracy_drop)}</td>
                <td className="px-3 py-2">{pct(model.new_category_accuracy, 0)}</td>
                <td className="px-3 py-2">
                    <GateBadge gate={model.gate} />
                </td>
                <td className="px-3 py-2">
                    <div className="flex gap-1">
                        {!model.active && model.complete && (
                            <Button
                                variant={model.gate && !model.gate.passed ? "secondary" : "primary"}
                                disabled={promote.isPending}
                                onClick={onPromote}
                            >
                                Promote
                            </Button>
                        )}
                        <Button variant="ghost" onClick={() => setOpen(!open)}>
                            {open ? "Hide" : "Details"}
                        </Button>
                    </div>
                </td>
            </tr>
            {(open || promote.error) && (
                <tr className="border-b border-slate-100 bg-slate-50">
                    <td colSpan={8} className="p-0">
                        <div className={`grid transition-all duration-300 ease-in-out ${open || promote.error ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}>
                            <div className="overflow-hidden">
                                <div className="space-y-2 px-3 py-3 text-sm">
                                    <ErrorBox error={promote.error} />
                                    {open && (
                                        <>
                                            <div className="text-slate-600">
                                                {model.notes || "No notes."}
                                                {model.parent && ` Parent: ${model.parent}.`}
                                            </div>
                                            {model.new_categories.length > 0 && (
                                                <div>Added: {model.new_categories.join(", ")}</div>
                                            )}
                                            {model.gate && <GateChecks gate={model.gate} />}
                                            {calibration.threshold !== undefined && (
                                                <div className="text-slate-600">
                                                    Auto-label threshold{" "}
                                                    {calibration.threshold === null
                                                        ? "off"
                                                        : pct(calibration.threshold, 0)}
                                                    {calibration.auto_rate !== undefined &&
                                                        ` · about ${pct(calibration.auto_rate, 0)} of tickets auto-labeled`}
                                                    {calibration.temperature !== undefined &&
                                                        ` · temperature ${calibration.temperature.toFixed(2)}`}
                                                </div>
                                            )}
                                        </>
                                    )}
                                </div>
                            </div>
                        </div>
                    </td>
                </tr>
            )}
        </>
    );
}

// ---- page ------------------------------------------------------------------
export default function Models() {
    const queryClient = useQueryClient();
    const models = useModels();
    const jobs = useJobs();
    const rollback = useRollback();

    const running =
        jobs.data?.some((j) => j.status === "queued" || j.status === "running") ??
        false;

    // When a job finishes, refresh everything (new version, new plan).
    const wasRunning = useRef(false);
    useEffect(() => {
        if (wasRunning.current && !running) queryClient.invalidateQueries();
        wasRunning.current = running;
    }, [running, queryClient]);

    if (models.isLoading || jobs.isLoading) return <Spinner />;
    if (models.error) return <ErrorBox error={models.error} />;

    const versions = models.data ?? [];

    return (
        <>
            <PageHeader
                title="Model & training"
                subtitle="Train a new version, check it, then decide whether it goes live."
            >
                <Button
                    disabled={versions.length < 2 || rollback.isPending}
                    onClick={() => {
                        if (window.confirm("Go back to the previously promoted version?")) {
                            rollback.mutate();
                        }
                    }}
                >
                    Roll back
                </Button>
            </PageHeader>

            <div className="space-y-4">
                <ErrorBox error={rollback.error} />

                <TrainingCard running={running} />

                <Card title="Recent training jobs">
                    {jobs.error && <ErrorBox error={jobs.error} />}
                    {jobs.data && jobs.data.length === 0 ? (
                        <Empty>No training runs yet.</Empty>
                    ) : (
                        <ul className="space-y-2">
                            {jobs.data?.map((job) => <JobRow key={job.id} job={job} />)}
                        </ul>
                    )}
                </Card>

                <Card title="Versions" className="overflow-x-auto p-0">
                    <table className="w-full text-left text-sm">
                        <thead className="border-b border-slate-200 text-xs uppercase text-slate-500">
                            <tr>
                                <th className="px-3 py-2">Version</th>
                                <th className="px-3 py-2">Created</th>
                                <th className="px-3 py-2">Categories</th>
                                <th className="px-3 py-2">Accuracy</th>
                                <th className="px-3 py-2">Old categories</th>
                                <th className="px-3 py-2">New category</th>
                                <th className="px-3 py-2">Gate</th>
                                <th className="px-3 py-2" />
                            </tr>
                        </thead>
                        <tbody>
                            {versions.map((model) => (
                                <VersionRow key={model.name} model={model} />
                            ))}
                        </tbody>
                    </table>
                </Card>
            </div>
        </>
    );
}