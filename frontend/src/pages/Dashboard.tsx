import { useStats } from "../api/hooks";
import { Card, ErrorBox, PageHeader, ProgressBar, Spinner } from "../components/ui";
import { pct } from "../format";

function Stat({
    label,
    value,
    hint,
}: {
    label: string;
    value: string;
    hint?: string;
}) {
    return (
        <Card>
            <div className="text-xs uppercase tracking-wide text-slate-800">{label}</div>
            <div className="mt-1 text-2xl font-semibold">{value}</div>
            {hint && <div className="mt-1 text-xs text-slate-600">{hint}</div>}
        </Card>
    );
}

const statusLabels: Record<string, string> = {
    needs_review: "Needs review",
    auto_labeled: "Auto-labeled",
    confirmed: "Confirmed by a human",
};

export default function Dashboard() {
    const { data, error, isLoading } = useStats();

    if (isLoading) return <Spinner />;
    if (error || !data) return <ErrorBox error={error} />;

    const { model, tickets, spot_checks, categories } = data;
    const threshold = model.threshold;

    return (
        <>
            <PageHeader
                title="Dashboard"
                subtitle="How much work the model takes off the review queue."
            />

            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <Stat
                    label="Active model"
                    value={model.version ?? "None"}
                    hint={model.n_categories ? `${model.n_categories} categories` : undefined}
                />
                <Stat label="Tickets" value={String(tickets.total)} />
                <Stat
                    label="Review queue"
                    value={String(tickets.review_queue)}
                    hint="waiting for a human"
                />
                <Stat
                    label="Auto-labeled"
                    value={pct(tickets.auto_label_rate, 0)}
                    hint="of tickets the model has seen"
                />
                <Stat
                    label="Average confidence"
                    value={pct(tickets.average_confidence, 0)}
                />
                <Stat
                    label="Spot-check agreement"
                    value={pct(spot_checks.agreement, 0)}
                    hint={`${spot_checks.reviewed} reviewed; humans agreeing with the model`}
                />
                <Stat
                    label="Auto-label threshold"
                    value={threshold === null || threshold === undefined ? "Off" : pct(threshold, 0)}
                    hint={
                        threshold === null || threshold === undefined
                            ? "no calibrated threshold: everything is reviewed"
                            : "confidence needed to skip review"
                    }
                />
                <Stat
                    label="Categories"
                    value={String(categories.approved ?? 0)}
                    hint={`${categories.proposed ?? 0} proposed`}
                />
            </div>

            <Card title="Tickets by status" className="mt-4">
                <div className="space-y-3">
                    {Object.entries(statusLabels).map(([key, label]) => (
                        <div key={key}>
                            <div className="mb-1 flex justify-between text-sm">
                                <span>{label}</span>
                                <span className="text-slate-900">{tickets.by_status[key] ?? 0}</span>
                            </div>
                            <ProgressBar
                                value={tickets.by_status[key] ?? 0}
                                max={tickets.total}
                            />
                        </div>
                    ))}
                </div>
            </Card>
        </>
    );
}