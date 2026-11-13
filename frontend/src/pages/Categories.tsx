import { useState } from "react";

import {
    useCategories,
    useCategoryAction,
    useCreateCategory,
    useMergeCategory,
    usePlan,
} from "../api/hooks";
import type { Category, CategoryStatus } from "../api/types";
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

const tabs: { value: CategoryStatus | ""; label: string }[] = [
    { value: "approved", label: "Approved" },
    { value: "proposed", label: "Proposed" },
    { value: "rejected", label: "Rejected" },
    { value: "merged", label: "Merged" },
    { value: "", label: "All" },
];

function ModelColumn({
    category,
    minTickets,
}: {
    category: Category;
    minTickets: number;
}) {
    if (category.in_model) return <Badge tone="green">In the model</Badge>;
    if (category.status !== "approved") return <span className="text-slate-400">–</span>;

    if (category.tickets_needed === 0) {
        return <Badge tone="indigo">Ready to train</Badge>;
    }
    return (
        <div className="w-40">
            <div className="mb-1 text-xs text-slate-500">
                Collecting examples: {category.tickets_available}/{minTickets}
            </div>
            <ProgressBar value={category.tickets_available} max={minTickets} />
        </div>
    );
}

function CategoryRow({
    category,
    approved,
    minTickets,
}: {
    category: Category;
    approved: Category[];
    minTickets: number;
}) {
    const action = useCategoryAction();
    const merge = useMergeCategory();
    const [merging, setMerging] = useState(false);
    const [target, setTarget] = useState("");

    const busy = action.isPending || merge.isPending;
    const error = action.error ?? merge.error;
    const mergeTargets = approved.filter((c) => c.id !== category.id);

    return (
        <tr className="border-b border-slate-100 align-top">
            <td className="px-3 py-2">
                <div className="font-medium">{category.name}</div>
                {category.description && (
                    <div className="text-xs text-slate-500">{category.description}</div>
                )}
                {error && (
                    <div className="mt-1">
                        <ErrorBox error={error} />
                    </div>
                )}
            </td>
            <td className="px-3 py-2">
                <StatusBadge status={category.status} />
            </td>
            <td className="px-3 py-2">
                <ModelColumn category={category} minTickets={minTickets} />
            </td>
            <td className="px-3 py-2 tabular-nums">{category.tickets_available}</td>
            <td className="px-3 py-2">
                {category.status === "proposed" && !merging && (
                    <div className="flex flex-wrap gap-1">
                        <Button
                            variant="primary"
                            disabled={busy}
                            onClick={() => action.mutate({ id: category.id, action: "approve" })}
                        >
                            Approve
                        </Button>
                        <Button
                            disabled={busy}
                            onClick={() => {
                                if (
                                    window.confirm(
                                        `Reject "${category.name}"? Its tickets go back to the review queue.`,
                                    )
                                ) {
                                    action.mutate({ id: category.id, action: "reject" });
                                }
                            }}
                        >
                            Reject
                        </Button>
                        <Button variant="ghost" disabled={busy} onClick={() => setMerging(true)}>
                            Merge…
                        </Button>
                    </div>
                )}

                {merging && (
                    <div className="flex flex-wrap gap-1">
                        <select
                            value={target}
                            onChange={(event) => setTarget(event.target.value)}
                            className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm"
                        >
                            <option value="">Merge into…</option>
                            {mergeTargets.map((c) => (
                                <option key={c.id} value={c.id}>
                                    {c.name}
                                </option>
                            ))}
                        </select>
                        <Button
                            variant="primary"
                            disabled={!target || busy}
                            onClick={() =>
                                merge.mutate(
                                    { id: category.id, intoId: Number(target) },
                                    { onSuccess: () => setMerging(false) },
                                )
                            }
                        >
                            Merge
                        </Button>
                        <Button variant="ghost" onClick={() => setMerging(false)}>
                            Cancel
                        </Button>
                    </div>
                )}
            </td>
        </tr>
    );
}

export default function Categories() {
    const categories = useCategories();
    const plan = usePlan();
    const create = useCreateCategory();
    const [tab, setTab] = useState<CategoryStatus | "">("approved");
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");

    if (categories.isLoading) return <Spinner />;
    if (categories.error) return <ErrorBox error={categories.error} />;

    const all = categories.data ?? [];
    const approved = all.filter((c) => c.status === "approved");
    const shown = tab ? all.filter((c) => c.status === tab) : all;
    const minTickets = plan.data?.min_tickets_new_category ?? 40;
    const proposedCount = all.filter((c) => c.status === "proposed").length;

    return (
        <>
            <PageHeader
                title="Categories"
                subtitle="Approve what agents propose. A new category is trained once it has enough confirmed tickets."
            />

            <div className="mb-4 flex flex-wrap gap-1">
                {tabs.map((t) => (
                    <Button
                        key={t.value}
                        variant={tab === t.value ? "primary" : "secondary"}
                        onClick={() => setTab(t.value)}
                    >
                        {t.label}
                        {t.value === "proposed" && proposedCount > 0 ? ` (${proposedCount})` : ""}
                    </Button>
                ))}
            </div>

            {shown.length === 0 ? (
                <Empty>No categories in this view.</Empty>
            ) : (
                <Card className="overflow-x-auto p-0">
                    <table className="w-full text-left text-sm">
                        <thead className="border-b border-slate-200 text-xs uppercase text-slate-500">
                            <tr>
                                <th className="px-3 py-2">Category</th>
                                <th className="px-3 py-2">Status</th>
                                <th className="px-3 py-2">Model</th>
                                <th className="px-3 py-2">Unused tickets</th>
                                <th className="px-3 py-2">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {shown.map((category) => (
                                <CategoryRow
                                    key={category.id}
                                    category={category}
                                    approved={approved}
                                    minTickets={minTickets}
                                />
                            ))}
                        </tbody>
                    </table>
                </Card>
            )}

            <Card title="Create a category directly" className="mt-6">
                <div className="flex flex-wrap gap-2">
                    <input
                        value={name}
                        onChange={(event) => setName(event.target.value)}
                        placeholder="Name"
                        className="w-56 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
                    />
                    <input
                        value={description}
                        onChange={(event) => setDescription(event.target.value)}
                        placeholder="Description (optional)"
                        className="min-w-0 flex-1 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
                    />
                    <Button
                        variant="primary"
                        disabled={!name.trim() || create.isPending}
                        onClick={() =>
                            create.mutate(
                                { name: name.trim(), description },
                                {
                                    onSuccess: () => {
                                        setName("");
                                        setDescription("");
                                    },
                                },
                            )
                        }
                    >
                        Create
                    </Button>
                </div>
                <div className="mt-2">
                    <ErrorBox error={create.error} />
                </div>
            </Card>
        </>
    );
}