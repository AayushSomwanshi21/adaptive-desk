import { useState } from "react";

import {
    useCategories,
    useLabelTicket,
    useTickets,
} from "../api/hooks";
import type { Category, Ticket } from "../api/types";
import {
    Badge,
    Button,
    Card,
    ConfidenceBar,
    Empty,
    ErrorBox,
    PageHeader,
    Spinner,
} from "../components/ui";
import { normalize, pct } from "../format";

function TicketReview({
    ticket,
    categories,
}: {
    ticket: Ticket;
    categories: Category[];
}) {
    const label = useLabelTicket();
    const [otherId, setOtherId] = useState("");
    const [newName, setNewName] = useState("");
    const [newDescription, setNewDescription] = useState("");

    const byName = new Map(categories.map((c) => [c.name, c]));
    const typed = normalize(newName);
    const similar =
        typed.length >= 3
            ? categories
                .filter((c) => {
                    const name = normalize(c.name);
                    return name.includes(typed) || typed.includes(name);
                })
                .slice(0, 5)
            : [];

    const choose = (categoryId: number) =>
        label.mutate({ ticketId: ticket.id, body: { category_id: categoryId } });

    const propose = () =>
        label.mutate({
            ticketId: ticket.id,
            body: {
                new_category: { name: newName.trim(), description: newDescription },
            },
        });

    return (
        <div className="space-y-4">
            <Card>
                <div className="mb-2 flex flex-wrap items-center gap-2">
                    <span className="text-xs text-slate-500">Ticket #{ticket.id}</span>
                    {ticket.is_spot_check && (
                        <Badge tone="indigo">Spot check: the model was confident</Badge>
                    )}
                    {ticket.model_version && (
                        <span className="text-xs text-slate-500">
                            predicted by {ticket.model_version}
                        </span>
                    )}
                </div>
                <p className="whitespace-pre-wrap text-base">{ticket.text}</p>
            </Card>

            <Card title="Model's guesses">
                {ticket.top_predictions.length === 0 ? (
                    <p className="text-sm text-slate-500">
                        The model made no prediction for this ticket.
                    </p>
                ) : (
                    <ul className="space-y-2">
                        {ticket.top_predictions.map((guess) => {
                            const category = byName.get(guess.label);
                            return (
                                <li
                                    key={guess.label}
                                    className="flex items-center justify-between gap-3"
                                >
                                    <div>
                                        <div className="text-sm font-medium">{guess.label}</div>
                                        <ConfidenceBar value={guess.confidence} />
                                    </div>
                                    <Button
                                        variant="primary"
                                        disabled={!category || label.isPending}
                                        onClick={() => category && choose(category.id)}
                                    >
                                        Accept
                                    </Button>
                                </li>
                            );
                        })}
                    </ul>
                )}
            </Card>

            <Card title="Another existing category">
                <div className="flex gap-2">
                    <select
                        value={otherId}
                        onChange={(event) => setOtherId(event.target.value)}
                        className="min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
                    >
                        <option value="">Choose a category…</option>
                        {categories.map((category) => (
                            <option key={category.id} value={category.id}>
                                {category.name}
                                {category.status === "proposed" ? " (proposed)" : ""}
                            </option>
                        ))}
                    </select>
                    <Button
                        disabled={!otherId || label.isPending}
                        onClick={() => choose(Number(otherId))}
                    >
                        Use this
                    </Button>
                </div>
            </Card>

            <Card title="None of these fit: propose a new category">
                <div className="space-y-2">
                    <input
                        value={newName}
                        onChange={(event) => setNewName(event.target.value)}
                        placeholder="Category name"
                        className="w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
                    />
                    {similar.length > 0 && (
                        <div className="rounded-md bg-amber-50 p-2 text-xs text-amber-900">
                            Similar categories already exist. Use one of them instead?
                            <div className="mt-1 flex flex-wrap gap-1">
                                {similar.map((category) => (
                                    <button
                                        key={category.id}
                                        onClick={() => choose(category.id)}
                                        className="rounded border border-amber-300 bg-white px-2 py-0.5 hover:bg-amber-100"
                                    >
                                        {category.name}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                    <textarea
                        value={newDescription}
                        onChange={(event) => setNewDescription(event.target.value)}
                        placeholder="One line describing what belongs in it (optional)"
                        rows={2}
                        className="w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
                    />
                    <Button
                        disabled={!newName.trim() || label.isPending}
                        onClick={propose}
                    >
                        Propose category
                    </Button>
                    <p className="text-xs text-slate-500">
                        An admin has to approve it before the model learns it.
                    </p>
                </div>
            </Card>

            <ErrorBox error={label.error} />
        </div>
    );
}

export default function ReviewQueue() {
    const tickets = useTickets({
        status: "needs_review",
        sort: "lowest_confidence",
        limit: 50,
    });
    const categories = useCategories();
    const [selectedId, setSelectedId] = useState<number | null>(null);

    if (tickets.isLoading || categories.isLoading) return <Spinner />;
    if (tickets.error) return <ErrorBox error={tickets.error} />;
    if (categories.error) return <ErrorBox error={categories.error} />;

    const items = tickets.data?.items ?? [];
    const total = tickets.data?.total ?? 0;
    const selected = items.find((t) => t.id === selectedId) ?? items[0];
    const usable = (categories.data ?? []).filter(
        (c) => c.status === "approved" || c.status === "proposed",
    );

    return (
        <>
            <PageHeader
                title="Review queue"
                subtitle={`${total} ticket${total === 1 ? "" : "s"} waiting, least confident first.`}
            />

            {!selected ? (
                <Empty>Nothing to review. New low-confidence tickets will show up here.</Empty>
            ) : (
                <div className="grid gap-4 lg:grid-cols-[22rem_1fr]">
                    <ul className="max-h-[75vh] space-y-2 overflow-y-auto pr-1">
                        {items.map((ticket) => (
                            <li key={ticket.id}>
                                <button
                                    onClick={() => setSelectedId(ticket.id)}
                                    className={`w-full rounded-lg border p-3 text-left text-sm ${ticket.id === selected.id
                                            ? "border-indigo-400 bg-indigo-50"
                                            : "border-slate-200 bg-white hover:bg-slate-50"
                                        }`}
                                >
                                    <div className="line-clamp-2">{ticket.text}</div>
                                    <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                                        <span>#{ticket.id}</span>
                                        <span>
                                            {ticket.confidence === null
                                                ? "no prediction"
                                                : `confidence ${pct(ticket.confidence, 0)}`}
                                        </span>
                                        {ticket.is_spot_check && <Badge tone="indigo">spot check</Badge>}
                                    </div>
                                </button>
                            </li>
                        ))}
                    </ul>

                    <TicketReview key={selected.id} ticket={selected} categories={usable} />
                </div>
            )}
        </>
    );
}