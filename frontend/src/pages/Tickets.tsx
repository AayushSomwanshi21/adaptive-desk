import { useState } from "react";

import { useTickets } from "../api/hooks";
import type { TicketStatus } from "../api/types";
import {
    Button,
    Card,
    Empty,
    ErrorBox,
    PageHeader,
    Spinner,
    StatusBadge,
} from "../components/ui";
import { pct, time } from "../format";

const PAGE_SIZE = 25;

const tabs: { value: TicketStatus | ""; label: string }[] = [
    { value: "", label: "All" },
    { value: "needs_review", label: "Needs review" },
    { value: "auto_labeled", label: "Auto-labeled" },
    { value: "confirmed", label: "Confirmed" },
];

export default function Tickets() {
    const [status, setStatus] = useState<TicketStatus | "">("");
    const [page, setPage] = useState(0);

    const { data, error, isLoading } = useTickets({
        status: status || undefined,
        sort: "newest",
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
    });

    const total = data?.total ?? 0;
    const lastPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);

    return (
        <>
            <PageHeader title="Tickets" subtitle={`${total} in this view`} />

            <div className="mb-4 flex gap-1">
                {tabs.map((tab) => (
                    <Button
                        key={tab.value}
                        variant={status === tab.value ? "primary" : "secondary"}
                        onClick={() => {
                            setStatus(tab.value);
                            setPage(0);
                        }}
                    >
                        {tab.label}
                    </Button>
                ))}
            </div>

            {isLoading && <Spinner />}
            <ErrorBox error={error} />

            {data && data.items.length === 0 && <Empty>No tickets here yet.</Empty>}

            {data && data.items.length > 0 && (
                <Card className="overflow-x-auto p-0">
                    <table className="w-full text-left text-sm">
                        <thead className="border-b border-slate-200 text-xs uppercase text-slate-500">
                            <tr>
                                <th className="px-3 py-2">#</th>
                                <th className="px-3 py-2">Ticket</th>
                                <th className="px-3 py-2">Status</th>
                                <th className="px-3 py-2">Model Prediction</th>
                                <th className="px-3 py-2">Final category</th>
                                <th className="px-3 py-2">Created</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.items.map((ticket) => (
                                <tr key={ticket.id} className="border-b border-slate-100 align-top">
                                    <td className="px-3 py-2 text-slate-500">{ticket.id}</td>
                                    <td className="max-w-md px-3 py-2">
                                        <div className="line-clamp-2">{ticket.text}</div>
                                    </td>
                                    <td className="px-3 py-2">
                                        <StatusBadge status={ticket.status} />
                                    </td>
                                    <td className="px-3 py-2">
                                        {ticket.predicted_label ? (
                                            <>
                                                <div>{ticket.predicted_label}</div>
                                                <div className="text-xs text-slate-500">
                                                    {pct(ticket.confidence, 0)} · {ticket.model_version}
                                                </div>
                                            </>
                                        ) : (
                                            "–"
                                        )}
                                    </td>
                                    <td className="px-3 py-2">{ticket.category?.name ?? "–"}</td>
                                    <td className="whitespace-nowrap px-3 py-2 text-slate-500">
                                        {time(ticket.created_at)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </Card>
            )}

            <div className="mt-4 flex items-center justify-end gap-2 text-sm">
                <span className="text-slate-500">
                    Page {page + 1} of {lastPage + 1}
                </span>
                <Button disabled={page === 0} onClick={() => setPage(page - 1)}>
                    Previous
                </Button>
                <Button disabled={page >= lastPage} onClick={() => setPage(page + 1)}>
                    Next
                </Button>
            </div>
        </>
    );
}