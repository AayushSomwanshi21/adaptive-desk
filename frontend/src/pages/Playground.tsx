import { useState } from "react";

import { useCreateTicket, usePredict } from "../api/hooks";
import {
    Badge,
    Button,
    Card,
    ConfidenceBar,
    ErrorBox,
    PageHeader,
    StatusBadge,
} from "../components/ui";

export default function Playground() {
    const [text, setText] = useState("");
    const predict = usePredict();
    const createTicket = useCreateTicket();

    const prediction = predict.data;
    const ticket = createTicket.data;

    return (
        <>
            <PageHeader
                title="Try it"
                subtitle="Classify some text, or submit it as a real ticket to see where it goes."
            />

            <Card className="mb-4">
                <textarea
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    rows={4}
                    placeholder="e.g. My transfer still hasn't arrived and the recipient is waiting."
                    className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                />
                <div className="mt-3 flex gap-2">
                    <Button
                        variant="primary"
                        disabled={!text.trim() || predict.isPending}
                        onClick={() => predict.mutate(text)}
                    >
                        Classify only
                    </Button>
                    <Button
                        disabled={!text.trim() || createTicket.isPending}
                        onClick={() => { createTicket.mutate(text), setText("") }}
                    >
                        Submit as ticket
                    </Button>
                </div>
            </Card>

            <div className="space-y-4">
                <ErrorBox error={predict.error ?? createTicket.error} />

                {prediction && (
                    <Card
                        title={`Prediction (model ${prediction.model_version ?? "none"})`}
                        actions={
                            prediction.needs_review ? (
                                <Badge tone="amber">Would go to review</Badge>
                            ) : (
                                <Badge tone="green">Would be auto-labeled</Badge>
                            )
                        }
                    >
                        {prediction.top.length === 0 ? (
                            <p className="text-sm text-slate-500">No model is active yet.</p>
                        ) : (
                            <ul className="space-y-2">
                                {prediction.top.map((guess) => (
                                    <li key={guess.label} className="flex items-center justify-between">
                                        <span className="text-sm font-medium">{guess.label}</span>
                                        <ConfidenceBar value={guess.confidence} />
                                    </li>
                                ))}
                            </ul>
                        )}
                    </Card>
                )}

                {ticket && (
                    <Card title={`Ticket #${ticket.id} created`}>
                        <div className="flex items-center gap-2 text-sm">
                            <StatusBadge status={ticket.status} />
                            {ticket.status === "auto_labeled" && ticket.category && (
                                <span>
                                    labeled as <b>{ticket.category.name}</b>
                                </span>
                            )}
                            {ticket.status === "needs_review" && (
                                <span>
                                    {ticket.is_spot_check
                                        ? "kept for a spot check"
                                        : "waiting in the review queue"}
                                </span>
                            )}
                        </div>
                    </Card>
                )}
            </div>
        </>
    );
}