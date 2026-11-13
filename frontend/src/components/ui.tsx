import type { ButtonHTMLAttributes, ReactNode } from "react";

import { ApiError } from "../api/client";

// ---- buttons ---------------------------------------------------------------
type Variant = "primary" | "secondary" | "danger" | "ghost";

const variants: Record<Variant, string> = {
    primary: "bg-indigo-600 text-white hover:bg-indigo-700",
    secondary: "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50",
    danger: "bg-red-600 text-white hover:bg-red-700",
    ghost: "text-slate-600 hover:bg-slate-100",
};

export function Button({
    variant = "secondary",
    className = "",
    ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
    return (
        <button
            {...props}
            className={`inline-flex items-center justify-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${variants[variant]} ${className}`}
        />
    );
}

// ---- layout pieces ---------------------------------------------------------
export function PageHeader({
    title,
    subtitle,
    children,
}: {
    title: string;
    subtitle?: string;
    children?: ReactNode;
}) {
    return (
        <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
            <div>
                <h1 className="text-2xl font-semibold text-gray-200">{title}</h1>
                {subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}
            </div>
            {children}
        </div>
    );
}

export function Card({
    title,
    actions,
    children,
    className = "",
}: {
    title?: string;
    actions?: ReactNode;
    children: ReactNode;
    className?: string;
}) {
    return (
        <section
            className={`rounded-lg border border-slate-200 bg-gray-100 p-4 shadow-sm ${className}`}
        >
            {(title || actions) && (
                <div className="mb-3 flex items-center justify-between gap-2">
                    {title && <h2 className="font-semibold">{title}</h2>}
                    {actions}
                </div>
            )}
            {children}
        </section>
    );
}

// ---- small pieces ----------------------------------------------------------
type Tone = "gray" | "green" | "red" | "amber" | "blue" | "indigo";

const tones: Record<Tone, string> = {
    gray: "bg-slate-100 text-slate-700",
    green: "bg-green-100 text-green-800",
    red: "bg-red-100 text-red-800",
    amber: "bg-amber-100 text-amber-800",
    blue: "bg-blue-100 text-blue-800",
    indigo: "bg-indigo-100 text-indigo-800",
};

export function Badge({
    tone = "gray",
    children,
}: {
    tone?: Tone;
    children: ReactNode;
}) {
    return (
        <span
            className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${tones[tone]}`}
        >
            {children}
        </span>
    );
}

const statusTones: Record<string, [Tone, string]> = {
    needs_review: ["amber", "Needs review"],
    auto_labeled: ["green", "Auto-labeled"],
    confirmed: ["blue", "Confirmed"],
    proposed: ["amber", "Proposed"],
    approved: ["green", "Approved"],
    rejected: ["red", "Rejected"],
    merged: ["gray", "Merged"],
    queued: ["gray", "Queued"],
    running: ["blue", "Running"],
    succeeded: ["green", "Succeeded"],
    failed: ["red", "Failed"],
};

export function StatusBadge({ status }: { status: string }) {
    const [tone, label] = statusTones[status] ?? ["gray", status];
    return <Badge tone={tone}>{label}</Badge>;
}

export function ProgressBar({ value, max }: { value: number; max: number }) {
    const share = max > 0 ? Math.min(100, (value / max) * 100) : 0;
    return (
        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
            <div
                className="h-full rounded-full bg-indigo-500"
                style={{ width: `${share}%` }}
            />
        </div>
    );
}

export function ConfidenceBar({ value }: { value: number }) {
    const color =
        value >= 0.8 ? "bg-green-500" : value >= 0.5 ? "bg-amber-500" : "bg-red-500";
    return (
        <div className="flex items-center gap-2">
            <div className="h-2 w-24 overflow-hidden rounded-full bg-slate-100">
                <div
                    className={`h-full rounded-full ${color}`}
                    style={{ width: `${Math.round(value * 100)}%` }}
                />
            </div>
            <span className="w-10 text-right text-xs tabular-nums text-slate-600">
                {Math.round(value * 100)}%
            </span>
        </div>
    );
}

export function Spinner() {
    return <p className="py-8 text-center text-sm text-slate-500">Loading…</p>;
}

export function Empty({ children }: { children: ReactNode }) {
    return (
        <div className="rounded-lg border border-dashed border-slate-300 py-12 text-center text-sm text-slate-500">
            {children}
        </div>
    );
}

export function ErrorBox({ error }: { error: unknown }) {
    if (!error) return null;
    const message =
        error instanceof ApiError || error instanceof Error
            ? error.message
            : "Something went wrong.";
    return (
        <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
            {message}
        </div>
    );
}