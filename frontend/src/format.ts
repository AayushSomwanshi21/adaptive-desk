export function pct(value: number | null | undefined, digits = 1): string {
    return value === null || value === undefined
        ? "–"
        : `${(value * 100).toFixed(digits)}%`;
}

export function time(iso: string | null | undefined): string {
    return iso ? new Date(iso).toLocaleString() : "–";
}

/** Lower-case and drop separators so "Card_Arrival" matches "card arrival". */
export function normalize(text: string): string {
    return text.toLowerCase().replace(/[\s_-]+/g, "");
}