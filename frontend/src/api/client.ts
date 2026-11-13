const BASE =
    (import.meta.env.VITE_API_URL as string | undefined) ??
    "http://localhost:8000";

export class ApiError extends Error {
    status: number;

    constructor(status: number, message: string) {
        super(message);
        this.status = status;
    }
}

/** FastAPI errors come as a string, a list of field errors, or an object. */
function describe(detail: unknown): string {
    if (typeof detail === "string") return detail;

    if (Array.isArray(detail)) {
        return detail
            .map((item) =>
                item && typeof item === "object" && "msg" in item
                    ? String((item as { msg: unknown }).msg)
                    : JSON.stringify(item),
            )
            .join("; ");
    }

    if (detail && typeof detail === "object") {
        const d = detail as {
            message?: string;
            failed_checks?: string[];
            hint?: string;
        };
        return [d.message, ...(d.failed_checks ?? []), d.hint]
            .filter(Boolean)
            .join(" ");
    }

    return "The request failed.";
}

type Options = Omit<RequestInit, "body"> & { json?: unknown };

export async function api<T>(path: string, options: Options = {}): Promise<T> {
    const { json, ...init } = options;
    const headers = new Headers(init.headers);
    let body: string | undefined;

    if (json !== undefined) {
        headers.set("Content-Type", "application/json");
        body = JSON.stringify(json);
    }

    let response: Response;
    try {
        response = await fetch(`${BASE}${path}`, { ...init, headers, body });
    } catch {
        throw new ApiError(
            0,
            `Cannot reach the API at ${BASE}. Is the backend running?`,
        );
    }

    if (!response.ok) {
        let detail: unknown = response.statusText;
        try {
            detail = (await response.json()).detail;
        } catch {
            // keep the status text
        }
        throw new ApiError(response.status, describe(detail));
    }

    return (await response.json()) as T;
}