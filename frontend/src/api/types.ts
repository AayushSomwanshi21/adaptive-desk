// These mirror the response models in backend/app/schemas.py.

export interface Prediction {
    label: string;
    confidence: number;
}

export interface CategoryRef {
    id: number;
    name: string;
    status: string;
}

export type TicketStatus = "needs_review" | "auto_labeled" | "confirmed";

export interface Ticket {
    id: number;
    text: string;
    status: TicketStatus;
    predicted_label: string | null;
    confidence: number | null;
    top_predictions: Prediction[];
    is_spot_check: boolean;
    category: CategoryRef | null;
    model_version: string | null;
    created_at: string;
    reviewed_at: string | null;
}

export interface TicketList {
    items: Ticket[];
    total: number;
}

export interface PredictResult {
    model_version: string | null;
    label: string | null;
    confidence: number | null;
    top: Prediction[];
    needs_review: boolean;
}

export type CategoryStatus = "proposed" | "approved" | "rejected" | "merged";

export interface Category {
    id: number;
    name: string;
    description: string;
    status: CategoryStatus;
    merged_into_id: number | null;
    created_at: string;
    in_model: boolean;
    tickets_available: number;
    tickets_needed: number | null;
}

export type LabelBody =
    | { category_id: number }
    | { new_category: { name: string; description: string } };

export interface PlanCategory {
    name: string;
    tickets: number;
    needed: number;
}

export interface TrainingPlan {
    can_start: boolean;
    reason: string | null;
    ready_categories: PlanCategory[];
    collecting: PlanCategory[];
    // Only present if you added the optional edit to schemas.py.
    refresh_categories?: PlanCategory[];
    min_tickets_new_category: number;
    n_tickets: number;
}

export interface GateCheck {
    name: string;
    passed: boolean;
    value: number | null;
    limit: number;
    detail: string;
}

export interface Gate {
    passed: boolean;
    checks: GateCheck[];
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface Job {
    id: number;
    status: JobStatus;
    created_at: string;
    started_at: string | null;
    finished_at: string | null;
    base_version: string | null;
    version: string | null;
    error: string | null;
    n_tickets: number;
    summary: {
        new_categories: string[] | null;
        gate: Gate | null;
        accuracy: number | null;
        old_accuracy: number | null;
        old_accuracy_drop: number | null;
        new_category_accuracy: number | null;
    } | null;
}

export interface ModelInfo {
    name: string;
    parent: string | null;
    created_at: string | null;
    notes: string;
    active: boolean;
    complete: boolean;
    n_categories: number | null;
    new_categories: string[];
    accuracy: number | null;
    old_accuracy: number | null;
    old_accuracy_drop: number | null;
    new_category_accuracy: number | null;
    gate: Gate | null;
    calibration: {
        temperature?: number;
        threshold?: number | null;
        auto_rate?: number;
        review_rate?: number;
        target_precision?: number;
    };
}

export interface Stats {
    model: {
        version: string | null;
        n_categories?: number;
        temperature?: number;
        threshold?: number | null;
    };
    tickets: {
        total: number;
        by_status: Record<string, number>;
        review_queue: number;
        auto_label_rate: number | null;
        average_confidence: number | null;
    };
    spot_checks: { reviewed: number; agreement: number | null };
    categories: Record<string, number>;
}