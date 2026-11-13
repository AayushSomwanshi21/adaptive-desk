import {
    keepPreviousData,
    useMutation,
    useQuery,
    useQueryClient,
} from "@tanstack/react-query";

import { api } from "./client";
import type {
    Category,
    Job,
    LabelBody,
    ModelInfo,
    PredictResult,
    Stats,
    Ticket,
    TicketList,
    TicketStatus,
    TrainingPlan,
} from "./types";

// reading data
export function useStats() {
    return useQuery({
        queryKey: ["stats"],
        queryFn: () => api<Stats>("/stats"),
        refetchInterval: 10_000,
    });
}

export interface TicketParams {
    status?: TicketStatus;
    sort?: "newest" | "lowest_confidence";
    limit?: number;
    offset?: number;
}

export function useTickets(params: TicketParams) {
    const query = new URLSearchParams();
    if (params.status) query.set("status", params.status);
    if (params.sort) query.set("sort", params.sort);
    query.set("limit", String(params.limit ?? 50));
    query.set("offset", String(params.offset ?? 0));

    return useQuery({
        queryKey: ["tickets", params],
        queryFn: () => api<TicketList>(`/tickets?${query}`),
        placeholderData: keepPreviousData,
    });
}

export function useCategories() {
    return useQuery({
        queryKey: ["categories"],
        queryFn: () => api<Category[]>("/categories"),
    });
}

export function usePlan() {
    return useQuery({
        queryKey: ["plan"],
        queryFn: () => api<TrainingPlan>("/training/plan"),
    });
}

export function useJobs() {
    return useQuery({
        queryKey: ["jobs"],
        queryFn: () => api<Job[]>("/training/jobs"),
        // Poll while something is running.
        refetchInterval: (query) =>
            query.state.data?.some(
                (job) => job.status === "queued" || job.status === "running",
            )
                ? 3_000
                : false,
    });
}

export function useModels() {
    return useQuery({
        queryKey: ["models"],
        queryFn: () => api<ModelInfo[]>("/models"),
    });
}

// changing data
/** Any change can affect several screens, so refresh everything afterwards. */
function useAction<TVars, TData>(run: (vars: TVars) => Promise<TData>) {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: run,
        onSuccess: () => queryClient.invalidateQueries(),
    });
}

export const useCreateTicket = () =>
    useAction((text: string) =>
        api<Ticket>("/tickets", { method: "POST", json: { text } }),
    );

export const usePredict = () =>
    useMutation({
        mutationFn: (text: string) =>
            api<PredictResult>("/predict", { method: "POST", json: { text } }),
    });

export const useLabelTicket = () =>
    useAction((vars: { ticketId: number; body: LabelBody }) =>
        api<Ticket>(`/tickets/${vars.ticketId}/label`, {
            method: "POST",
            json: vars.body,
        }),
    );

export const useCreateCategory = () =>
    useAction((body: { name: string; description: string }) =>
        api<Category>("/categories", { method: "POST", json: body }),
    );

export const useCategoryAction = () =>
    useAction((vars: { id: number; action: "approve" | "reject" }) =>
        api<Category>(`/categories/${vars.id}/${vars.action}`, { method: "POST" }),
    );

export const useMergeCategory = () =>
    useAction((vars: { id: number; intoId: number }) =>
        api<Category>(`/categories/${vars.id}/merge`, {
            method: "POST",
            json: { into_id: vars.intoId },
        }),
    );

export const useStartJob = () =>
    useAction<void, Job>(() => api<Job>("/training/jobs", { method: "POST" }));

export const usePromote = () =>
    useAction((vars: { name: string; force: boolean }) =>
        api<ModelInfo>(`/models/${vars.name}/promote`, {
            method: "POST",
            json: { force: vars.force },
        }),
    );

export const useRollback = () =>
    useAction<void, ModelInfo>(() =>
        api<ModelInfo>("/models/rollback", { method: "POST" }),
    );