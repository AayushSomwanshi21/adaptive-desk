import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { useStats } from "./api/hooks";
import { Badge } from "./components/ui";
import Categories from "./pages/Categories";
import Dashboard from "./pages/Dashboard";
import Models from "./pages/Models";
import Playground from "./pages/Playground";
import ReviewQueue from "./pages/ReviewQueue";
import Tickets from "./pages/Tickets";

function NavItem({
    to,
    label,
    count,
}: {
    to: string;
    label: string;
    count?: number;
}) {
    return (
        <NavLink
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
                `flex items-center justify-between rounded-md px-3 py-2 text-sm font-medium ${isActive
                    ? "bg-indigo-50 text-indigo-800"
                    : "text-slate-300 hover:bg-slate-100 hover:text-slate-800 transition-colors duration-100"
                }`
            }
        >
            <span>{label}</span>
            {count !== undefined && count > 0 && <Badge tone="amber">{count}</Badge>}
        </NavLink>
    );
}

export default function App() {
    const { data: stats, error } = useStats();

    return (
        <div className="flex min-h-screen">
            <aside className="flex w-56 shrink-0 flex-col gap-1 border-r border-slate-200 bg-gray-950 p-3">
                <div className="mb-4 px-3 pt-2">
                    <div className="text-2xl font-bold text-indigo-400">AdaptiveDesk</div>
                    <div className="text-sm text-slate-500">
                        {error
                            ? "API unreachable"
                            : stats?.model.version
                                ? `Model ${stats.model.version}`
                                : "No model yet"}
                    </div>
                </div>

                <NavItem to="/" label="Dashboard" />
                <NavItem
                    to="/review"
                    label="Review queue"
                    count={stats?.tickets.review_queue}
                />
                <NavItem to="/tickets" label="Tickets" />
                <NavItem
                    to="/categories"
                    label="Categories"
                    count={stats?.categories.proposed}
                />
                <NavItem to="/model" label="Model & training" />
                <NavItem to="/playground" label="Playground" />
            </aside>

            <main className="min-w-0 flex-1 p-6">
                <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/review" element={<ReviewQueue />} />
                    <Route path="/tickets" element={<Tickets />} />
                    <Route path="/categories" element={<Categories />} />
                    <Route path="/model" element={<Models />} />
                    <Route path="/playground" element={<Playground />} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
            </main>
        </div>
    );
}