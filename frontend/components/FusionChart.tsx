import { useMemo } from "react";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    CartesianGrid,
    ReferenceLine,
    Legend,
    LabelList,
} from "recharts";

interface FusionChartProps {
    fusion: {
        classes?: string[];
        scores?: Record<string, number>;
        modal_scores?: {
            image?: Record<string, number>;
            osm?: Record<string, number>;
            database?: Record<string, number>;
            agent?: Record<string, number>;
        };
        explanation?: string[];
        threshold?: number;
    };
}

type FusionRow = {
    name: string;
    image: number;
    osm: number;
    database: number;
    agent: number;
    total: number;
    selected: boolean;
    passedThreshold: boolean;
    status: "selected" | "excluded" | "below-threshold";
};

const DEFAULT_THRESHOLD = 0.4;

function formatClassName(name: string) {
    return name.replace(/_/g, " ");
}

function buildRows(fusion: FusionChartProps["fusion"], threshold: number): FusionRow[] {
    const scores = fusion?.scores || {};
    const modalScores = fusion?.modal_scores || {};
    const image = modalScores.image || {};
    const osm = modalScores.osm || {};
    const database = modalScores.database || {};
    const agent = modalScores.agent || {};
    const selectedSet = new Set(fusion?.classes || []);

    const labels = Array.from(
        new Set([
            ...Object.keys(scores),
            ...Object.keys(image),
            ...Object.keys(osm),
            ...Object.keys(database),
            ...Object.keys(agent),
        ])
    );

    return labels
        .map((label) => {
            const total = Number(scores[label] || 0);
            const selected = selectedSet.has(label);
            const passedThreshold = total >= threshold;

            let status: FusionRow["status"] = "below-threshold";
            if (selected) status = "selected";
            else if (passedThreshold) status = "excluded";

            return {
                name: label,
                image: Number(image[label] || 0),
                osm: Number(osm[label] || 0),
                database: Number(database[label] || 0),
                agent: Number(agent[label] || 0),
                total,
                selected,
                passedThreshold,
                status,
            };
        })
        .filter((row) => row.total > 0)
        .sort((a, b) => b.total - a.total);
}

function CustomTooltip({ active, payload, label }: any) {
    if (!active || !payload || !payload.length) return null;

    const row = payload[0]?.payload;
    if (!row) return null;

    return (
        <div className="rounded border border-slate-200 bg-white p-3 shadow text-sm">
            <div className="font-semibold mb-2">{formatClassName(label)}</div>
            <div className="space-y-1">
                <div>Total: <span className="font-mono">{row.total.toFixed(3)}</span></div>
                <div>Image: <span className="font-mono">{row.image.toFixed(3)}</span></div>
                <div>OSM: <span className="font-mono">{row.osm.toFixed(3)}</span></div>
                <div>Database: <span className="font-mono">{row.database.toFixed(3)}</span></div>
                <div>Agent: <span className="font-mono">{row.agent.toFixed(3)}</span></div>
                <div>Status: <span className="font-medium">{row.status}</span></div>
            </div>
        </div>
    );
}

export default function FusionChart({ fusion }: FusionChartProps) {
    const threshold = fusion?.threshold ?? DEFAULT_THRESHOLD;

    const data = useMemo(() => {
        return buildRows(fusion, threshold);
    }, [fusion, threshold]);

    if (!fusion || !fusion.scores || Object.keys(fusion.scores).length === 0) {
        return <div className="text-sm text-slate-500">No fusion data available</div>;
    }

    const maxScore = Math.max(...data.map((d) => d.total), threshold, 0.5);
    const xMax = Math.max(1, Math.ceil((maxScore + 0.1) * 10) / 10);

    return (
        <div className="space-y-4">
            {/* Final selected classes */}
            <div>
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                    Final prediction set
                </div>
                <div className="flex flex-wrap gap-2">
                    {(fusion.classes || []).length > 0 ? (
                        fusion.classes!.map((cls) => (
                            <span
                                key={cls}
                                className="rounded-full bg-blue-100 text-blue-800 px-3 py-1 text-xs font-medium"
                            >
                {formatClassName(cls)}
              </span>
                        ))
                    ) : (
                        <span className="text-sm text-slate-500">No class retained</span>
                    )}
                </div>
            </div>

            {/* Chart */}
            <div style={{ width: "100%", height: 420 }}>
                <ResponsiveContainer>
                    <BarChart
                        data={data}
                        layout="vertical"
                        margin={{ top: 10, right: 50, left: 80, bottom: 10 }}
                        barCategoryGap={12}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" domain={[0, xMax]} />
                        <YAxis
                            type="category"
                            dataKey="name"
                            width={190}
                            tickFormatter={formatClassName}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />

                        <ReferenceLine
                            x={threshold}
                            stroke="red"
                            strokeDasharray="4 4"
                            label={{ value: `Threshold (${threshold})`, position: "top" }}
                        />

                        <Bar dataKey="image" stackId="a" fill="#3b82f6" name="Image">
                            <LabelList
                                dataKey="total"
                                position="right"
                                formatter={(v: number) => v.toFixed(2)}
                            />
                        </Bar>
                        <Bar dataKey="osm" stackId="a" fill="#10b981" name="OSM" />
                        <Bar dataKey="database" stackId="a" fill="#f59e0b" name="Database" />
                        <Bar dataKey="agent" stackId="a" fill="#8b5cf6" name="Agent" />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Status list */}
            <div className="space-y-2">
                {data.map((row) => (
                    <div
                        key={row.name}
                        className="flex items-center justify-between rounded border border-slate-200 bg-white px-3 py-2 text-sm"
                    >
                        <div className="font-medium">{formatClassName(row.name)}</div>
                        <div className="flex items-center gap-3">
                            <span className="font-mono text-slate-700">{row.total.toFixed(3)}</span>
                            {row.status === "selected" && (
                                <span className="rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
                  Selected
                </span>
                            )}
                            {row.status === "excluded" && (
                                <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
                  Above threshold, excluded
                </span>
                            )}
                            {row.status === "below-threshold" && (
                                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">
                  Below threshold
                </span>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            {/* Decision trace */}
            {Array.isArray(fusion.explanation) && fusion.explanation.length > 0 && (
                <div className="rounded border border-slate-200 bg-white p-4">
                    <div className="text-sm font-semibold mb-2">Decision trace</div>
                    <ul className="list-disc pl-5 space-y-1 text-sm text-slate-700">
                        {fusion.explanation.map((line, idx) => (
                            <li key={idx}>{line}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}