import { useState } from "react";

export default function BatchFormatInfo() {
    const [open, setOpen] = useState(false);

    return (
        <div style={{ marginBottom: 20 }}>
            <div
                onClick={() => setOpen(!open)}
                style={{
                    cursor: "pointer",
                    fontWeight: 600,
                    fontSize: 14,
                    color: "#2563eb",
                    marginBottom: 8,
                }}
            >
                {open ? "▼" : "▶"} Input File Requirements
            </div>

            {open && (
                <div
                    style={{
                        background: "#f9fafb",
                        border: "1px solid #e5e7eb",
                        borderRadius: 8,
                        padding: 16,
                        fontSize: 13,
                        lineHeight: 1.6,
                    }}
                >
                    <p>
                        Upload a <b>.csv</b> or <b>.xlsx</b> file containing the following
                        required columns (case-sensitive):
                    </p>

                    <ul style={{ marginTop: 8 }}>
                        <li><b>latitude</b> — decimal degrees (e.g. 52.5352)</li>
                        <li><b>longitude</b> — decimal degrees (e.g. 13.2433)</li>
                        <li><b>buffer</b> — radius in kilometers (e.g. 1.4)</li>
                    </ul>

                    <div style={{ marginTop: 12 }}>
                        <b>Example CSV:</b>
                        <pre
                            style={{
                                background: "#111827",
                                color: "#f9fafb",
                                padding: 12,
                                borderRadius: 6,
                                overflowX: "auto",
                                marginTop: 6,
                            }}
                        >
{`latitude,longitude,buffer
52.535223392234045,13.243375821692387,1.4
53.165035,9.49738,1.0`}
            </pre>
                    </div>

                    <div style={{ marginTop: 12 }}>
                        <b>Important:</b>
                        <ul>
                            <li>No empty rows</li>
                            <li>Decimal separator must be a dot (.)</li>
                            <li>First row must contain headers</li>
                            <li>Recommended max size: 500–1000 rows</li>
                        </ul>
                    </div>
                </div>
            )}
        </div>
    );
}