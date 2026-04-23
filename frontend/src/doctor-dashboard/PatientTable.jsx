import React, { useMemo } from 'react';

const getSurvivalDetails = (status, survivalProb) => {
    if (!status || status === 'NONE') {
        return { chance: "—", label: "No triage submitted" };
    }
    const conf = (survivalProb || 0) / 100;
    let chance = 0;
    let label = "";

    switch (status?.toUpperCase()) {
        case 'GREEN':
            chance = 90 + (conf * 10);
            label = "Stable";
            break;
        case 'YELLOW':
            chance = 50 + (conf * 30);
            label = "Moderate Risk";
            break;
        case 'RED':
            chance = 50 - (conf * 40);
            label = "Critical";
            break;
        case 'BLACK':
            chance = 10 - (conf * 9);
            label = "No Survival Likely";
            break;
        default:
            return { chance: "N/A", label: "Unknown" };
    }
    return { chance: `${chance.toFixed(1)}%`, label };
};

const PatientTable = ({ patients, teams = [], onSelectPatient }) => {
    const rankMap = useMemo(() => {
        const sorted = [...patients].sort((a, b) => (a.priority ?? 999) - (b.priority ?? 999));
        const map = new Map();
        let currentRank = 1;
        for (let i = 0; i < sorted.length; i++) {
            const p = sorted[i];
            if (i > 0) {
                const prev = sorted[i - 1];
                if (Math.abs((p.priority ?? 999) - (prev.priority ?? 999)) > 0.02) {
                    currentRank++;
                }
            }
            map.set(p.patientId, currentRank);
        }
        return map;
    }, [patients]);

    return (
        <div className="table-container">
            <div className="table-responsive">
                <table>
                <thead>
                    <tr>
                        <th>Patient ID</th>
                        <th>Status</th>
                        <th>Survival Chance (%)</th>
                        <th>Assigned Team</th>
                        <th>Priority Rank</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {patients.map((patient) => {
                        const { chance, label } = getSurvivalDetails(patient.status, patient.survivalProbability);
                        const statusText = patient.status === 'NONE' ? 'NO TRIAGE' : patient.status;
                        const rank = rankMap.get(patient.patientId);
                        const assignedTeam = teams.find(t => t.id === patient.assigned_team_id);
                        const teamName = assignedTeam ? assignedTeam.name : "—";

                        return (
                            <tr key={patient.patientId} onClick={() => onSelectPatient(patient)}>
                                <td>
                                    <div style={{ lineHeight: 1.4 }}>
                                        <div>{patient.patientId}</div>
                                        {patient.patientName && (
                                            <div style={{ fontSize: '0.8rem', color: '#a78bfa', marginTop: '2px' }}>👤 {patient.patientName}</div>
                                        )}
                                    </div>
                                </td>
                                <td>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                                        <span className={`status-badge status-${patient.status}`}>
                                            {statusText}
                                        </span>
                                        <span style={{ fontSize: '0.65rem', color: '#94a3b8', marginLeft: '4px' }}>{label}</span>
                                    </div>
                                </td>
                                <td style={{ fontWeight: 'bold', color: patient.status === 'GREEN' ? '#10b981' : '#fff' }}>
                                    {chance}
                                </td>
                                <td>
                                    {teamName !== "—" ? (
                                        <span style={{ background: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa', padding: '2px 8px', borderRadius: '12px', fontSize: '0.8rem', fontWeight: 'bold' }}>
                                            🚁 {teamName}
                                        </span>
                                    ) : "—"}
                                </td>
                                <td className="priority-cell">
                                    {patient.priority != null ? (
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                            <span style={{
                                                background: 'rgba(56, 189, 248, 0.1)',
                                                color: '#38bdf8',
                                                padding: '2px 8px',
                                                borderRadius: '12px',
                                                fontWeight: 'bold'
                                            }}>
                                                #{rank}
                                            </span>
                                            <span style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>
                                                ({patient.priority.toFixed(4)})
                                            </span>
                                        </div>
                                    ) : "—"}
                                </td>
                                <td>{patient.timestamp ? new Date(patient.timestamp).toLocaleString() : "—"}</td>
                            </tr>
                        );
                    })}
                    {patients.length === 0 && (
                        <tr>
                            <td colSpan="6" style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                                No patients found.
                            </td>
                        </tr>
                    )}
                </tbody>
            </table>
            </div>
        </div>
    );
};

export default PatientTable;

