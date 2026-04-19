import React from 'react';

const getSurvivalDetails = (status, survivalProb) => {
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

const PatientTable = ({ patients, onSelectPatient }) => {
    return (
        <div className="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Patient ID</th>
                        <th>Status</th>
                        <th>Survival Chance (%)</th>
                        <th>Priority</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {patients.map((patient) => {
                        const { chance, label } = getSurvivalDetails(patient.status, patient.survivalProbability);
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
                                            {patient.status}
                                        </span>
                                        <span style={{ fontSize: '0.65rem', color: '#94a3b8', marginLeft: '4px' }}>{label}</span>
                                    </div>
                                </td>
                                <td style={{ fontWeight: 'bold', color: patient.status === 'GREEN' ? '#10b981' : '#fff' }}>
                                    {chance}
                                </td>
                                <td className="priority-cell">{patient.priority}</td>
                                <td>{new Date(patient.timestamp).toLocaleString()}</td>
                            </tr>
                        );
                    })}
                    {patients.length === 0 && (
                        <tr>
                            <td colSpan="5" style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                                No patients found.
                            </td>
                        </tr>
                    )}
                </tbody>
            </table>
        </div>
    );
};

export default PatientTable;
