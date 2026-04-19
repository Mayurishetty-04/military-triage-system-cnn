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

const PatientDetails = ({ patient, onClose, onViewHistory }) => {
    if (!patient) return null;
    const { chance, label: survivalLabel } = getSurvivalDetails(patient.status, patient.survivalProbability);

    const theme = {
        RED: { color: '#ef4444', gradient: 'linear-gradient(135deg, #7f1d1d 0%, #ef4444 100%)', bg: 'rgba(239, 68, 68, 0.1)' },
        YELLOW: { color: '#eab308', gradient: 'linear-gradient(135deg, #713f12 0%, #eab308 100%)', bg: 'rgba(234, 179, 8, 0.1)' },
        GREEN: { color: '#22c55e', gradient: 'linear-gradient(135deg, #14532d 0%, #22c55e 100%)', bg: 'rgba(34, 197, 94, 0.1)' },
        BLACK: { color: '#475569', gradient: 'linear-gradient(135deg, #1e293b 0%, #475569 100%)', bg: 'rgba(71, 85, 105, 0.1)' }
    }[patient.status?.toUpperCase()] || { color: '#3b82f6', gradient: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)', bg: 'rgba(59, 130, 246, 0.1)' };

    return (
        <div className="details-overlay" onClick={onClose} style={{ backdropFilter: 'blur(8px)', background: 'rgba(0,0,0,0.7)' }}>
            <div 
                className="details-card" 
                onClick={(e) => e.stopPropagation()}
                style={{ 
                    background: '#0f172a', 
                    border: '1px solid rgba(255,255,255,0.1)',
                    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8)',
                    padding: 0,
                    maxHeight: '90vh',
                    overflowY: 'auto',
                    maxWidth: '650px',
                    position: 'relative'
                }}
            >
                {/* 🌟 Dynamic Status Header */}
                <div style={{ background: theme.gradient, padding: '2rem', position: 'relative' }}>
                    <button 
                        className="close-btn" 
                        onClick={onClose}
                        style={{ position: 'absolute', top: '1rem', right: '1rem', color: '#fff', opacity: 0.7 }}
                    >
                        &times;
                    </button>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                        <div>
                            <div style={{ textTransform: 'uppercase', fontSize: '0.75rem', fontWeight: 900, letterSpacing: '0.2em', color: 'rgba(255,255,255,0.8)', marginBottom: '0.4rem' }}>
                                Triage Priority
                            </div>
                            <h2 style={{ fontSize: '2.5rem', fontWeight: 900, margin: 0, color: '#fff', lineHeight: 1 }}>
                                {patient.status}
                            </h2>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                            <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#fff' }}>{survivalLabel}</div>
                            <div style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.7)' }}>Tactical Condition</div>
                        </div>
                    </div>
                </div>

                <div style={{ padding: '2rem' }}>
                    {/* Metadata Bar */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2rem', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '1rem' }}>
                        <div>
                            <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Patient Identity</div>
                            <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#f8fafc' }}>{patient.patientId}</div>
                            {patient.patientName && (
                                <div style={{ color: '#a78bfa', fontSize: '0.85rem', fontWeight: 600 }}>👤 {patient.patientName}</div>
                            )}
                        </div>
                        <div style={{ textAlign: 'right' }}>
                            <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase' }}>Submission Time</div>
                            <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#94a3b8' }}>{new Date(patient.timestamp).toLocaleString()}</div>
                        </div>
                    </div>

                    {/* Vitals Intelligence Grid */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '2.5rem' }}>
                        {[
                            { label: 'SPO2', value: `${patient.spo2}%`, icon: '🫁', color: patient.spo2 < 90 ? '#ef4444' : '#10b981' },
                            { label: 'HR', value: `${patient.heartRate} bpm`, icon: '💓', color: (patient.heartRate > 120 || patient.heartRate < 50) ? '#ef4444' : '#38bdf8' },
                            { label: 'SURVIVAL', value: chance, icon: '🛡️', color: theme.color }
                        ].map((v, i) => (
                            <div key={i} style={{ background: 'rgba(30, 41, 59, 0.5)', padding: '1.25rem', borderRadius: '1rem', border: '1px solid rgba(255,255,255,0.03)', textAlign: 'center' }}>
                                <div style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>{v.icon}</div>
                                <div style={{ fontSize: '0.65rem', color: '#64748b', fontWeight: 800, textTransform: 'uppercase', marginBottom: '0.25rem' }}>{v.label}</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 900, color: v.color }}>{v.value}</div>
                            </div>
                        ))}
                    </div>

                    {/* AI Confidence Analysis */}
                    <div style={{ marginBottom: '2.5rem' }}>
                        <h3 style={{ fontSize: '0.8rem', fontWeight: 900, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span style={{ width: '12px', height: '12px', background: theme.color, borderRadius: '50%' }}></span>
                            AI Diagnostic Confidence Matrix
                        </h3>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
                            {[
                                { name: 'Visual Evidence', score: patient.imageScore },
                                { name: 'Acoustic SONAR', score: patient.audioScore },
                                { name: 'Medical Text', score: patient.textScore }
                            ].map((m, i) => (
                                <div key={i}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontWeight: 700, color: '#94a3b8', marginBottom: '0.5rem' }}>
                                        <span>{m.name}</span>
                                        <span style={{ color: '#f8fafc' }}>{(m.score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                                        <div style={{ width: `${m.score * 100}%`, height: '100%', background: theme.gradient, borderRadius: '3px', transition: 'width 1s ease-out' }}></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Recommendation Tactical Box */}
                    <div style={{ background: theme.bg, borderLeft: `4px solid ${theme.color}`, padding: '1.5rem', borderRadius: '0.75rem', marginBottom: '2rem' }}>
                        <div style={{ fontSize: '0.7rem', fontWeight: 900, color: theme.color, textTransform: 'uppercase', marginBottom: '0.5rem', letterSpacing: '0.05em' }}>
                            AI Recommended Action
                        </div>
                        <p style={{ margin: 0, fontSize: '1rem', lineHeight: '1.6', color: '#e2e8f0', fontWeight: 500 }}>
                            {patient.recommendation}
                        </p>
                    </div>

                    {/* Action Footer */}
                    <button
                        className="history-btn"
                        onClick={onViewHistory}
                        style={{ 
                            width: '100%', 
                            padding: '1.25rem', 
                            background: 'transparent',
                            border: '1px dashed rgba(255,255,255,0.2)',
                            borderRadius: '1rem',
                            color: '#cbd5e1',
                            fontWeight: 700,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '0.75rem',
                            cursor: 'pointer',
                            transition: 'all 0.2s'
                        }}
                        onMouseOver={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; e.currentTarget.style.borderColor = theme.color; e.currentTarget.style.color = '#fff'; }}
                        onMouseOut={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)'; e.currentTarget.style.color = '#cbd5e1'; }}
                    >
                        📜 Retrieve Comprehensive Clinical History
                    </button>
                </div>
            </div>
        </div>
    );
};

export default PatientDetails;
