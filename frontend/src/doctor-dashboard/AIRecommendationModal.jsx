import React from 'react';
import './styles.css';

const AIRecommendationModal = ({ patient, onClose }) => {
    if (!patient) return null;

    // Parse the recommendations from the backend (comma separated string)
    const checklist = patient.recommendation ? patient.recommendation.split(',').map(item => item.trim()) : [];

    // Determine resources based on priority severity
    const getResources = (status) => {
        switch (status) {
            case 'RED':
                return [
                    { icon: '🚁', label: '1x Immediate MedEvac Required' },
                    { icon: '👨‍⚕️', label: 'Advanced Trauma Rescue Team' },
                    { icon: '🩸', label: 'Whole Blood / Tourniquets' }
                ];
            case 'YELLOW':
                return [
                    { icon: '🚑', label: 'Priority Ground Transport' },
                    { icon: '👨‍⚕️', label: 'Standard Medic Team' },
                    { icon: '🩹', label: 'Sterile Trauma Dressings' }
                ];
            case 'GREEN':
                return [
                    { icon: '🚶‍♂️', label: 'Walking Wounded / Ambulate to Collection Point' },
                    { icon: '🏥', label: 'Basic First Aid Kit' }
                ];
            case 'BLACK':
                return [
                    { icon: '⚠️', label: 'Palliative Care / Do Not Resuscitate' },
                    { icon: '🪦', label: 'Activate Mortuary Affairs Protocol' }
                ];
            default:
                return [];
        }
    };

    const resources = getResources(patient.status);

    return (
        <div className="history-modal-overlay" onClick={onClose} style={{ zIndex: 10005, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="history-modal" onClick={(e) => e.stopPropagation()} style={{ background: '#1e293b', width: '100%', maxWidth: '700px', borderRadius: '1.5rem', padding: '2rem', maxHeight: '85vh', overflowY: 'auto', position: 'relative', border: '1px solid rgba(255,255,255,0.05)', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)' }}>

                <button className="close-btn" onClick={onClose} style={{ position: 'absolute', top: '1.2rem', right: '1.5rem', background: 'transparent', border: 'none', color: '#94a3b8', fontSize: '1.5rem', cursor: 'pointer', transition: 'color 0.2s' }}>
                    &times;
                </button>

                {/* Header Section */}
                <header style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1.5rem', marginBottom: '1.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
                        <div className={`status-dot status-${patient.status}`}></div>
                        <h2 style={{ fontSize: '1.8rem', fontWeight: 800, margin: 0, color: '#f8fafc' }}>AI Clinical Action Plan</h2>
                    </div>
                    <div style={{ color: '#cbd5e1', fontSize: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                            <span>Target: <strong style={{ color: '#fff' }}>{patient.patientId}</strong></span>
                            {patient.patientName && (
                                <div style={{ color: '#a78bfa', fontSize: '0.85rem', marginTop: '0.2rem' }}>👤 {patient.patientName}</div>
                            )}
                        </div>
                        <span className={`status-pill ${patient.status.toLowerCase()}`} style={{ fontSize: '0.75rem' }}>{patient.status} PRIORITY</span>
                    </div>
                </header>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>

                    {/* Treatment Checklist */}
                    <section>
                        <h3 style={{ fontSize: '1.2rem', color: '#38bdf8', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span>📋</span> Step-by-Step Treatment Protocol
                        </h3>
                        <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.25rem', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                            {checklist.map((task, idx) => (
                                <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', marginBottom: idx !== checklist.length - 1 ? '1rem' : '0' }}>
                                    <div style={{ background: '#334155', color: '#cbd5e1', width: '24px', height: '24px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', flexShrink: 0, marginTop: '2px' }}>
                                        {idx + 1}
                                    </div>
                                    <div style={{ color: '#f1f5f9', lineHeight: '1.4' }}>{task}</div>
                                </div>
                            ))}
                        </div>
                    </section>

                    {/* Resources Needed */}
                    <section>
                        <h3 style={{ fontSize: '1.2rem', color: '#eab308', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span>🚑</span> Resource Deployment Matrix
                        </h3>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                            {resources.map((res, idx) => (
                                <div key={idx} style={{ background: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.2)', padding: '1rem', borderRadius: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                    <div style={{ fontSize: '1.5rem' }}>{res.icon}</div>
                                    <div style={{ color: '#f8fafc', fontSize: '0.9rem', fontWeight: 500 }}>{res.label}</div>
                                </div>
                            ))}
                        </div>
                    </section>

                    {/* Underlying AI Evidence */}
                    <section>
                        <h3 style={{ fontSize: '1.2rem', color: '#a78bfa', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span>🧠</span> AI Diagnostic Reasoning
                        </h3>
                        <div style={{ background: 'rgba(167, 139, 250, 0.05)', padding: '1.25rem', borderRadius: '0.75rem', border: '1px solid rgba(167, 139, 250, 0.2)' }}>
                            <div style={{ color: '#cbd5e1', fontSize: '0.9rem', marginBottom: '1rem' }}>
                                <strong>Extracted Medical Context:</strong> "{patient.injuryType || "Undetermined mechanism of injury"}"
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1.5rem' }}>
                                <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                        <span style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase' }}>Visual (CNN)</span>
                                        <span style={{ fontSize: '0.8rem', color: '#e2e8f0', fontWeight: 'bold' }}>{(patient.imageScore * 100).toFixed(1)}%</span>
                                    </div>
                                    <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                                        <div style={{ width: `${Math.min(patient.imageScore * 100, 100)}%`, height: '100%', background: '#a78bfa', borderRadius: '4px' }}></div>
                                    </div>
                                </div>

                                <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                        <span style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase' }}>Acoustic (SONAR)</span>
                                        <span style={{ fontSize: '0.8rem', color: '#e2e8f0', fontWeight: 'bold' }}>{(patient.audioScore * 100).toFixed(1)}%</span>
                                    </div>
                                    <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                                        <div style={{ width: `${Math.min(patient.audioScore * 100, 100)}%`, height: '100%', background: '#a78bfa', borderRadius: '4px' }}></div>
                                    </div>
                                </div>

                                <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                        <span style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase' }}>Medical Text (NLP)</span>
                                        <span style={{ fontSize: '0.8rem', color: '#e2e8f0', fontWeight: 'bold' }}>{(patient.textScore * 100).toFixed(1)}%</span>
                                    </div>
                                    <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                                        <div style={{ width: `${Math.min(patient.textScore * 100, 100)}%`, height: '100%', background: '#a78bfa', borderRadius: '4px' }}></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>

                </div>
            </div>
        </div>
    );
};

export default AIRecommendationModal;
