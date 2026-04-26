import React from 'react';
import axios from 'axios';
import { BASE_URL } from '../config';
import './styles.css';

const EmergencyResponseModal = ({ patient, onClose, onAcknowledge }) => {
    const [isAcknowledging, setIsAcknowledging] = React.useState(false);
    const [isDone, setIsDone] = React.useState(patient.is_acknowledged === 1);
    const [dispatchStatus, setDispatchStatus] = React.useState(null);
    const [isDispatching, setIsDispatching] = React.useState(false);

    const handleDispatch = async () => {
        setIsDispatching(true);
        try {
            const res = await axios.post(`${BASE_URL}/patients/${patient.patientId}/dispatch`, {}, {
                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
            });
            setDispatchStatus(res.data.message);
        } catch (err) {
            console.error("Dispatch failed", err);
            setDispatchStatus("Dispatch failed. No teams available or network error.");
        } finally {
            setIsDispatching(false);
        }
    };

    if (!patient) return null;

    const handleAcknowledge = async () => {
        setIsAcknowledging(true);
        try {
            await axios.post(`${BASE_URL}/patients/${patient.id}/acknowledge`, {}, {
                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
            });
            setIsDone(true);
            if (onAcknowledge) onAcknowledge(patient.id);
            // Auto close after 1.5 seconds on success
            setTimeout(onClose, 1500);
        } catch (err) {
            console.error("Acknowledgment failed", err);
            alert("Deployment confirmation failed. Check network connection.");
        } finally {
            setIsAcknowledging(false);
        }
    };

    // Dynamic Theme Generator
    const getTheme = (status) => {
        switch (status) {
            case 'RED':
                return {
                    color: '#ef4444',
                    bg: 'linear-gradient(to r, #ef4444, #991b1b)',
                    icon: '🚨',
                    label: 'URGENT RESPONSE: RED ALERT',
                    shadow: 'rgba(239, 68, 68, 0.3)',
                    resources: [
                        { icon: '🚁', label: '1x Immediate MedEvac Required' },
                        { icon: '👨‍⚕️', label: 'Advanced Trauma Rescue Team' },
                        { icon: '🩸', label: 'Whole Blood / Tourniquets' }
                    ]
                };
            case 'YELLOW':
                return {
                    color: '#eab308',
                    bg: 'linear-gradient(to r, #eab308, #854d0e)',
                    icon: '⚠️',
                    label: 'TACTICAL ALERT: YELLOW STATUS',
                    shadow: 'rgba(234, 179, 8, 0.3)',
                    resources: [
                        { icon: '🚑', label: 'Priority Ground Transport' },
                        { icon: '👨‍⚕️', label: 'Standard Medic Team' },
                        { icon: '🩹', label: 'Sterile Trauma Dressings' }
                    ]
                };
            case 'GREEN':
                return {
                    color: '#22c55e',
                    bg: 'linear-gradient(to r, #22c55e, #166534)',
                    icon: '✅',
                    label: 'STABLE MONITORING: GREEN STATUS',
                    shadow: 'rgba(34, 197, 94, 0.3)',
                    resources: [
                        { icon: '🚶‍♂️', label: 'Walking Wounded / Collection Point' },
                        { icon: '🏥', label: 'Basic First Aid Kit' }
                    ]
                };
            case 'BLACK':
                return {
                    color: '#475569',
                    bg: 'linear-gradient(to r, #475569, #1e293b)',
                    icon: '💀',
                    label: 'MORTALITY PROTOCOL: BLACK STATUS',
                    shadow: 'rgba(71, 85, 105, 0.3)',
                    resources: [
                        { icon: '⚠️', label: 'Palliative Care / DNR' },
                        { icon: '🪦', label: 'Activate Mortuary Affairs' }
                    ]
                };
            default:
                return { color: '#3b82f6', bg: '#3b82f6', icon: 'ℹ️', label: 'PATIENT INTEL', shadow: 'rgba(59, 130, 246, 0.3)', resources: [] };
        }
    };

    const theme = getTheme(patient.status);
    const checklist = patient.recommendation ? patient.recommendation.split(',').map(item => item.trim()) : [];

    return (
        <div className="history-modal-overlay" onClick={onClose} style={{ zIndex: 10005, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="history-modal" onClick={(e) => e.stopPropagation()} style={{ background: '#0f172a', width: '100%', maxWidth: '800px', borderRadius: '1.5rem', padding: '0', maxHeight: '90vh', overflowY: 'auto', position: 'relative', border: `2px solid ${theme.color}`, boxShadow: `0 0 50px ${theme.shadow}` }}>
                
                {/* Emergency Header */}
                <div style={{ background: theme.bg, padding: '1.5rem 2rem', color: 'white', position: 'sticky', top: 0, zIndex: 10 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.25rem' }}>
                                <span style={{ fontSize: '1.5rem' }}>{theme.icon}</span>
                                <h1 style={{ fontSize: '1.5rem', fontWeight: 900, margin: 0, letterSpacing: '0.05em' }}>{theme.label}</h1>
                            </div>
                            <p style={{ margin: 0, opacity: 0.9, fontSize: '0.85rem', fontWeight: 600 }}>ID: {patient.patientId} • TIMESTAMP: {new Date(patient.timestamp).toLocaleString()}</p>
                        </div>
                        <button onClick={onClose} style={{ background: 'rgba(0,0,0,0.2)', border: 'none', color: 'white', fontSize: '1.5rem', cursor: 'pointer', width: '32px', height: '32px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>&times;</button>
                    </div>
                </div>

                <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                    
                    {/* Top Row: Vitals & Status */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
                        <div style={{ background: `${theme.color}11`, border: `1px solid ${theme.color}33`, borderRadius: '1rem', padding: '1.25rem', textAlign: 'center' }}>
                            <div style={{ color: theme.color, fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase', marginBottom: '0.5rem' }}>SpO2 Level</div>
                            <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#f8fafc' }}>{patient.spo2}%</div>
                            <div style={{ color: theme.color, fontSize: '0.7rem', fontWeight: 700 }}>{patient.status === 'RED' ? 'CRITICAL RANGE' : 'OBSERVED'}</div>
                        </div>
                        <div style={{ background: 'rgba(56, 189, 248, 0.1)', border: '1px solid rgba(56, 189, 248, 0.2)', borderRadius: '1rem', padding: '1.25rem', textAlign: 'center' }}>
                            <div style={{ color: '#38bdf8', fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase', marginBottom: '0.5rem' }}>Heart Rate</div>
                            <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#f8fafc' }}>{patient.heartRate} <span style={{ fontSize: '1rem' }}>BPM</span></div>
                            <div style={{ color: '#38bdf8', fontSize: '0.7rem', fontWeight: 700 }}>{patient.heartRate > 100 ? 'ELEVATED' : 'NOMINAL'}</div>
                        </div>
                        <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '1rem', padding: '1.25rem', textAlign: 'center' }}>
                            <div style={{ color: '#10b981', fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase', marginBottom: '0.5rem' }}>Survival Chance</div>
                            <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#f8fafc' }}>{patient.survivalProbability}%</div>
                            <div style={{ color: '#10b981', fontSize: '0.7rem', fontWeight: 700 }}>ANALYSIS CONFIDENT</div>
                        </div>
                    </div>

                    {/* Middle Row: Treatment & Resources */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '2rem' }}>
                        
                        {/* Action Checklist */}
                        <section>
                            <h3 style={{ fontSize: '1.1rem', color: '#f8fafc', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                <span style={{ background: theme.color, padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.9rem', color: theme.status === 'BLACK' ? '#fff' : 'inherit' }}>STEP</span>
                                IMMEDATE CLINICAL PROTOCOL
                            </h3>
                            <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '1rem', padding: '1.5rem' }}>
                                {checklist.map((task, idx) => (
                                    <div key={idx} style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', alignItems: 'flex-start' }}>
                                        <input type="checkbox" style={{ width: '20px', height: '20px', marginTop: '2px', cursor: 'pointer', accentColor: theme.color }} />
                                        <div style={{ color: '#f1f5f9', fontSize: '1rem', fontWeight: 500 }}>{task}</div>
                                    </div>
                                ))}
                            </div>
                        </section>

                        {/* Resource Matrix */}
                        <section>
                            <h3 style={{ fontSize: '1.1rem', color: '#f8fafc', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                <span style={{ background: '#eab308', padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.9rem', color: '#000' }}>ASSET</span>
                                RECOMMENDED DEPLOYMENT
                            </h3>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                                {theme.resources.map((res, idx) => (
                                    <div key={idx} style={{ background: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.2)', padding: '1rem', borderRadius: '0.75rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                        <div style={{ fontSize: '1.75rem' }}>{res.icon}</div>
                                        <div style={{ color: '#f8fafc', fontWeight: 600 }}>{res.label}</div>
                                    </div>
                                ))}
                            </div>
                        </section>
                    </div>

                    {/* Bottom Row: AI Reasoning & Location */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '2rem' }}>
                        
                        {/* Tactical Location */}
                        <section>
                            <h3 style={{ fontSize: '1.1rem', color: '#f8fafc', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                <span style={{ background: '#38bdf8', padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.9rem' }}>GRID</span>
                                GEOSPATIAL INTELLIGENCE
                            </h3>
                            <div style={{ background: 'rgba(56, 189, 248, 0.05)', border: '1px solid rgba(56, 189, 248, 0.2)', borderRadius: '1rem', padding: '1.5rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                                    <div style={{ background: '#38bdf8', color: '#0f172a', width: '40px', height: '40px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.25rem' }}>📍</div>
                                    <div>
                                        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
                                            {patient.latitude ? `${patient.latitude.toFixed(4)}, ${patient.longitude.toFixed(4)}` : "Position Unavailable"}
                                        </div>
                                        <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Tactical Grid Coordinates</div>
                                    </div>
                                </div>
                                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '0.5rem', fontSize: '0.85rem', color: '#cbd5e1' }}>
                                    <strong>Status:</strong> Signal strength nominal. {patient.latitude ? 'Location verified by GPS.' : 'Awaiting position lock.'}
                                </div>
                            </div>
                        </section>

                        {/* AI Evidence */}
                        <section>
                            <h3 style={{ fontSize: '1.1rem', color: '#f8fafc', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                <span style={{ background: '#a78bfa', padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.9rem' }}>INTEL</span>
                                DIAGNOSTIC EVIDENCE
                            </h3>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                <div style={{ background: 'rgba(167, 139, 250, 0.05)', border: '1px solid rgba(167, 139, 250, 0.2)', borderRadius: '0.75rem', padding: '1rem' }}>
                                    <div style={{ fontSize: '0.75rem', color: '#a78bfa', fontWeight: 800, marginBottom: '0.5rem' }}>EXTRACTED INJURY CONTEXT</div>
                                    <div style={{ color: '#f1f5f9', fontStyle: 'italic' }}>"{patient.injuryType || "No descriptive text provided"}"</div>
                                </div>
                                
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: '1rem' }}>
                                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                                        <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginBottom: '0.25rem' }}>VISUAL CNN</div>
                                        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>{(patient.imageScore * 100).toFixed(1)}%</div>
                                    </div>
                                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                                        <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginBottom: '0.25rem' }}>ACOUSTIC SONAR</div>
                                        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>{(patient.audioScore * 100).toFixed(1)}%</div>
                                    </div>
                                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                                        <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginBottom: '0.25rem' }}>MEDICAL TEXT</div>
                                        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>{(patient.textScore * 100).toFixed(1)}%</div>
                                    </div>
                                </div>
                            </div>
                        </section>
                    </div>

                    {/* Action Buttons */}
                    <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
                        <button 
                            onClick={handleDispatch} 
                            disabled={isDispatching || dispatchStatus?.includes("Assigned")}
                            style={{ 
                                flex: 1, 
                                padding: '1.25rem', 
                                background: dispatchStatus?.includes("Assigned") ? '#3b82f6' : '#ef4444', 
                                color: '#fff', 
                                border: 'none', 
                                borderRadius: '1rem', 
                                fontSize: '1.1rem', 
                                fontWeight: 800, 
                                cursor: (isDispatching || dispatchStatus?.includes("Assigned")) ? 'default' : 'pointer', 
                                transition: 'all 0.3s', 
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '0.75rem',
                                boxShadow: `0 4px 15px rgba(239, 68, 68, 0.4)`
                            }}
                        >
                            {isDispatching ? (
                                <>
                                    <span style={{ width: '20px', height: '20px', border: '3px solid #fff', borderTopColor: 'transparent', borderRadius: '50%', display: 'inline-block', animation: 'spin 1s linear infinite' }}></span>
                                    🚁 DISPATCHING...
                                </>
                            ) : dispatchStatus?.includes("Assigned") ? (
                                <>✅ {dispatchStatus.toUpperCase()}</>
                            ) : (
                                <>🚨 EMERGENCY ALERT: DISPATCH TEAM</>
                            )}
                        </button>

                        <button 
                            onClick={handleAcknowledge} 
                            disabled={isAcknowledging || isDone}
                            style={{ 
                                flex: 1, 
                                padding: '1.25rem', 
                                background: isDone ? '#10b981' : theme.color === '#f8fafc' ? '#334155' : theme.color, 
                                color: '#fff', 
                                border: 'none', 
                                borderRadius: '1rem', 
                                fontSize: '1.1rem', 
                                fontWeight: 800, 
                                cursor: (isAcknowledging || isDone) ? 'default' : 'pointer', 
                                transition: 'all 0.3s', 
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '0.75rem',
                                boxShadow: `0 4px 15px ${theme.shadow}`
                            }}
                        >
                            {isAcknowledging ? (
                                <>
                                    <span style={{ width: '20px', height: '20px', border: '3px solid #fff', borderTopColor: 'transparent', borderRadius: '50%', display: 'inline-block', animation: 'spin 1s linear infinite' }}></span>
                                    🧬 ACKNOWLEDGING...
                                </>
                            ) : isDone ? (
                                <>✅ CASE ACKNOWLEDGED</>
                            ) : (
                                <>⚡ ACKNOWLEDGE CASE</>
                            )}
                        </button>
                    </div>
                    {dispatchStatus && !dispatchStatus.includes("Assigned") && (
                        <div style={{ color: '#ef4444', textAlign: 'center', marginTop: '0.5rem', fontWeight: 'bold' }}>
                            {dispatchStatus}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default EmergencyResponseModal;
