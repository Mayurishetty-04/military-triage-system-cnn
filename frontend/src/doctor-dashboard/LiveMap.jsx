import React, { useEffect, useState, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import 'leaflet/dist/leaflet.css';
import Sidebar from './Sidebar';

// Custom Marker Icons based on Status
const getMarkerIcon = (status, isNew, isHighlighted) => {
    const colors = {
        RED: '#ef4444',
        YELLOW: '#eab308',
        GREEN: '#22c55e',
        BLACK: '#475569'
    };
    const color = colors[status] || '#3b82f6';
    const size = isNew ? 32 : 24;

    const html = `
        <div class="custom-marker ${isNew ? 'pulse-marker' : ''} ${isHighlighted ? 'glow-marker' : ''}" style="background-color: ${color}; color: ${color}; width: ${size}px; height: ${size}px;">
            <div class="marker-dot"></div>
        </div>
    `;

    return L.divIcon({
        className: 'marker-container',
        html: html,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2]
    });
};

// Map center and flyTo management
function MapController({ center, zoom, shouldFly }) {
    const map = useMap();
    useEffect(() => {
        if (center && shouldFly) {
            map.flyTo(center, zoom, {
                duration: 1.5,
                easeLinearity: 0.25
            });
        } else if (center) {
            map.setView(center, zoom);
        }
    }, [center, zoom, map, shouldFly]);
    return null;
}

const LiveMap = ({ patients: initialPatients, isStandAlone = false }) => {
    const navigate = useNavigate();
    const [patients, setPatients] = useState(initialPatients || []);
    const [selectedPatient, setSelectedPatient] = useState(null);
    const [mapCenter, setMapCenter] = useState([12.9716, 77.5946]); // Bangalore
    const [zoom, setZoom] = useState(13);
    const [shouldFly, setShouldFly] = useState(false);
    const [addresses, setAddresses] = useState({});
    const [history, setHistory] = useState([]);
    const [showHistory, setShowHistory] = useState(false);
    const [highlightedStatus, setHighlightedStatus] = useState(null);

    // Standalone data fetching
    useEffect(() => {
        if (isStandAlone) {
            const fetchPatients = async () => {
                try {
                    const backendHost = window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname;
                    const res = await axios.get(`http://${backendHost}:8000/patients`, {
                        headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
                    });
                    setPatients(res.data);
                } catch (err) {
                    console.error("Failed to fetch patients", err);
                }
            };
            fetchPatients();
            const interval = setInterval(fetchPatients, 10000);
            return () => clearInterval(interval);
        }
    }, [isStandAlone]);

    // Handle new patient focus
    useEffect(() => {
        const patientsWithLocation = patients.filter(p => p.latitude && p.longitude);
        if (patientsWithLocation.length > 0) {
            const sorted = [...patientsWithLocation].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            const latestPatient = sorted[0];
            const isNew = isPatientNew(latestPatient.timestamp);

            if (isNew && (latestPatient.latitude !== mapCenter[0] || latestPatient.longitude !== mapCenter[1])) {
                setMapCenter([latestPatient.latitude, latestPatient.longitude]);
                setZoom(15);
                setShouldFly(true);
            }
        }
    }, [patients]);

    const fetchAddress = async (lat, lon, id) => {
        if (addresses[id]) return;
        try {
            const res = await fetch(`https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}`);
            const data = await res.json();
            setAddresses(prev => ({ ...prev, [id]: data.display_name || "Tactical Grid Location" }));
        } catch (err) {
            setAddresses(prev => ({ ...prev, [id]: `${lat.toFixed(4)}, ${lon.toFixed(4)}` }));
        }
    };

    const fetchHistory = async (patientId) => {
        try {
            const res = await axios.get(`http://127.0.0.1:8000/patients/${patientId}/history`, {
                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
            });
            setHistory(res.data);
            setShowHistory(true);
        } catch (err) {
            console.error("Failed to fetch history:", err);
        }
    };

    const isPatientNew = (timestamp) => {
        if (!timestamp) return false;
        const patientTime = new Date(timestamp).getTime();
        const now = new Date().getTime();
        return (now - patientTime) < 5 * 60 * 1000;
    };

    const stats = useMemo(() => ({
        total: patients.length,
        red: patients.filter(p => p.status === 'RED').length,
        yellow: patients.filter(p => p.status === 'YELLOW').length,
        green: patients.filter(p => p.status === 'GREEN').length,
        black: patients.filter(p => p.status === 'BLACK').length,
        newCount: patients.filter(p => isPatientNew(p.timestamp)).length
    }), [patients]);

    const criticalPatients = patients.filter(p => p.status === 'RED').slice(0, 3);
    const recentPatients = [...patients].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)).slice(0, 5);

    const handleMarkerClick = (patient) => {
        setSelectedPatient(patient);
        fetchAddress(patient.latitude, patient.longitude, patient.id);
    };

    return (
        <div className={`dashboard-layout ${isStandAlone ? 'standalone-map' : ''}`}>
            {isStandAlone && (
                <Sidebar
                    activeView="map"
                    setActiveView={() => { }}
                    onLogout={() => {
                        localStorage.removeItem("token");
                        navigate("/");
                    }}
                />
            )}

            <main className="dashboard-content full-bleed" style={{ position: 'relative' }}>
                {/* Map Legend */}
                <div className="map-legend">
                    <h4>Tactical Legend</h4>
                    <div className="legend-item"><span className="dot red"></span> Critical (RED)</div>
                    <div className="legend-item"><span className="dot yellow"></span> Moderate (YELLOW)</div>
                    <div className="legend-item"><span className="dot green"></span> Stable (GREEN)</div>
                    <div className="legend-item"><span className="dot black"></span> No Survival (BLACK)</div>
                    <div className="legend-item"><span className="dot pulse"></span> New Patient (&lt; 5m)</div>
                </div>

                {/* Right Side Tactical Panel */}
                <aside className="tactical-panel">
                    <div className="panel-header">
                        <h3>TACTICAL OVERLAY</h3>
                        <div className="system-status">SYSTEM ACTIVE</div>
                    </div>

                    <div className="panel-sections">
                        {/* 1. Critical Summary */}
                        <section className="panel-section">
                            <div className="section-label">🔴 CRITICAL SUMMARY ({stats.red})</div>
                            <div className="mini-card-list">
                                {criticalPatients.map(p => (
                                    <div key={p.id} className="mini-card red-border" onClick={() => setSelectedPatient(p)}>
                                        <strong>{p.patientId}</strong>
                                        <span>Surv: {p.survivalProbability}%</span>
                                    </div>
                                ))}
                                {stats.red === 0 && <p className="empty-text">No critical cases detected.</p>}
                            </div>
                        </section>

                        {/* 2. Triage Distribution */}
                        <section className="panel-section">
                            <div className="section-label">📊 TRIAGE DISTRIBUTION</div>
                            <div className="distribution-grid">
                                <div 
                                    className={`dist-item ${highlightedStatus === 'RED' ? 'active-highlight' : ''}`} 
                                    onClick={() => setHighlightedStatus(prev => prev === 'RED' ? null : 'RED')}
                                >
                                    <span className="count red">{stats.red}</span> RED
                                </div>
                                <div 
                                    className={`dist-item ${highlightedStatus === 'YELLOW' ? 'active-highlight' : ''}`} 
                                    onClick={() => setHighlightedStatus(prev => prev === 'YELLOW' ? null : 'YELLOW')}
                                >
                                    <span className="count yellow">{stats.yellow}</span> YEL
                                </div>
                                <div 
                                    className={`dist-item ${highlightedStatus === 'GREEN' ? 'active-highlight' : ''}`} 
                                    onClick={() => setHighlightedStatus(prev => prev === 'GREEN' ? null : 'GREEN')}
                                >
                                    <span className="count green">{stats.green}</span> GRN
                                </div>
                                <div 
                                    className={`dist-item ${highlightedStatus === 'BLACK' ? 'active-highlight' : ''}`} 
                                    onClick={() => setHighlightedStatus(prev => prev === 'BLACK' ? null : 'BLACK')}
                                >
                                    <span className="count black">{stats.black}</span> BLK
                                </div>
                            </div>
                        </section>

                        {/* 3. New Patients Live Feed */}
                        <section className="panel-section">
                            <div className="section-label">✨ LIVE FEED (NEW)</div>
                            <div className="live-feed">
                                {patients.filter(p => isPatientNew(p.timestamp)).map(p => (
                                    <div key={p.id} className="feed-item">
                                        <span className="feed-dot pulse"></span>
                                        <strong>{p.patientId}</strong>
                                        <span className="feed-time">JUST NOW</span>
                                    </div>
                                ))}
                                {stats.newCount === 0 && <p className="empty-text">Waiting for new field reports...</p>}
                            </div>
                        </section>

                        {/* 4. Selected Patient Details */}
                        <section className="panel-section selected-details">
                            <div className="section-label">👤 PATIENT INTEL</div>
                            {selectedPatient ? (
                                <div className="patient-intel-card">
                                    <div className="intel-row"><strong>ID:</strong> {selectedPatient.patientId}</div>
                                    <div className="intel-row"><strong>STATUS:</strong> <span className={`status-text ${selectedPatient.status.toLowerCase()}`}>{selectedPatient.status}</span></div>
                                    <div className="intel-row"><strong>SURVIVAL:</strong> {selectedPatient.survivalProbability}%</div>
                                    <div className="intel-row"><strong>VITALS:</strong> {selectedPatient.heartRate} BPM | {selectedPatient.spo2}% SpO2</div>
                                    <div className="intel-recommendation">{selectedPatient.recommendation}</div>
                                    <button
                                        className="history-btn"
                                        onClick={() => fetchHistory(selectedPatient.patientId)}
                                    >
                                        📜 View Clinical History
                                    </button>
                                </div>
                            ) : (
                                <p className="empty-text">Select a marker for intelligence report.</p>
                            )}
                        </section>

                        {/* 5. AI Suggestions */}
                        <section className="panel-section ai-insights">
                            <div className="section-label">🧠 AI STRATEGIC INSIGHTS</div>
                            <div className="ai-insight-box">
                                {stats.red > 3 ? (
                                    <div className="ai-alert">⚠️ HIGH CRITICAL DENSITY: Deploy Emergency Response Team Alpha.</div>
                                ) : stats.total > 10 ? (
                                    <div className="ai-neutral">ℹ️ RESOURCE LOAD: Monitor stable patients and prioritize P1 extractions.</div>
                                ) : (
                                    <div className="ai-success">✅ FIELD STATUS NOMINAL: Maintain standard observation protocols.</div>
                                )}
                            </div>
                        </section>
                    </div>
                </aside>

                {/* Clinical History Modal */}
                {showHistory && (
                    <div className="history-modal-overlay">
                        <div className="history-modal">
                            <div className="modal-header">
                                <h2>CLINICAL TIMELINE: {selectedPatient?.patientId}</h2>
                                <button className="close-btn" onClick={() => setShowHistory(false)}>×</button>
                            </div>
                            <div className="timeline-container">
                                {history.map((entry, idx) => (
                                    <div key={entry.id} className="timeline-item">
                                        <div className="timeline-marker"></div>
                                        <div className="timeline-content">
                                            <div className="timeline-time">
                                                {new Date(entry.timestamp).toLocaleString()}
                                                {idx === 0 && <span className="latest-badge">LATEST</span>}
                                            </div>
                                            <div className="timeline-details">
                                                <span className={`status-pill ${entry.status.toLowerCase()}`}>{entry.status}</span>
                                                <div className="vitals-snapshot">
                                                    <span>💓 {entry.heartRate} BPM</span>
                                                    <span>🩸 {entry.spo2}% SpO2</span>
                                                    <span>📈 {entry.survivalProbability}% Surv</span>
                                                </div>
                                                <div className="recommendation-snapshot">{entry.recommendation}</div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                <MapContainer
                    center={mapCenter}
                    zoom={zoom}
                    style={{ height: '100%', width: '100%', background: '#0f172a' }}
                    zoomControl={false}
                >
                    <TileLayer
                        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                        attribution='&copy; CARTO'
                    />

                    <MapController center={mapCenter} zoom={zoom} shouldFly={shouldFly} />

                    {patients.filter(p => p.latitude && p.longitude).map((patient) => (
                        <Marker
                            key={patient.id}
                            position={[patient.latitude, patient.longitude]}
                            icon={getMarkerIcon(patient.status, isPatientNew(patient.timestamp), highlightedStatus === patient.status)}
                            eventHandlers={{
                                click: () => handleMarkerClick(patient),
                            }}
                        >
                            <Popup className="dark-popup">
                                <div className="popup-content">
                                    <strong>ID: {patient.patientId}</strong>
                                    <div className={`status-pill ${patient.status.toLowerCase()}`}>
                                        {patient.status}
                                    </div>
                                    <div className="address-box">
                                        <p style={{ fontSize: '0.85rem', color: '#38bdf8' }}>
                                            {addresses[patient.id] || "Analyzing address..."}
                                        </p>
                                    </div>
                                    <p>Survival: {patient.survivalProbability}%</p>
                                    <p className="rec-text">{patient.recommendation}</p>
                                </div>
                            </Popup>
                        </Marker>
                    ))}
                </MapContainer>
            </main>
        </div>
    );
};

export default LiveMap;
