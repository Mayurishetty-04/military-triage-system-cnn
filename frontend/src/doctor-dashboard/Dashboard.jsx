import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import Sidebar from './Sidebar';
import PatientTable from './PatientTable';
import PatientDetails from './PatientDetails';
import AIRecommendationModal from './AIRecommendationModal';
import EmergencyResponseModal from './EmergencyResponseModal';
import ChatPanel from '../components/ChatPanel';
import useMessageNotifications from '../hooks/useMessageNotifications';
import './styles.css';

const Dashboard = () => {
    const navigate = useNavigate();
    const [patients, setPatients] = useState([]);
    const [registeredPatients, setRegisteredPatients] = useState([]);
    const [activeView, setActiveView] = useState('dashboard');
    const [selectedPatient, setSelectedPatient] = useState(null);
    const [selectedAIPatient, setSelectedAIPatient] = useState(null);
    const [selectedEmergencyPatient, setSelectedEmergencyPatient] = useState(null);
    const [loading, setLoading] = useState(true);
    const [history, setHistory] = useState([]);
    const [showHistory, setShowHistory] = useState(false);
    const [notifications, setNotifications] = useState([]);       // Array of alert objects
    const [chatOpen, setChatOpen] = useState(false);
    const [selectedChatPatientId, setSelectedChatPatientId] = useState(null);
    const [selectedChatPatientName, setSelectedChatPatientName] = useState(null);
    const [chatInboxOpen, setChatInboxOpen] = useState(false);
    const [conversations, setConversations] = useState([]);
    const seenKeys = React.useRef(new Set());                     // Track seen (patientId + timestamp) combos
    const isFirstLoad = React.useRef(true);                       // Suppress alerts on initial page load

    // Filter and Sort states for Patients view
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('ALL');
    const [sortOrder, setSortOrder] = useState('newest');

    const handleLogout = () => {
        localStorage.removeItem("token");
        localStorage.removeItem("role");
        localStorage.removeItem("userId");
        navigate("/");
    };

    // Request browser notification permission on load and store user ID
    useEffect(() => {
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
        
        const token = localStorage.getItem("token");
        if (token) {
            try {
                const payload = JSON.parse(atob(token.split(".")[1]));
                if (payload.user_id) {
                    localStorage.setItem("userId", payload.user_id);
                }
            } catch (e) {
                console.error("Failed to decode token:", e);
            }
        }
    }, []);

    // Enable message notifications
    useMessageNotifications();

    const fetchConversations = async () => {
        try {
            const res = await axios.get('http://127.0.0.1:8000/messages/conversations', {
                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
            });
            setConversations(Array.isArray(res.data) ? res.data : []);
        } catch (err) {
            // Silent fail - chat panel will surface issues when opened
        }
    };

    // Poll conversations so new patients appear in inbox
    useEffect(() => {
        fetchConversations();
        const interval = setInterval(fetchConversations, 3000);
        return () => clearInterval(interval);
    }, []);

    const fetchRegisteredPatients = async () => {
        try {
            const backendHost = window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname;
            const res = await axios.get(`http://${backendHost}:8000/users/patients`, {
                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
            });
            setRegisteredPatients(Array.isArray(res.data) ? res.data : []);
        } catch (err) {
            // Non-fatal: dashboard can still run on triage submissions
        }
    };

    useEffect(() => {
        fetchPatients();
        fetchRegisteredPatients();
        const interval = setInterval(fetchPatients, 10000);
        const rosterInterval = setInterval(fetchRegisteredPatients, 30000);
        return () => {
            clearInterval(interval);
            clearInterval(rosterInterval);
        };
    }, []);

    const dismissNotification = (id) => {
        setNotifications(prev => prev.filter(n => n.id !== id));
    };

    const fetchPatients = async () => {
        try {
            const backendHost = window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname;
            // Fetch ALL records (history) to properly detect every new triage event
            const res = await axios.get(`http://${backendHost}:8000/patients`, {
                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
            });
            const data = res.data;

            // Detect new triage events using unique (patientId + timestamp) key
            const newAlerts = [];
            data.forEach(p => {
                const key = `${p.patientId}__${p.timestamp}`;
                if (!seenKeys.current.has(key)) {
                    seenKeys.current.add(key);
                    // Only alert after the first load is complete (avoid false alarms on page load)
                    if (!isFirstLoad.current) {
                        newAlerts.push({
                            id: key,
                            patient: p,
                            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                        });
                    }
                }
            });
            // Mark first load complete so subsequent polls can trigger alerts
            isFirstLoad.current = false;

            if (newAlerts.length > 0) {
                setNotifications(prev => [...newAlerts, ...prev].slice(0, 5)); // keep latest 5
                // Browser push notification
                newAlerts.forEach(alert => {
                    if ('Notification' in window && Notification.permission === 'granted') {
                        new Notification('🚨 New Triage Input', {
                            body: `Patient ${alert.patient.patientId} — Status: ${alert.patient.status}`,
                            icon: '/favicon.ico'
                        });
                    }
                });
            }

            setPatients(data);
        } catch (err) {
            console.error("Failed to fetch patients", err);
        } finally {
            setLoading(false);
        }
    };

    const fetchHistory = async (patientId) => {
        try {
            const backendHost = window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname;
            const res = await axios.get(`http://${backendHost}:8000/patients/${patientId}/history`, {
                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
            });
            setHistory(res.data);
            setShowHistory(true);
        } catch (err) {
            console.error("Failed to fetch history:", err);
        }
    };

    const handleAcknowledge = (recordId) => {
        setPatients(prev => prev.map(p => 
            p.id === recordId ? { ...p, is_acknowledged: 1 } : p
        ));
    };

    const renderContent = () => {
        if (loading) return <div className="loading-state">Loading triage data...</div>;

        switch (activeView) {
            case 'dashboard':
                const criticalPatients = patients.filter(p => p.status === 'RED' && p.is_acknowledged !== 1);
                const stats = {
                    total: registeredPatients.length || patients.length,
                    triageTotal: patients.length,
                    red: criticalPatients.length,
                    yellow: patients.filter(p => p.status === 'YELLOW').length,
                    green: patients.filter(p => p.status === 'GREEN').length,
                    black: patients.filter(p => p.status === 'BLACK').length
                };
                const recentPatients = [...patients].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)).slice(0, 5);

                return (
                    <div className="view-container">

                        <header className="view-header">
                            <h2 style={{ fontSize: '2.5rem', letterSpacing: '-0.03em' }}>Triage Dashboard</h2>
                            <p>Real-time clinical monitoring & strategic oversight</p>
                        </header>

                        {/* Summary Grid */}
                        <div className="stats-grid">
                            <div className="stat-card" style={{ '--accent-color': '#64748b' }}>
                                <div className="stat-value">{stats.total}</div>
                                <div className="stat-label">Total Patients (Registered)</div>
                                <div style={{ marginTop: '0.25rem', fontSize: '0.75rem', color: '#94a3b8', fontWeight: 700 }}>
                                    Triage submissions: {stats.triageTotal}
                                </div>
                            </div>
                            <div className="stat-card" style={{ '--accent-color': '#ef4444' }}>
                                <div className="stat-value" style={{ color: '#ef4444' }}>{stats.red}</div>
                                <div className="stat-label">Critical Alerts</div>
                            </div>
                            <div className="stat-card" style={{ '--accent-color': '#eab308' }}>
                                <div className="stat-value" style={{ color: '#eab308' }}>{stats.yellow}</div>
                                <div className="stat-label">Moderate Risk</div>
                            </div>
                            <div className="stat-card" style={{ '--accent-color': '#22c55e' }}>
                                <div className="stat-value" style={{ color: '#22c55e' }}>{stats.green}</div>
                                <div className="stat-label">Stable Cases</div>
                            </div>
                        </div>

                        {/* Critical Alerts Card UI */}
                        <div className="critical-alerts-section">
                            <div className="section-title">
                                <span>🚨</span>
                                <span>Immediate Medical Response Required</span>
                                <span style={{ color: '#ef4444', fontSize: '0.875rem' }}>({stats.red})</span>
                            </div>
                            <div className="alert-card-grid">
                                {criticalPatients.map(p => (
                                    <div key={p.patientId} className="alert-card alert-card-red">
                                        <div className="alert-id">{p.patientId}</div>
                                        <div className="alert-vitals">
                                            <div className="vital-stat">
                                                <span className="label">Priority</span>
                                                <span className="value">P1</span>
                                            </div>
                                            <div className="vital-stat">
                                                <span className="label">SpO2</span>
                                                <span className="value">{p.spo2}%</span>
                                            </div>
                                            <div className="vital-stat">
                                                <span className="label">HR</span>
                                                <span className="value">{p.heartRate} bpm</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                                {criticalPatients.length === 0 && (
                                    <div className="empty-state" style={{ gridColumn: '1 / -1' }}>
                                        No active critical alerts. System status nominal.
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Recent Activity & Actions */}
                        <div className="dashboard-bottom-row">
                            <div className="activity-panel">
                                <h3 className="section-title" style={{ fontSize: '1rem' }}>Clinical Activity Log</h3>
                                <div className="activity-list">
                                    {recentPatients.map(p => (
                                        <div key={p.patientId} className="activity-item">
                                            <div className="activity-meta">
                                                <span className={`status-dot status-${p.status}`}></span>
                                                <strong>{p.patientId}</strong>
                                                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>- Triage: {p.status}</span>
                                            </div>
                                            <div className="activity-time">
                                                {new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div className="action-card-grid">
                                <button className="action-btn" onClick={() => setActiveView('patients')}>
                                    <div className="action-icon">📋</div>
                                    <div className="action-label">
                                        <strong>Patient Archive</strong>
                                        <span>Access full medical records</span>
                                    </div>
                                </button>
                                <button className="action-btn" onClick={() => setActiveView('alerts')}>
                                    <div className="action-icon">⚠️</div>
                                    <div className="action-label">
                                        <strong>Emergency Mode</strong>
                                        <span>Deploy urgent response triage</span>
                                    </div>
                                </button>
                                <button className="action-btn" onClick={() => setActiveView('analytics')}>
                                    <div className="action-icon">📊</div>
                                    <div className="action-label">
                                        <strong>Clinical Intelligence</strong>
                                        <span>View system performance metrics</span>
                                    </div>
                                </button>
                            </div>
                        </div>
                    </div>
                );
            case 'patients':
                // Merge registered patients with latest triage record (if any)
                const triageByUserId = new Map(
                    patients
                        .filter(p => p.user_id != null)
                        .map(p => [Number(p.user_id), p])
                );

                const mergedPatients = (registeredPatients.length > 0 ? registeredPatients : []).map(u => {
                    const triage = triageByUserId.get(Number(u.id));
                    if (triage) return triage;
                    return {
                        patientId: u.patientId || `USER-${u.id}`,
                        patientName: u.username,
                        status: 'NONE',
                        survivalProbability: null,
                        priority: null,
                        timestamp: null,
                        user_id: u.id
                    };
                });

                // If roster endpoint failed, fall back to triage list
                const baseList = mergedPatients.length > 0 ? mergedPatients : patients;

                let filteredPatients = baseList.filter(p =>
                    String(p.patientId || '').toLowerCase().includes(searchTerm.toLowerCase()) &&
                    (statusFilter === 'ALL' || p.status === statusFilter || (statusFilter === 'ALL' && p.status === 'NONE'))
                );

                if (sortOrder === 'newest') {
                    filteredPatients.sort((a, b) => {
                        const at = a.timestamp ? new Date(a.timestamp).getTime() : 0;
                        const bt = b.timestamp ? new Date(b.timestamp).getTime() : 0;
                        return bt - at;
                    });
                } else if (sortOrder === 'priority') {
                    filteredPatients.sort((a, b) => {
                        const ap = a.priority != null ? a.priority : 999;
                        const bp = b.priority != null ? b.priority : 999;
                        return ap - bp;
                    });
                }

                return (
                    <div className="view-container">
                        <header className="view-header">
                            <h2>Patient Management</h2>
                            <p>Complete medical database and triage history</p>
                        </header>

                        <div className="filter-bar">
                            <input
                                type="text"
                                placeholder="Search Patient ID..."
                                className="search-input"
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                            />
                            <select className="filter-select" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                                <option value="ALL">All Status</option>
                                <option value="RED">RED</option>
                                <option value="YELLOW">YELLOW</option>
                                <option value="GREEN">GREEN</option>
                                <option value="BLACK">BLACK</option>
                                <option value="NONE">NO TRIAGE</option>
                            </select>
                            <select className="filter-select" value={sortOrder} onChange={(e) => setSortOrder(e.target.value)}>
                                <option value="newest">Sort: Newest First</option>
                                <option value="priority">Sort: Highest Priority</option>
                            </select>
                        </div>

                        <PatientTable patients={filteredPatients} onSelectPatient={setSelectedPatient} />


                    </div>
                );
            case 'alerts':
                const unacknowledged = patients.filter(p => p.is_acknowledged !== 1);
                return (
                    <div className="view-container">
                        <header className="view-header" style={{ textAlign: 'center' }}>
                            <h2 style={{ color: '#ef4444', fontSize: '2.5rem' }}>🚨 URGENT RESPONSE MODE</h2>
                            <p style={{ color: '#fca5a5', fontWeight: 'bold' }}>IMMEDIATE MEDICAL INTERVENTION REQUIRED</p>
                        </header>
                        <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '2rem', borderRadius: '1.5rem', border: '2px solid rgba(255,255,255,0.1)' }}>
                            <PatientTable patients={unacknowledged} onSelectPatient={setSelectedEmergencyPatient} />
                        </div>
                    </div>
                );
            case 'analytics':
                const counts = {
                    RED: patients.filter(p => p.status === 'RED').length,
                    YELLOW: patients.filter(p => p.status === 'YELLOW').length,
                    GREEN: patients.filter(p => p.status === 'GREEN').length,
                    BLACK: patients.filter(p => p.status === 'BLACK').length
                };
                const total = patients.length || 1;

                // System Load Logic
                const loadPercent = Math.round((counts.RED * 2.5 + counts.YELLOW * 1.5) / (total * 2) * 100);
                const loadLevel = loadPercent > 70 ? 'CRITICAL' : loadPercent > 30 ? 'MODERATE' : 'NOMINAL';
                const loadColor = loadLevel === 'CRITICAL' ? '#ef4444' : loadLevel === 'MODERATE' ? '#eab308' : '#22c55e';

                // For now, let's create a reusable SVG Donut Component logic here
                const getCircumference = (radius) => 2 * Math.PI * radius;
                const radius = 70;
                const circumference = getCircumference(radius);

                let currentOffset = 0;
                const segments = [
                    { label: 'RED', count: counts.RED, color: '#ef4444' },
                    { label: 'YELLOW', count: counts.YELLOW, color: '#eab308' },
                    { label: 'GREEN', count: counts.GREEN, color: '#22c55e' },
                    { label: 'BLACK', count: counts.BLACK, color: '#475569' }
                ].map(s => {
                    const percent = (s.count / total) * 100;
                    const strokeDasharray = `${(percent / 100) * circumference} ${circumference}`;
                    const strokeDashoffset = -currentOffset;
                    currentOffset += (percent / 100) * circumference;
                    return { ...s, percent, strokeDasharray, strokeDashoffset };
                });

                // Advanced Automated Intelligence Engine
                const insights = [];
                const criticalRatio = counts.RED / total;

                // 1. Critical Mass Detection
                if (counts.RED >= 3) {
                    insights.push({
                        type: 'alert',
                        title: 'Critical Saturation',
                        text: `P1 cases at ${(criticalRatio * 100).toFixed(0)}% of total. Clinical bandwidth reaching dangerous levels.`
                    });
                } else if (counts.RED > 0) {
                    insights.push({
                        type: 'neutral',
                        title: 'Active Critical Monitoring',
                        text: `${counts.RED} critical case(s) detected. Current trajectory manageable.`
                    });
                }

                // 2. Triage Efficiency & Bottlenecks
                if (counts.YELLOW > 4) {
                    insights.push({
                        type: 'alert',
                        title: 'Delayed Triage Risk',
                        text: `High volume of P2 (Yellow) patients suggests a bottleneck in secondary assessment.`
                    });
                }

                // 3. Operational Stability
                if (counts.GREEN / total > 0.7) {
                    insights.push({
                        type: 'success',
                        title: 'Operational Readiness: HIGH',
                        text: 'System stable with significant reserve capacity for incoming influx.'
                    });
                }

                // 4. Mortality Intelligence
                if (counts.BLACK > 0) {
                    insights.push({
                        type: 'alert',
                        title: 'Field Mortality Protocol',
                        text: `${counts.BLACK} P0 (Black) case(s) confirmed. Activate non-survivable handling procedures.`
                    });
                }

                if (insights.length === 0) insights.push({ type: 'neutral', title: 'Nominal Operations', text: 'All clinical metrics within expected strategic variance.' });

                // Quantified Resource Suggestions
                const medEvacCount = Math.ceil(counts.RED / 2);
                const teamCount = Math.max(1, Math.ceil((counts.RED * 2 + counts.YELLOW) / 3));

                return (
                    <div className="view-container">
                        <header className="view-header">
                            <h2 style={{ fontSize: '2.5rem', letterSpacing: '-0.03em' }}>Clinical Intelligence</h2>
                            <p>Strategic metrics and operational decision support</p>
                        </header>

                        <div className="analytics-dashboard">
                            {/* Left: Visualization & Load */}
                            <div className="analytics-card">
                                <h3 className="section-title" style={{ fontSize: '1rem' }}>Triage Distribution</h3>
                                <div className="chart-container">
                                    <svg width="240" height="240" viewBox="0 0 200 200" style={{ transform: 'rotate(-90deg)' }}>
                                        {segments.map((s, i) => (
                                            <circle
                                                key={i}
                                                cx="100"
                                                cy="100"
                                                r={radius}
                                                fill="transparent"
                                                stroke={s.color}
                                                strokeWidth="25"
                                                strokeDasharray={s.strokeDasharray}
                                                strokeDashoffset={s.strokeDashoffset}
                                                className="donut-segment"
                                                style={{ transition: 'all 0.3s' }}
                                            >
                                                <title>{`${s.label}: ${s.count} (${s.percent.toFixed(1)}%)`}</title>
                                            </circle>
                                        ))}
                                    </svg>
                                    <div className="donut-center" style={{ transform: 'none' }}>
                                        <div className="total-count">{patients.length}</div>
                                        <div className="total-label">Patients</div>
                                    </div>
                                </div>
                                <p style={{ textAlign: 'center', fontSize: '0.75rem', color: '#64748b', marginTop: '-1rem' }}>
                                    Hover over segments for details
                                </p>

                                <div className="load-meter-container">
                                    <div className="load-status">
                                        <span>SYSTEM LOAD</span>
                                        <span style={{ color: loadColor }}>{loadLevel}</span>
                                    </div>
                                    <div className="load-bar-bg">
                                        <div className="load-bar-fill" style={{ width: `${Math.min(loadPercent, 100)}%`, background: loadColor }}></div>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#64748b' }}>
                                        <span>{loadPercent}% Operational Stress</span>
                                        {loadLevel !== 'NOMINAL' && <div className="trend-pill trend-up">↑ Intensity Spike</div>}
                                    </div>
                                </div>
                            </div>

                            {/* Right: AI Insights & Resources */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                                <div className="analytics-card">
                                    <h3 className="section-title" style={{ fontSize: '1rem' }}>Strategic AI Insights</h3>
                                    <div className="insight-grid">
                                        {insights.map((insight, idx) => (
                                            <div key={idx} className={`insight-card ${insight.type}`}>
                                                <div className="insight-title">{insight.title}</div>
                                                <div className="insight-text">{insight.text}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="analytics-card">
                                    <h3 className="section-title" style={{ fontSize: '1rem' }}>Resource Suggestions</h3>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                        <div className="trend-item">
                                            <div style={{ fontSize: '1.25rem' }}>🚁</div>
                                            <div>
                                                <div style={{ fontWeight: 'bold' }}>{medEvacCount > 0 ? `Deploy ${medEvacCount} MedEvac Units` : 'Standby Evacuation'}</div>
                                                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                                                    {counts.RED > 0 ? `Urgent air extraction prioritized for ${counts.RED} P1 cases.` : 'Extraction assets nominal on standby.'}
                                                </div>
                                            </div>
                                        </div>
                                        <div className="trend-item">
                                            <div style={{ fontSize: '1.25rem' }}>👨‍⚕️</div>
                                            <div>
                                                <div style={{ fontWeight: 'bold' }}>Personnel Allocation</div>
                                                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                                                    Recommend {teamCount} response team(s) for current patient density.
                                                </div>
                                            </div>
                                        </div>
                                        <div className="trend-item">
                                            <div style={{ fontSize: '1.25rem' }}>💊</div>
                                            <div>
                                                <div style={{ fontWeight: 'bold' }}>Inventory Prep</div>
                                                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                                                    {counts.RED + counts.YELLOW > 4 ? 'Critical medkit depletion risk. Expedite restocking.' : 'Current supplies sufficient for field load.'}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                );
            case 'ai':
                return (
                    <div className="view-container">
                        <header className="view-header">
                            <h2>AI Clinical Recommendations</h2>
                            <p>Intelligent treatment paths and resource optimization</p>
                        </header>
                        <div className="recommendations-list">
                            {patients.map(p => (
                                <div
                                    key={p.patientId}
                                    className="recommendation-card"
                                    onClick={() => setSelectedAIPatient(p)}
                                    style={p.status === 'RED' ? { borderLeft: '4px solid #ef4444', background: 'rgba(239, 68, 68, 0.1)' } : {}}
                                >
                                    <div className="card-header">
                                        <span className={`status-dot status-${p.status}`}></span>
                                        <strong style={p.status === 'RED' ? { color: '#ef4444' } : {}}>{p.patientId}</strong>
                                        {p.status === 'RED' && <span style={{ fontSize: '0.75rem', background: '#ef4444', color: '#fff', padding: '2px 6px', borderRadius: '4px' }}>HIGH PRIORITY</span>}
                                    </div>
                                    <p style={{ color: p.status === 'RED' ? '#fca5a5' : '#cbd5e1' }}>{p.recommendation}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                );
            default:
                return <div>Select a view from the sidebar.</div>;
        }
    };

    return (
        <div className="dashboard-layout">
            <Sidebar
                activeView={activeView}
                setActiveView={setActiveView}
                onLogout={handleLogout}
            />
            <main className={`dashboard-content ${activeView === 'map' ? 'full-bleed' : ''}`}>
                {renderContent()}
            </main>

            {/* Chat Inbox Button */}
            <button
                onClick={() => setChatInboxOpen(true)}
                title="Open Chat Inbox"
                style={{
                    position: 'fixed',
                    top: '1.25rem',
                    right: '1.25rem',
                    zIndex: 25000,
                    padding: '0.75rem 1rem',
                    borderRadius: '999px',
                    border: '1px solid rgba(56, 189, 248, 0.35)',
                    background: 'linear-gradient(135deg, rgba(2,132,199,0.9) 0%, rgba(37,99,235,0.9) 100%)',
                    color: '#fff',
                    fontWeight: 800,
                    cursor: 'pointer',
                    boxShadow: '0 10px 30px rgba(0,0,0,0.45)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem'
                }}
            >
                💬 Inbox
                {conversations.some(c => (c.unread_count || 0) > 0) && (
                    <span style={{
                        marginLeft: '0.15rem',
                        minWidth: '22px',
                        height: '22px',
                        padding: '0 7px',
                        borderRadius: '999px',
                        background: '#ef4444',
                        color: '#fff',
                        fontSize: '0.75rem',
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                        {conversations.reduce((sum, c) => sum + (c.unread_count || 0), 0)}
                    </span>
                )}
            </button>

            {/* Chat Inbox Modal */}
            {chatInboxOpen && (
                <div
                    onClick={() => setChatInboxOpen(false)}
                    style={{
                        position: 'fixed',
                        inset: 0,
                        background: 'rgba(0,0,0,0.6)',
                        backdropFilter: 'blur(6px)',
                        zIndex: 24000,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        padding: '1rem'
                    }}
                >
                    <div
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            width: '100%',
                            maxWidth: '520px',
                            background: '#0f172a',
                            border: '1px solid rgba(255,255,255,0.08)',
                            borderRadius: '1rem',
                            boxShadow: '0 25px 60px rgba(0,0,0,0.55)',
                            overflow: 'hidden'
                        }}
                    >
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '1rem 1.25rem',
                            background: 'linear-gradient(90deg, #2563eb 0%, #0284c7 100%)',
                            color: '#fff'
                        }}>
                            <div style={{ fontWeight: 900, letterSpacing: '0.02em' }}>💬 Chat Inbox</div>
                            <button
                                onClick={() => setChatInboxOpen(false)}
                                style={{
                                    background: 'rgba(255,255,255,0.2)',
                                    border: 'none',
                                    color: '#fff',
                                    width: '34px',
                                    height: '34px',
                                    borderRadius: '0.5rem',
                                    cursor: 'pointer',
                                    fontSize: '1.1rem'
                                }}
                            >
                                ✕
                            </button>
                        </div>

                        <div style={{ padding: '0.75rem', maxHeight: '70vh', overflowY: 'auto' }}>
                            {conversations.length === 0 ? (
                                <div style={{ padding: '1.25rem', color: '#94a3b8', textAlign: 'center' }}>
                                    No conversations yet.
                                </div>
                            ) : (
                                conversations.map((c) => (
                                    <button
                                        key={c.id}
                                        onClick={() => {
                                            setSelectedChatPatientId(c.patient_id);
                                            setSelectedChatPatientName(c.patient_username || "Patient");
                                            setChatInboxOpen(false);
                                            setChatOpen(true);
                                        }}
                                        style={{
                                            width: '100%',
                                            textAlign: 'left',
                                            background: 'rgba(30,41,59,0.55)',
                                            border: '1px solid rgba(255,255,255,0.06)',
                                            borderRadius: '0.85rem',
                                            padding: '0.9rem 1rem',
                                            marginBottom: '0.65rem',
                                            cursor: 'pointer',
                                            color: '#e2e8f0',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            gap: '0.75rem'
                                        }}
                                    >
                                        <div style={{ minWidth: 0 }}>
                                            <div style={{ fontWeight: 900, color: '#f8fafc' }}>
                                                {c.patient_username || "Patient"}{" "}
                                                <span style={{ color: '#94a3b8', fontWeight: 700, fontSize: '0.8rem' }}>
                                                    (id: {c.patient_id})
                                                </span>
                                            </div>
                                            <div style={{
                                                color: '#94a3b8',
                                                fontSize: '0.85rem',
                                                marginTop: '0.25rem',
                                                whiteSpace: 'nowrap',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                maxWidth: '420px'
                                            }}>
                                                {c.last_message || "—"}
                                            </div>
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>
                                            {(c.unread_count || 0) > 0 && (
                                                <span style={{
                                                    minWidth: '26px',
                                                    height: '22px',
                                                    padding: '0 8px',
                                                    borderRadius: '999px',
                                                    background: '#ef4444',
                                                    color: '#fff',
                                                    fontWeight: 900,
                                                    fontSize: '0.75rem',
                                                    display: 'inline-flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center'
                                                }}>
                                                    {c.unread_count}
                                                </span>
                                            )}
                                            <span style={{ color: '#38bdf8', fontWeight: 900 }}>Open →</span>
                                        </div>
                                    </button>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            )}

            {showHistory && (
                <div className="history-modal-overlay" onClick={() => setShowHistory(false)}>
                    <div className="history-modal" onClick={e => e.stopPropagation()} style={{ background: '#1e293b', width: '100%', maxWidth: '600px', borderRadius: '1.5rem', padding: '2rem', maxHeight: '80vh', overflowY: 'auto', position: 'relative' }}>
                        <button className="close-btn" onClick={() => setShowHistory(false)}>&times;</button>
                        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, marginBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem' }}>Clinical History Timeline</h2>
                        <div className="timeline-container" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                            {history.length > 0 ? history.map((record, idx) => (
                                <div key={idx} className="timeline-item" style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start' }}>
                                    <div className="timeline-date" style={{ minWidth: '130px', fontSize: '0.85rem', color: '#94a3b8', paddingTop: '0.2rem' }}>
                                        {new Date(record.timestamp).toLocaleString()}
                                    </div>
                                    <div className="timeline-content" style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '0.75rem', flex: 1, border: '1px solid rgba(255,255,255,0.05)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                                            <div className={`status-pill ${record.status.toLowerCase()}`}>{record.status}</div>
                                            <span style={{ fontSize: '0.85rem', color: '#38bdf8', fontWeight: 'bold' }}>{record.survivalProbability}% Survival</span>
                                        </div>
                                        <div className="timeline-vitals" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.85rem', color: '#cbd5e1' }}>
                                            <span><strong>SpO2:</strong> {record.spo2}%</span>
                                            <span><strong>Heart Rate:</strong> {record.heartRate} bpm</span>
                                        </div>
                                        {(record.imageScore > 0 || record.audioScore > 0) && (
                                            <div style={{ marginTop: '0.75rem', fontSize: '0.8rem', color: '#94a3b8', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.5rem' }}>
                                                <strong>AI Flags:</strong> Image: {(record.imageScore * 100).toFixed(1)}% | Audio: {(record.audioScore * 100).toFixed(1)}%
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )) : (
                                <div style={{ color: '#94a3b8', textAlign: 'center', padding: '2rem 0' }}>No history found.</div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* 🔔 Global Notification Toast Stack — appears on all views */}
            {notifications.length > 0 && (
                <div style={{
                    position: 'fixed', bottom: '1.5rem', right: '1.5rem',
                    display: 'flex', flexDirection: 'column-reverse', gap: '0.75rem',
                    zIndex: 20000, maxWidth: '380px', width: '100%'
                }}>
                    {notifications.map(notif => {
                        const statusColor = {
                            RED: '#ef4444', YELLOW: '#eab308',
                            GREEN: '#22c55e', BLACK: '#475569'
                        }[notif.patient.status] || '#38bdf8';

                        return (
                            <div key={notif.id} style={{
                                background: '#1e293b',
                                border: `1px solid ${statusColor}55`,
                                borderLeft: `4px solid ${statusColor}`,
                                borderRadius: '0.75rem',
                                padding: '1rem 1.25rem',
                                boxShadow: '0 10px 40px rgba(0,0,0,0.4)',
                                display: 'flex', alignItems: 'flex-start',
                                gap: '0.75rem', animation: 'slideInRight 0.3s ease'
                            }}>
                                <div style={{ fontSize: '1.5rem', marginTop: '-2px' }}>
                                    {notif.patient.status === 'RED' ? '🚨' :
                                        notif.patient.status === 'YELLOW' ? '⚠️' :
                                            notif.patient.status === 'BLACK' ? '💀' : '✅'}
                                </div>
                                <div style={{ flex: 1 }}>
                                    <div style={{ fontWeight: 'bold', color: '#f8fafc', fontSize: '0.95rem' }}>
                                        New Triage Input
                                    </div>
                                    <div style={{ color: '#94a3b8', fontSize: '0.8rem', marginTop: '0.2rem' }}>
                                        <strong style={{ color: statusColor }}>ID: {notif.patient.patientId}</strong>
                                        {notif.patient.patientName && (
                                            <span style={{ color: '#a78bfa', marginLeft: '6px' }}>👤 {notif.patient.patientName}</span>
                                        )}
                                    </div>
                                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                                        <span style={{ background: `${statusColor}22`, color: statusColor, padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold' }}>
                                            {notif.patient.status}
                                        </span>
                                        <span style={{ color: '#64748b', fontSize: '0.75rem', alignSelf: 'center' }}>{notif.time}</span>
                                    </div>
                                    <button
                                        onClick={() => { setSelectedPatient(notif.patient); dismissNotification(notif.id); }}
                                        style={{ marginTop: '0.6rem', fontSize: '0.75rem', color: '#38bdf8', background: 'transparent', border: 'none', cursor: 'pointer', padding: 0, textDecoration: 'underline' }}
                                    >
                                        View Patient →
                                    </button>
                                </div>
                                <button onClick={() => dismissNotification(notif.id)} style={{ background: 'transparent', border: 'none', color: '#475569', cursor: 'pointer', fontSize: '1.1rem', lineHeight: 1, marginTop: '-2px' }}>
                                    ×
                                </button>
                            </div>
                        );
                    })}
                </div>
            )}

            {selectedPatient && (
                <PatientDetails
                    patient={selectedPatient}
                    onClose={() => setSelectedPatient(null)}
                    onViewHistory={() => fetchHistory(selectedPatient.patientId)}
                    onOpenChat={(patientUserId, patientName) => {
                        setSelectedChatPatientId(patientUserId);
                        setSelectedChatPatientName(patientName);
                        setChatOpen(true);
                    }}
                />
            )}

            {selectedAIPatient && (
                <AIRecommendationModal
                    patient={selectedAIPatient}
                    onClose={() => setSelectedAIPatient(null)}
                />
            )}

            {selectedEmergencyPatient && (
                <EmergencyResponseModal
                    patient={selectedEmergencyPatient}
                    onClose={() => setSelectedEmergencyPatient(null)}
                    onAcknowledge={handleAcknowledge}
                />
            )}

            {/* Chat Panel */}
            <ChatPanel
                recipientId={selectedChatPatientId}
                recipientName={selectedChatPatientName || "Patient"}
                isOpen={chatOpen}
                onClose={() => setChatOpen(false)}
            />
        </div>
    );
};

export default Dashboard;
