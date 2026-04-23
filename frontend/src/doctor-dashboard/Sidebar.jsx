import React from 'react';
import { useNavigate } from 'react-router-dom';

const Sidebar = ({ activeView, setActiveView, onLogout, isMobileMenuOpen, setIsMobileMenuOpen }) => {
    const navigate = useNavigate();
    const menuItems = [
        { id: 'dashboard', label: 'Dashboard', icon: '📊' },
        { id: 'patients', label: 'Patients', icon: '👥' },
        { id: 'map', label: 'Live Map', icon: '📍', route: '/live-map' },
        { id: 'alerts', label: 'Emergency Alerts', icon: '🚨' },
        { id: 'analytics', label: 'Analytics', icon: '📈' },
        { id: 'ai', label: 'AI Recommendations', icon: '🧠' },
    ];

    const handleClick = (item) => {
        if (item.route) {
            navigate(item.route);
        } else {
            if (window.location.pathname !== '/dashboard') {
                navigate('/dashboard');
            }
            setActiveView(item.id);
        }
        if (setIsMobileMenuOpen) setIsMobileMenuOpen(false);
    };

    return (
        <div className={`sidebar ${isMobileMenuOpen ? 'open' : ''}`}>
            <div className="sidebar-logo">
                <div className="logo-icon">➕</div>
                <span>TRIAGE MD</span>
            </div>

            <nav className="sidebar-nav">
                {menuItems.map((item) => (
                    <button
                        key={item.id}
                        className={`nav-item ${activeView === item.id ? 'active' : ''}`}
                        onClick={() => handleClick(item)}
                    >
                        <span className="icon-span">{item.icon}</span>
                        <span>{item.label}</span>
                    </button>
                ))}
            </nav>

            <div className="sidebar-footer">
                <button className="nav-item logout-item" onClick={onLogout}>
                    <span className="icon-span">🚪</span>
                    <span>Logout</span>
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
