import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";

function Home() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split(".")[1]));
        setUsername(payload.sub || payload.username || "Personnel");
      } catch (e) {
        setUsername("Personnel");
      }
    }
  }, []);

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    localStorage.removeItem("patientId");
    navigate("/");
  };

  return (
    <div className="w-full min-h-screen bg-[#0f172a] text-white overflow-x-hidden font-['Inter']">
      {/* Dynamic Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] bg-blue-600/10 rounded-full blur-[120px] animate-pulse"></div>
        <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] bg-cyan-600/10 rounded-full blur-[120px] animate-pulse" style={{animationDelay: '1.5s'}}></div>
      </div>

      <div className="relative z-10 max-w-lg mx-auto px-6 pt-12 pb-24">
        {/* Header Section */}
        <header className="mb-10 animate-fade-in">
          <div className="flex justify-between items-start mb-6">
            <div>
              <p className="text-cyan-400 font-semibold tracking-[0.2em] uppercase text-[10px] mb-1">Status: Operational</p>
              <h1 className="text-3xl font-black font-['Outfit'] tracking-tight">
                Welcome, <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">{username}</span>
              </h1>
            </div>
            <div className="w-12 h-12 rounded-full glass-panel flex items-center justify-center border-cyan-500/30">
              <span className="text-2xl">🪖</span>
            </div>
          </div>
          
          <div className="glass-panel p-5 rounded-2xl border-white/5 flex items-center gap-4">
            <div className="w-10 h-10 bg-amber-500/20 rounded-xl flex items-center justify-center text-amber-500">
              <span className="text-xl">📡</span>
            </div>
            <div>
              <p className="text-xs text-gray-400 font-medium">Tactical Network</p>
              <p className="text-sm font-bold text-gray-100">AI-Nodes Connected</p>
            </div>
          </div>
        </header>

        {/* Quick Actions Grid */}
        <section className="space-y-6">
          <h2 className="text-sm font-bold text-gray-500 uppercase tracking-widest px-1">Primary Modules</h2>
          
          <div className="grid grid-cols-1 gap-4">
            {/* Main Action: New Assessment */}
            <button 
              onClick={() => navigate("/triage")}
              className="group relative w-full p-1 rounded-3xl bg-gradient-to-r from-blue-600 via-cyan-500 to-blue-600 bg-[length:200%_100%] animate-[gradient_4s_linear_infinite] hover:scale-[1.02] transition-transform duration-300"
            >
              <div className="bg-[#0f172a] rounded-[1.4rem] p-6 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 bg-cyan-500/10 rounded-2xl flex items-center justify-center text-3xl group-hover:scale-110 transition-transform">
                    🚑
                  </div>
                  <div className="text-left">
                    <h3 className="text-xl font-black font-['Outfit']">New Triage</h3>
                    <p className="text-xs text-gray-400">Start AI Assessment</p>
                  </div>
                </div>
                <div className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-cyan-500 transition-colors">
                  <span className="text-lg">→</span>
                </div>
              </div>
            </button>

            <div className="grid grid-cols-2 gap-4">
              {/* Secondary Action: Dashboard */}
              <button 
                onClick={() => navigate("/dashboard")}
                className="glass-card p-5 rounded-3xl text-left hover:border-blue-500/50 transition-all group"
              >
                <div className="w-10 h-10 bg-blue-500/10 rounded-xl flex items-center justify-center text-xl mb-3 group-hover:scale-110 transition-transform">
                  📊
                </div>
                <h3 className="font-bold text-sm">Dashboard</h3>
                <p className="text-[10px] text-gray-500">Live Analytics</p>
              </button>

              {/* Secondary Action: Live Map */}
              <button 
                onClick={() => navigate("/live-map")}
                className="glass-card p-5 rounded-3xl text-left hover:border-emerald-500/50 transition-all group"
              >
                <div className="w-10 h-10 bg-emerald-500/10 rounded-xl flex items-center justify-center text-xl mb-3 group-hover:scale-110 transition-transform">
                  🗺️
                </div>
                <h3 className="font-bold text-sm">Field Map</h3>
                <p className="text-[10px] text-gray-500">Unit Tracking</p>
              </button>
            </div>
          </div>
        </section>

        {/* System Health */}
        <section className="mt-10">
          <h2 className="text-sm font-bold text-gray-500 uppercase tracking-widest px-1 mb-4">System Pulse</h2>
          <div className="glass-panel p-6 rounded-3xl border-white/5 space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-400">AI Accuracy</span>
              <span className="text-xs font-bold text-emerald-400">98.4%</span>
            </div>
            <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div className="h-full bg-emerald-500 w-[98.4%]"></div>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-400">Server Latency</span>
              <span className="text-xs font-bold text-cyan-400">12ms</span>
            </div>
            <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div className="h-full bg-cyan-500 w-[15%]"></div>
            </div>
          </div>
        </section>

        {/* Logout Button */}
        <button 
          onClick={logout}
          className="w-full mt-12 py-4 rounded-2xl glass-panel border-red-500/20 text-red-400 font-bold text-sm hover:bg-red-500/10 transition-all"
        >
          Terminate Session 🚪
        </button>

        <footer className="mt-12 text-center">
          <p className="text-[10px] text-gray-600 font-black uppercase tracking-[0.3em]">
            MilTriage v2.5.0 • Tactical Edition
          </p>
        </footer>
      </div>
    </div>
  );
}

export default Home;
