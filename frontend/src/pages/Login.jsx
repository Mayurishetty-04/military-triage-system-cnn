import { useState, useEffect } from "react";
import { useNavigate, Link, useLocation } from "react-router-dom";
import API from "../services/api";

function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Get role from state or localStorage
  const role = location.state?.role || localStorage.getItem("role") || "patient";

  useEffect(() => {
    // Sync localStorage if role came from state
    if (location.state?.role) {
      localStorage.setItem("role", location.state.role);
    }
  }, [location.state]);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await API.post("/auth/login", {
        username,
        password,
        role: role
      });

      localStorage.setItem("token", res.data.access_token);
      localStorage.setItem("role", res.data.role);
      if (res.data.patientId) {
        localStorage.setItem("patientId", res.data.patientId);
      }

      if (res.data.role === "doctor") {
        navigate("/dashboard");
      } else {
        navigate("/triage");
      }
    } catch (err) {
      const msg = err.response?.data?.detail || "Invalid credentials. Please try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const roleLabel = role.charAt(0).toUpperCase() + role.slice(1);
  const themeColor = role === "doctor" ? "amber-400" : "cyan-400";
  const emoji = role === "doctor" ? "🩺" : "🏥";

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-black flex items-center justify-center overflow-hidden relative">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/4 w-64 h-64 bg-amber-500/10 rounded-full blur-3xl animate-float"></div>
      </div>

      <div className="absolute top-6 left-6">
        <Link to="/" className="text-gray-400 hover:text-white flex items-center gap-2 group transition-all font-bold uppercase tracking-widest text-[10px]">
          <span className="group-hover:-translate-x-1 transition-transform">←</span> Back to Role Selection
        </Link>
      </div>

      <div className="relative w-full max-w-md mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-20">
        <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-xl p-8 sm:p-10 rounded-2xl sm:rounded-3xl shadow-2xl border border-slate-700/50 transform animate-scale-in">
          <div className="text-center mb-10 animate-fade-in">
            <div className="inline-block mb-4 text-6xl drop-shadow-lg">{emoji}</div>
            <h1 className={`text-3xl sm:text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-${themeColor} via-white to-${themeColor} mb-2`}>
              {roleLabel} Login
            </h1>
            <p className="text-sm text-gray-300 font-light italic">Military Emergency Triage</p>
            <div className={`h-1 w-16 bg-gradient-to-r from-${themeColor} to-transparent mx-auto rounded-full mt-4`}></div>
          </div>

          {location.state?.msg && (
            <div className={`mb-6 p-4 bg-green-500/20 border-l-4 border-green-500 rounded-lg text-green-200 text-xs font-semibold`}>
              {location.state.msg}
            </div>
          )}

          {error && (
            <div className="mb-6 p-4 bg-red-500/20 border-l-4 border-red-500 rounded-lg text-red-200 text-sm font-semibold animate-slide-in-right">
              <span className="mr-2">⚠️</span>{error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <label className="block text-gray-300 text-sm font-semibold uppercase tracking-widest">Username</label>
              <input
                type="text"
                placeholder="Enter username"
                className={`w-full px-4 py-4 rounded-xl bg-slate-700/30 border border-slate-600/50 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-${themeColor}/50 focus:border-transparent transition-all`}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <label className="block text-gray-300 text-sm font-semibold uppercase tracking-widest">Password</label>
              <input
                type="password"
                placeholder="Enter password"
                className={`w-full px-4 py-4 rounded-xl bg-slate-700/30 border border-slate-600/50 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-${themeColor}/50 focus:border-transparent transition-all`}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`w-full bg-gradient-to-r ${role === 'doctor' ? 'from-amber-500 to-yellow-400' : 'from-cyan-600 to-blue-500'} text-slate-900 font-black py-4 rounded-xl shadow-2xl transform hover:scale-[1.02] transition-all disabled:opacity-50 uppercase tracking-widest flex items-center justify-center gap-3`}
            >
              {loading ? "Authenticating..." : `Secure ${roleLabel} Login`}
            </button>
          </form>

          <div className="flex items-center gap-4 my-8">
            <div className="flex-1 h-px bg-slate-700"></div>
            <span className="text-gray-500 text-[10px] font-bold uppercase tracking-widest">secure gateway</span>
            <div className="flex-1 h-px bg-slate-700"></div>
          </div>

          <p className="text-gray-400 text-center text-sm">
            Need an account?{" "}
            <Link
              to="/register"
              state={{ role }}
              className={`text-${themeColor} font-bold hover:underline transition-all`}
            >
              Register as {roleLabel}
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Login;
