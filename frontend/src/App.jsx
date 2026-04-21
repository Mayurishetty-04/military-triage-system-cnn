import { BrowserRouter, Routes, Route } from "react-router-dom";
import ProtectedRoute from "./components/ProtectedRoute";
import "./App.css";

import Login from "./pages/Login";
import Register from "./pages/Register";
import Home from "./pages/Home";
import Triage from "./pages/Triage";
import Dashboard from "./doctor-dashboard/Dashboard";
import LiveMap from "./doctor-dashboard/LiveMap";
import RoleSelection from "./pages/RoleSelection";

function App() {
  return (
    <BrowserRouter>
      <Routes>

        <Route path="/" element={<RoleSelection />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route
          path="/home"
          element={
            <ProtectedRoute>
              <Home />
            </ProtectedRoute>
          }
        />

        <Route
          path="/triage"
          element={
            <ProtectedRoute>
              <Triage />
            </ProtectedRoute>
          }
        />

        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />

        <Route
          path="/live-map"
          element={
            <ProtectedRoute>
              <LiveMap isStandAlone={true} />
            </ProtectedRoute>
          }
        />


      </Routes>
    </BrowserRouter>
  );
}

export default App;
