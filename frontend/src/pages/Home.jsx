import { useNavigate } from "react-router-dom";

function Home() {

  const navigate = useNavigate();

  const logout = () => {
    localStorage.removeItem("token");
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">

      <h1 className="text-3xl font-bold text-center text-yellow-400 mb-8">
        Military Triage System
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">

        {/* Green */}
        <div className="bg-green-700 p-5 rounded">
          <h2 className="text-xl font-bold">🟢 Green</h2>
          <p>Minor injuries. Can wait.</p>
        </div>

        {/* Yellow */}
        <div className="bg-yellow-600 p-5 rounded text-black">
          <h2 className="text-xl font-bold">🟡 Yellow</h2>
          <p>Moderate injuries. Needs treatment.</p>
        </div>

        {/* Red */}
        <div className="bg-red-700 p-5 rounded">
          <h2 className="text-xl font-bold">🔴 Red</h2>
          <p>Severe injuries. Immediate care.</p>
        </div>

        {/* Black */}
        <div className="bg-black p-5 rounded border border-gray-600">
          <h2 className="text-xl font-bold">⚫ Black</h2>
          <p>Deceased / Unsurvivable.</p>
        </div>

      </div>
      <button
        onClick={logout}
        className="bg-red-600 px-4 py-2 rounded mt-6"
      >Logout</button>

    </div>
  );
}


export default Home;
