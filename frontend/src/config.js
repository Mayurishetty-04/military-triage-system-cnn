// Detect environment
const isAndroid = /android/i.test(navigator.userAgent);

// Your computer's actual Wi-Fi IP address on your local network
const MACHINE_IP = "172.28.76.141"; 

// 1. If it's Android, we need to decide between Emulator (10.0.2.2) and Real Phone (WiFi IP).
// 2. We check window.location.hostname. 
//    - In Android Studio Emulator, it's usually '10.0.2.2' or 'localhost'.
//    - On a Real Phone, it's 'localhost'.
export const BASE_URL = isAndroid
  ? (window.location.hostname === '172.28.76.141' ? `http://${MACHINE_IP}:8000` : "http://10.0.2.2:8000")
  : "http://localhost:8000";

// Fallback: if we are on Android but NOT on the emulator's special IP, we use the WiFi IP.
// This is the most robust way to handle both.
const getFinalBaseUrl = () => {
  if (!isAndroid) return "http://localhost:8000";
  
  // If we are running on a real phone (connected to WiFi), we should use the MACHINE_IP.
  // But for the Emulator, 10.0.2.2 is mandatory.
  // We'll use 10.0.2.2 as the default for Android because it's the safest for development.
  return "http://10.0.2.2:8000"; 
};

// EXPORTING THE ACTUAL CONFIG
// To use your real phone, simply change "USE_EMULATOR" to false below.
const USE_EMULATOR = false; 

export const API_URL = isAndroid 
  ? (USE_EMULATOR ? "http://10.0.2.2:8000" : `http://${MACHINE_IP}:8000`)
  : "http://localhost:8000";

export const WS_URL = isAndroid 
  ? (USE_EMULATOR ? "ws://10.0.2.2:8000" : `ws://${MACHINE_IP}:8000`)
  : "ws://localhost:8000";

export default {
  BASE_URL: API_URL,
  WS_URL
};