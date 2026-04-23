// Detect environment
const isAndroidEmulator = /android/i.test(navigator.userAgent) && window.location.hostname === '10.0.2.2';
const isAndroid = /android/i.test(navigator.userAgent);

// Your computer's actual Wi-Fi IP address on your local network
// Both your PC (running the backend) and your phone must be on the SAME Wi-Fi network
const MACHINE_IP = "192.168.56.1";

export const BASE_URL = isAndroidEmulator
  ? "http://10.0.2.2:8000"        // Android Studio Emulator
  : isAndroid
    ? `http://${MACHINE_IP}:8000` // Real Android Phone (same Wi-Fi)
    : "http://localhost:8000";     // Web Browser

export const WS_URL = isAndroidEmulator
  ? "ws://10.0.2.2:8000"
  : isAndroid
    ? `ws://${MACHINE_IP}:8000`
    : "ws://localhost:8000";

export default {
  BASE_URL,
  WS_URL
};