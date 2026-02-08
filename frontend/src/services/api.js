import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

API.interceptors.request.use((req) => {
  const token = localStorage.getItem("token");
  if (token) req.headers.Authorization = `Bearer ${token}`;
  return req;
});

export const predictTriage = (formData) =>
  API.post("/predict", formData);

export const getMetrics = () =>
  API.get("/model-metrics");

export default API;
