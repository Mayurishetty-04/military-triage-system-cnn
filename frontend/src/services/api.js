import axios from "axios";
import { BASE_URL } from "../config";

const API = axios.create({
  baseURL: BASE_URL,
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