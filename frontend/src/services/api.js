/**
 * API service for communicating with the backend
 */

import axios from 'axios'

// Base configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    
    const message = error.response?.data?.detail || error.message || 'An error occurred'
    return Promise.reject(new Error(message))
  }
)

// Authentication API
export const authAPI = {
  login: (credentials) => api.post('/api/auth/token', credentials),
  register: (userData) => api.post('/api/auth/register', userData),
  logout: () => api.post('/api/auth/logout'),
  getProfile: () => api.get('/api/auth/profile'),
  updateProfile: (profileData) => api.put('/api/auth/profile', profileData),
}

// Pain records API
export const painAPI = {
  getRecords: (params = {}) => api.get('/api/pain/records', { params }),
  getRecord: (recordId) => api.get(`/api/pain/records/${recordId}`),
  createRecord: (recordData) => api.post('/api/pain/records', recordData),
  updateRecord: (recordId, recordData) => api.put(`/api/pain/records/${recordId}`, recordData),
  deleteRecord: (recordId) => api.delete(`/api/pain/records/${recordId}`),
  quickEntry: (painLevel, notes) => api.post('/api/pain/quick-entry', null, {
    params: { pain_level: painLevel, notes }
  }),
  getSummary: (days = 7) => api.get('/api/pain/summary', { params: { days_back: days } }),
}

// AI Chat API
export const chatAPI = {
  sendMessage: (message, options = {}) => api.post('/api/ai/chat', {
    message,
    ...options
  }),
  getConversations: (limit = 10) => api.get('/api/ai/conversations', {
    params: { limit }
  }),
  analyzeMood: (text) => api.post('/api/ai/analyze-mood', null, {
    params: { text }
  }),
  emergencyCheck: (message) => api.post('/api/ai/emergency-check', { message }),
  streamChat: (message, painContext) => {
    const params = new URLSearchParams({
      message,
      ...(painContext && { pain_context: JSON.stringify(painContext) })
    })
    
    return new EventSource(`${API_BASE_URL}/api/ai/chat/stream?${params}`)
  }
}

// Analytics API
export const analyticsAPI = {
  getPainTrends: (days = 30) => api.get('/api/analytics/pain-trends', {
    params: { days }
  }),
  getCorrelations: (days = 30) => api.get('/api/analytics/correlations', {
    params: { days }
  }),
  getPatterns: (days = 30) => api.get('/api/analytics/patterns', {
    params: { days }
  }),
  getInsights: (days = 30) => api.get('/api/analytics/insights', {
    params: { days }
  }),
  exportData: (format = 'json', days = 90) => api.get('/api/analytics/export', {
    params: { format, days }
  }),
}

// Health integrations API
export const healthAPI = {
  configure: (config) => api.post('/api/health/configure', config),
  getStatus: () => api.get('/api/health/status'),
  triggerSync: (provider, daysBack = 1) => api.post('/api/health/sync', {
    provider,
    days_back: daysBack
  }),
  getData: (provider = null, daysBack = 7) => api.get('/api/health/data', {
    params: { provider, days_back: daysBack }
  }),
  getPermissions: () => api.get('/api/health/permissions'),
  removeProvider: (provider) => api.delete(`/api/health/provider/${provider}`),
  updateProviderSettings: (provider, settings) => api.put(`/api/health/provider/${provider}/settings`, settings),
  getSyncHistory: (daysBack = 30) => api.get('/api/health/sync-history', {
    params: { days_back: daysBack }
  }),
}

// System API
export const systemAPI = {
  getHealth: () => api.get('/health'),
  getStatus: () => api.get('/api/system/status'),
}

// Export default API instance
export default api