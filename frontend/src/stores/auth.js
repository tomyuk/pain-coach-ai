/**
 * Authentication store using Pinia
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { authAPI } from '../services/api'

export const useAuthStore = defineStore('auth', () => {
  // State
  const user = ref(null)
  const token = ref(localStorage.getItem('token') || null)
  const isLoading = ref(false)
  const error = ref(null)

  // Getters
  const isAuthenticated = computed(() => !!token.value)
  const currentUser = computed(() => user.value)

  // Actions
  const login = async (credentials) => {
    try {
      isLoading.value = true
      error.value = null
      
      const response = await authAPI.login(credentials)
      
      token.value = response.access_token
      user.value = { id: response.user_id }
      
      // Store token in localStorage
      localStorage.setItem('token', token.value)
      
      return response
    } catch (err) {
      error.value = err.message || 'Login failed'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const logout = async () => {
    try {
      isLoading.value = true
      
      if (token.value) {
        await authAPI.logout()
      }
      
      // Clear state
      user.value = null
      token.value = null
      error.value = null
      
      // Remove token from localStorage
      localStorage.removeItem('token')
      
    } catch (err) {
      console.error('Logout error:', err)
    } finally {
      isLoading.value = false
    }
  }

  const register = async (userData) => {
    try {
      isLoading.value = true
      error.value = null
      
      const response = await authAPI.register(userData)
      
      return response
    } catch (err) {
      error.value = err.message || 'Registration failed'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const checkAuthStatus = async () => {
    try {
      if (!token.value) {
        return false
      }
      
      const userProfile = await authAPI.getProfile()
      user.value = userProfile
      
      return true
    } catch (err) {
      console.error('Auth check failed:', err)
      // Clear invalid token
      token.value = null
      user.value = null
      localStorage.removeItem('token')
      return false
    }
  }

  const updateProfile = async (profileData) => {
    try {
      isLoading.value = true
      error.value = null
      
      const response = await authAPI.updateProfile(profileData)
      
      // Update user data
      user.value = { ...user.value, ...profileData }
      
      return response
    } catch (err) {
      error.value = err.message || 'Profile update failed'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const clearError = () => {
    error.value = null
  }

  return {
    // State
    user,
    token,
    isLoading,
    error,
    
    // Getters
    isAuthenticated,
    currentUser,
    
    // Actions
    login,
    logout,
    register,
    checkAuthStatus,
    updateProfile,
    clearError
  }
})