/**
 * Pain log store using Pinia
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { painAPI } from '../services/api'

export const usePainLogStore = defineStore('painLog', () => {
  // State
  const records = ref([])
  const currentRecord = ref(null)
  const isLoading = ref(false)
  const error = ref(null)
  const summary = ref(null)

  // Getters
  const recentRecords = computed(() => 
    records.value.slice(0, 10)
  )
  
  const averagePain = computed(() => {
    if (records.value.length === 0) return 0
    const sum = records.value.reduce((acc, record) => acc + record.pain_level, 0)
    return Math.round((sum / records.value.length) * 10) / 10
  })

  const painTrend = computed(() => {
    if (records.value.length < 2) return 'stable'
    
    const recent = records.value.slice(0, 5)
    const older = records.value.slice(5, 10)
    
    if (recent.length === 0 || older.length === 0) return 'stable'
    
    const recentAvg = recent.reduce((acc, r) => acc + r.pain_level, 0) / recent.length
    const olderAvg = older.reduce((acc, r) => acc + r.pain_level, 0) / older.length
    
    if (recentAvg > olderAvg + 0.5) return 'increasing'
    if (recentAvg < olderAvg - 0.5) return 'decreasing'
    return 'stable'
  })

  // Actions
  const fetchRecords = async (options = {}) => {
    try {
      isLoading.value = true
      error.value = null
      
      const response = await painAPI.getRecords(options)
      records.value = response
      
      return response
    } catch (err) {
      error.value = err.message || 'Failed to fetch pain records'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const addRecord = async (recordData) => {
    try {
      isLoading.value = true
      error.value = null
      
      const response = await painAPI.createRecord(recordData)
      
      // Add to local records
      records.value.unshift(response)
      
      return response
    } catch (err) {
      error.value = err.message || 'Failed to add pain record'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const updateRecord = async (recordId, recordData) => {
    try {
      isLoading.value = true
      error.value = null
      
      const response = await painAPI.updateRecord(recordId, recordData)
      
      // Update local records
      const index = records.value.findIndex(r => r.id === recordId)
      if (index !== -1) {
        records.value[index] = { ...records.value[index], ...recordData }
      }
      
      return response
    } catch (err) {
      error.value = err.message || 'Failed to update pain record'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const deleteRecord = async (recordId) => {
    try {
      isLoading.value = true
      error.value = null
      
      await painAPI.deleteRecord(recordId)
      
      // Remove from local records
      records.value = records.value.filter(r => r.id !== recordId)
      
    } catch (err) {
      error.value = err.message || 'Failed to delete pain record'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const quickEntry = async (painLevel, notes = '') => {
    try {
      isLoading.value = true
      error.value = null
      
      const response = await painAPI.quickEntry(painLevel, notes)
      
      // Add to local records
      records.value.unshift(response)
      
      return response
    } catch (err) {
      error.value = err.message || 'Failed to add quick entry'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const fetchSummary = async (days = 7) => {
    try {
      isLoading.value = true
      error.value = null
      
      const response = await painAPI.getSummary(days)
      summary.value = response
      
      return response
    } catch (err) {
      error.value = err.message || 'Failed to fetch summary'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const getRecord = async (recordId) => {
    try {
      isLoading.value = true
      error.value = null
      
      const response = await painAPI.getRecord(recordId)
      currentRecord.value = response
      
      return response
    } catch (err) {
      error.value = err.message || 'Failed to fetch record'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const clearError = () => {
    error.value = null
  }

  const clearCurrentRecord = () => {
    currentRecord.value = null
  }

  return {
    // State
    records,
    currentRecord,
    isLoading,
    error,
    summary,
    
    // Getters
    recentRecords,
    averagePain,
    painTrend,
    
    // Actions
    fetchRecords,
    addRecord,
    updateRecord,
    deleteRecord,
    quickEntry,
    fetchSummary,
    getRecord,
    clearError,
    clearCurrentRecord
  }
})