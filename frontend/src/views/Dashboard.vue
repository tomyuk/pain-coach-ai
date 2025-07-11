<template>
  <div class="dashboard">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-900">Dashboard</h1>
      <p class="text-gray-600 mt-2">Welcome back! Here's your pain management overview.</p>
    </div>

    <!-- Quick Stats -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <div class="bg-white rounded-lg shadow p-6">
        <div class="flex items-center">
          <div class="flex-1">
            <p class="text-sm font-medium text-gray-600">Average Pain (7 days)</p>
            <p class="text-2xl font-bold text-gray-900">{{ averagePain }}</p>
          </div>
          <div class="flex-shrink-0">
            <div class="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
              <span class="text-blue-600">📊</span>
            </div>
          </div>
        </div>
      </div>

      <div class="bg-white rounded-lg shadow p-6">
        <div class="flex items-center">
          <div class="flex-1">
            <p class="text-sm font-medium text-gray-600">Pain Trend</p>
            <p class="text-2xl font-bold text-gray-900 capitalize">{{ painTrend }}</p>
          </div>
          <div class="flex-shrink-0">
            <div class="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
              <span class="text-green-600">📈</span>
            </div>
          </div>
        </div>
      </div>

      <div class="bg-white rounded-lg shadow p-6">
        <div class="flex items-center">
          <div class="flex-1">
            <p class="text-sm font-medium text-gray-600">Total Records</p>
            <p class="text-2xl font-bold text-gray-900">{{ totalRecords }}</p>
          </div>
          <div class="flex-shrink-0">
            <div class="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center">
              <span class="text-purple-600">📝</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
      <div class="bg-white rounded-lg shadow p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Quick Pain Entry</h3>
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Current Pain Level (0-10)
            </label>
            <input
              v-model="quickPainLevel"
              type="range"
              min="0"
              max="10"
              class="w-full"
            />
            <div class="flex justify-between text-sm text-gray-500 mt-1">
              <span>0</span>
              <span class="font-medium">{{ quickPainLevel }}</span>
              <span>10</span>
            </div>
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Notes (optional)
            </label>
            <input
              v-model="quickNotes"
              type="text"
              placeholder="What were you doing before the pain?"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            @click="submitQuickEntry"
            :disabled="isLoading"
            class="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ isLoading ? 'Recording...' : 'Record Pain' }}
          </button>
        </div>
      </div>

      <div class="bg-white rounded-lg shadow p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">AI Assistant</h3>
        <p class="text-gray-600 mb-4">
          Get personalized insights and support from your AI pain coach.
        </p>
        <div class="space-y-3">
          <router-link
            to="/chat"
            class="block w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 text-center"
          >
            Start Conversation
          </router-link>
          <button
            @click="getQuickInsight"
            :disabled="isLoading"
            class="w-full bg-gray-600 text-white py-2 px-4 rounded-md hover:bg-gray-700 disabled:opacity-50"
          >
            {{ isLoading ? 'Generating...' : 'Get Quick Insight' }}
          </button>
        </div>
        <div v-if="quickInsight" class="mt-4 p-3 bg-blue-50 rounded-md">
          <p class="text-sm text-blue-800">{{ quickInsight }}</p>
        </div>
      </div>
    </div>

    <!-- Recent Activity -->
    <div class="bg-white rounded-lg shadow p-6">
      <h3 class="text-lg font-semibold text-gray-900 mb-4">Recent Pain Records</h3>
      <div v-if="recentRecords.length === 0" class="text-center py-8 text-gray-500">
        <p>No pain records yet. Start by recording your first pain entry above!</p>
      </div>
      <div v-else class="space-y-4">
        <div
          v-for="record in recentRecords"
          :key="record.id"
          class="flex items-center justify-between p-4 border border-gray-200 rounded-lg"
        >
          <div class="flex items-center space-x-4">
            <div class="flex-shrink-0">
              <div
                class="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold"
                :class="getPainLevelColor(record.pain_level)"
              >
                {{ record.pain_level }}
              </div>
            </div>
            <div>
              <p class="text-sm font-medium text-gray-900">
                {{ formatDate(record.recorded_at) }}
              </p>
              <p class="text-sm text-gray-500">
                {{ record.activity_before || 'No activity noted' }}
              </p>
            </div>
          </div>
          <div class="flex items-center space-x-2">
            <span
              v-for="type in record.pain_type || []"
              :key="type"
              class="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded-full"
            >
              {{ type }}
            </span>
          </div>
        </div>
      </div>
      <div v-if="recentRecords.length > 0" class="mt-4 text-center">
        <router-link
          to="/pain-log"
          class="text-blue-600 hover:text-blue-800 text-sm font-medium"
        >
          View all records →
        </router-link>
      </div>
    </div>

    <!-- Error Display -->
    <div v-if="error" class="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
      <div class="flex">
        <div class="flex-shrink-0">
          <span class="text-red-400">⚠️</span>
        </div>
        <div class="ml-3">
          <p class="text-sm text-red-800">{{ error }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { usePainLogStore } from '../stores/painLog'
import { chatAPI } from '../services/api'

const painLogStore = usePainLogStore()

// Reactive data
const quickPainLevel = ref(5)
const quickNotes = ref('')
const quickInsight = ref('')
const isLoading = ref(false)

// Computed properties
const averagePain = computed(() => painLogStore.averagePain)
const painTrend = computed(() => painLogStore.painTrend)
const totalRecords = computed(() => painLogStore.records.length)
const recentRecords = computed(() => painLogStore.recentRecords)
const error = computed(() => painLogStore.error)

// Methods
const submitQuickEntry = async () => {
  try {
    isLoading.value = true
    await painLogStore.quickEntry(quickPainLevel.value, quickNotes.value)
    
    // Reset form
    quickPainLevel.value = 5
    quickNotes.value = ''
    
    // Show success message
    alert('Pain entry recorded successfully!')
  } catch (err) {
    console.error('Quick entry failed:', err)
  } finally {
    isLoading.value = false
  }
}

const getQuickInsight = async () => {
  try {
    isLoading.value = true
    quickInsight.value = ''
    
    // Get recent pain data for context
    const recentData = recentRecords.value.slice(0, 5)
    
    if (recentData.length === 0) {
      quickInsight.value = 'Start recording your pain to get personalized insights!'
      return
    }
    
    // Generate insight message
    const avgPain = recentData.reduce((sum, r) => sum + r.pain_level, 0) / recentData.length
    const prompt = `Based on recent pain levels averaging ${avgPain.toFixed(1)}/10, provide a brief encouraging insight.`
    
    const response = await chatAPI.sendMessage(prompt, {
      conversation_type: 'analysis',
      pain_context: {
        current_pain: avgPain,
        recent_pattern: painTrend.value,
        total_records: totalRecords.value
      }
    })
    
    quickInsight.value = response.response
  } catch (err) {
    console.error('Quick insight failed:', err)
    quickInsight.value = 'Unable to generate insight at this time. Please try again.'
  } finally {
    isLoading.value = false
  }
}

const getPainLevelColor = (level) => {
  if (level <= 3) return 'bg-green-500'
  if (level <= 6) return 'bg-yellow-500'
  return 'bg-red-500'
}

const formatDate = (dateString) => {
  const date = new Date(dateString)
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// Lifecycle
onMounted(async () => {
  try {
    await painLogStore.fetchRecords({ limit: 10 })
    await painLogStore.fetchSummary()
  } catch (err) {
    console.error('Dashboard initialization failed:', err)
  }
})
</script>

<style scoped>
/* Component-specific styles */
.dashboard {
  /* Add any specific styling here */
}
</style>