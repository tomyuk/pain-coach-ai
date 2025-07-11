<template>
  <div id="app" class="min-h-screen bg-gray-50">
    <!-- Navigation -->
    <nav class="bg-white shadow-lg">
      <div class="max-w-7xl mx-auto px-4">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <h1 class="text-xl font-bold text-gray-900">Pain Coach AI Pascal</h1>
          </div>
          <div class="flex items-center space-x-4">
            <router-link to="/" class="text-gray-700 hover:text-blue-600">Dashboard</router-link>
            <router-link to="/pain-log" class="text-gray-700 hover:text-blue-600">Pain Log</router-link>
            <router-link to="/chat" class="text-gray-700 hover:text-blue-600">AI Chat</router-link>
            <router-link to="/analytics" class="text-gray-700 hover:text-blue-600">Analytics</router-link>
            <router-link to="/settings" class="text-gray-700 hover:text-blue-600">Settings</router-link>
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 px-4">
      <router-view />
    </main>

    <!-- Footer -->
    <footer class="bg-gray-800 text-white py-4 mt-12">
      <div class="max-w-7xl mx-auto px-4 text-center">
        <p>&copy; 2024 Pain Coach AI Pascal. All rights reserved.</p>
      </div>
    </footer>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useAuthStore } from './stores/auth'

export default {
  name: 'App',
  setup() {
    const authStore = useAuthStore()

    onMounted(async () => {
      // Initialize app
      await authStore.checkAuthStatus()
    })

    return {}
  }
}
</script>

<style scoped>
/* Component-specific styles */
</style>