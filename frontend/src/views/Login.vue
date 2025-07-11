<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8">
      <div>
        <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900">
          Welcome to Pain Coach AI Pascal
        </h2>
        <p class="mt-2 text-center text-sm text-gray-600">
          {{ isRegistering ? 'Create your account' : 'Sign in to your account' }}
        </p>
      </div>
      
      <form class="mt-8 space-y-6" @submit.prevent="handleSubmit">
        <div class="rounded-md shadow-sm -space-y-px">
          <div v-if="isRegistering">
            <label for="name" class="sr-only">Full Name</label>
            <input
              id="name"
              v-model="formData.name"
              type="text"
              class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
              placeholder="Full Name"
            />
          </div>
          
          <div>
            <label for="username" class="sr-only">Username</label>
            <input
              id="username"
              v-model="formData.username"
              type="text"
              required
              :class="[
                'appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm',
                isRegistering ? '' : 'rounded-t-md'
              ]"
              placeholder="Username"
            />
          </div>
          
          <div>
            <label for="password" class="sr-only">Password</label>
            <input
              id="password"
              v-model="formData.password"
              type="password"
              required
              class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
              placeholder="Password"
            />
          </div>
          
          <div v-if="isRegistering">
            <label for="email" class="sr-only">Email</label>
            <input
              id="email"
              v-model="formData.email"
              type="email"
              class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
              placeholder="Email Address"
            />
          </div>
          
          <div v-if="isRegistering">
            <label for="birth_year" class="sr-only">Birth Year</label>
            <input
              id="birth_year"
              v-model="formData.birth_year"
              type="number"
              min="1900"
              max="2100"
              class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
              placeholder="Birth Year (e.g., 1990)"
            />
          </div>
          
          <div v-if="isRegistering">
            <label for="gender" class="sr-only">Gender</label>
            <select
              id="gender"
              v-model="formData.gender"
              class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
            >
              <option value="">Select Gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
              <option value="prefer_not_to_say">Prefer not to say</option>
            </select>
          </div>
        </div>

        <div v-if="error" class="rounded-md bg-red-50 p-4">
          <div class="flex">
            <div class="flex-shrink-0">
              <span class="text-red-400">⚠️</span>
            </div>
            <div class="ml-3">
              <p class="text-sm text-red-800">{{ error }}</p>
            </div>
          </div>
        </div>

        <div>
          <button
            type="submit"
            :disabled="isLoading"
            class="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span v-if="isLoading" class="absolute left-0 inset-y-0 flex items-center pl-3">
              <div class="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>
            </span>
            {{ isLoading ? 'Please wait...' : (isRegistering ? 'Create Account' : 'Sign in') }}
          </button>
        </div>

        <div class="text-center">
          <button
            type="button"
            @click="toggleMode"
            class="text-sm text-blue-600 hover:text-blue-800"
          >
            {{ isRegistering ? 'Already have an account? Sign in' : 'Need an account? Register' }}
          </button>
        </div>
      </form>
      
      <div v-if="!isRegistering" class="mt-6">
        <div class="relative">
          <div class="absolute inset-0 flex items-center">
            <div class="w-full border-t border-gray-300" />
          </div>
          <div class="relative flex justify-center text-sm">
            <span class="px-2 bg-gray-50 text-gray-500">Demo Account</span>
          </div>
        </div>
        
        <button
          @click="loginAsDemo"
          :disabled="isLoading"
          class="mt-3 w-full flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
        >
          Try Demo Account
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const authStore = useAuthStore()

// Reactive data
const isRegistering = ref(false)
const formData = ref({
  username: '',
  password: '',
  name: '',
  email: '',
  birth_year: null,
  gender: ''
})

// Computed properties
const isLoading = computed(() => authStore.isLoading)
const error = computed(() => authStore.error)

// Methods
const handleSubmit = async () => {
  try {
    authStore.clearError()
    
    if (isRegistering.value) {
      // Register new user
      await authStore.register(formData.value)
      
      // After successful registration, switch to login
      isRegistering.value = false
      formData.value = {
        username: formData.value.username,
        password: formData.value.password,
        name: '',
        email: '',
        birth_year: null,
        gender: ''
      }
      
      alert('Account created successfully! Please sign in.')
    } else {
      // Login existing user
      await authStore.login({
        username: formData.value.username,
        password: formData.value.password
      })
      
      // Redirect to dashboard
      router.push('/')
    }
  } catch (err) {
    console.error('Authentication failed:', err)
  }
}

const toggleMode = () => {
  isRegistering.value = !isRegistering.value
  authStore.clearError()
  
  // Reset form data
  formData.value = {
    username: '',
    password: '',
    name: '',
    email: '',
    birth_year: null,
    gender: ''
  }
}

const loginAsDemo = async () => {
  try {
    authStore.clearError()
    
    await authStore.login({
      username: 'demo',
      password: 'demo'
    })
    
    // Redirect to dashboard
    router.push('/')
  } catch (err) {
    console.error('Demo login failed:', err)
  }
}
</script>