<template>
  <div class="min-h-screen bg-gray-50">
    <div class="container mx-auto px-4 py-8">
      <h1 class="text-3xl font-bold text-center mb-8 text-gray-800">
        Breast Cancer Detection
      </h1>
      
      <!-- Upload Section -->
      <div class="max-w-2xl mx-auto bg-white rounded-lg shadow-lg p-6 mb-8">
        <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <input
            ref="fileInput"
            type="file"
            accept="image/*"
            @change="handleFileSelect"
            class="hidden"
          />
          
          <div v-if="!selectedFile" @click="$refs.fileInput.click()" class="cursor-pointer">
            <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48">
              <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
            <p class="text-gray-600">Click to upload histopathological image</p>
            <p class="text-sm text-gray-400 mt-2">PNG, JPG up to 10MB</p>
          </div>
          
          <div v-else class="space-y-4">
            <img :src="imagePreview" alt="Preview" class="mx-auto max-h-64 rounded-lg" />
            <p class="text-sm text-gray-600">{{ selectedFile.name }}</p>
            <div class="flex gap-4 justify-center">
              <button @click="predictImage" :disabled="loading" class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50">
                {{ loading ? 'Analyzing...' : 'Analyze Image' }}
              </button>
              <button @click="clearImage" class="bg-gray-500 text-white px-6 py-2 rounded-lg hover:bg-gray-600">
                Clear
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Results Section -->
      <div v-if="prediction" class="max-w-4xl mx-auto bg-white rounded-lg shadow-lg p-6">
        <h2 class="text-2xl font-bold mb-6 text-gray-800">Analysis Results</h2>
        
        <!-- Main Prediction -->
        <div class="grid md:grid-cols-2 gap-6 mb-6">
          <div class="bg-gray-50 rounded-lg p-4">
            <h3 class="font-semibold text-lg mb-2">Prediction</h3>
            <p class="text-2xl font-bold" :class="prediction.risk_level === 'high' ? 'text-red-600' : 'text-green-600'">
              {{ formatClassName(prediction.prediction) }}
            </p>
            <p class="text-sm text-gray-600 mt-1">
              Confidence: {{ (prediction.confidence * 100).toFixed(1) }}%
            </p>
          </div>
          
          <div class="bg-gray-50 rounded-lg p-4">
            <h3 class="font-semibold text-lg mb-2">Risk Level</h3>
            <span class="inline-block px-3 py-1 rounded-full text-sm font-medium" 
                  :class="prediction.risk_level === 'high' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'">
              {{ prediction.risk_level.toUpperCase() }} RISK
            </span>
          </div>
        </div>

        <!-- Probabilities -->
        <div class="mb-6">
          <h3 class="font-semibold text-lg mb-4">All Probabilities</h3>
          <div class="space-y-2">
            <div v-for="(prob, className) in prediction.probabilities" :key="className" class="flex items-center">
              <div class="w-32 text-sm">{{ formatClassName(className) }}</div>
              <div class="flex-1 bg-gray-200 rounded-full h-2 mx-3">
                <div class="bg-blue-600 h-2 rounded-full" :style="{ width: (prob * 100) + '%' }"></div>
              </div>
              <div class="w-16 text-sm text-right">{{ (prob * 100).toFixed(1) }}%</div>
            </div>
          </div>
        </div>

        <!-- Explanation -->
        <div v-if="prediction.textual_explanation" class="bg-blue-50 rounded-lg p-4">
          <h3 class="font-semibold text-lg mb-2">Medical Explanation</h3>
          <p class="text-gray-700 leading-relaxed">{{ prediction.textual_explanation }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'App',
  data() {
    return {
      selectedFile: null,
      imagePreview: null,
      loading: false,
      prediction: null,
      apiBase: 'http://localhost:8000'
    }
  },
  methods: {
    handleFileSelect(event) {
      const file = event.target.files[0]
      if (file) {
        this.selectedFile = file
        const reader = new FileReader()
        reader.onload = (e) => {
          this.imagePreview = e.target.result
        }
        reader.readAsDataURL(file)
        this.prediction = null
      }
    },
    
    clearImage() {
      this.selectedFile = null
      this.imagePreview = null
      this.prediction = null
    },
    
    async predictImage() {
      if (!this.selectedFile) return
      
      this.loading = true
      const formData = new FormData()
      formData.append('image', this.selectedFile)
      
      try {
        const response = await axios.post(`${this.apiBase}/api/predict`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        this.prediction = response.data
      } catch (error) {
        console.error('Prediction failed:', error)
        alert('Analysis failed. Please try again.')
      } finally {
        this.loading = false
      }
    },
    
    formatClassName(className) {
      return className.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
  }
}
</script>