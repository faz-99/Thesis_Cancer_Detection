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
          <h3 class="font-semibold text-lg mb-4">Top 3 Probabilities</h3>
          <div class="space-y-2">
            <div v-for="(prob, className) in prediction.probabilities" :key="className" class="flex items-center">
              <div class="w-32 text-sm">{{ className }}</div>
              <div class="flex-1 bg-gray-200 rounded-full h-2 mx-3">
                <div class="bg-blue-600 h-2 rounded-full" :style="{ width: Math.min(prob * 100, 100) + '%' }"></div>
              </div>
              <div class="w-16 text-sm text-right">{{ (prob * 100).toFixed(1) }}%</div>
            </div>
          </div>
        </div>

        <!-- Test Image -->
        <div class="mb-6">
          <h3 class="font-semibold text-lg mb-4">Test Base64 Image</h3>
          <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" alt="Test" class="w-16 h-16 bg-red-500" />
        </div>

        <!-- Visualization -->
        <div v-if="prediction && prediction.visualization" class="mb-6">
          <h3 class="font-semibold text-lg mb-4">Visual Analysis</h3>
          <div class="space-y-4">
            <div v-if="prediction.visualization.original">
              <h4 class="text-sm font-medium mb-2">Original Image</h4>
              <img :src="prediction.visualization.original" alt="Original" class="max-w-xs rounded-lg border" />
            </div>
            <div v-if="prediction.visualization.heatmap">
              <h4 class="text-sm font-medium mb-2">Attention Heatmap</h4>
              <img :src="prediction.visualization.heatmap" alt="Heatmap" class="max-w-xs rounded-lg border" />
            </div>
            <div v-if="prediction.visualization.overlay">
              <h4 class="text-sm font-medium mb-2">Overlay Analysis</h4>
              <img :src="prediction.visualization.overlay" alt="Overlay" class="max-w-xs rounded-lg border" />
            </div>
          </div>
          <p class="text-sm text-gray-600 mt-2">
            Red areas indicate regions that most influenced the AI's decision
          </p>
        </div>

        <!-- Explanation -->
        <div v-if="prediction.textual_explanation" class="bg-blue-50 rounded-lg p-4">
          <h3 class="font-semibold text-lg mb-2">Medical Explanation</h3>
          <div v-if="typeof prediction.textual_explanation === 'object'">
            <p class="text-gray-700 leading-relaxed mb-3">{{ prediction.textual_explanation.explanation }}</p>
            <div v-if="prediction.textual_explanation.relevant_facts && prediction.textual_explanation.relevant_facts.length > 0">
              <h4 class="font-medium text-sm mb-2">Relevant Medical Facts:</h4>
              <ul class="text-sm text-gray-600 space-y-1">
                <li v-for="fact in prediction.textual_explanation.relevant_facts" :key="fact" class="flex items-start">
                  <span class="text-blue-500 mr-2">•</span>
                  {{ fact }}
                </li>
              </ul>
            </div>
          </div>
          <p v-else class="text-gray-700 leading-relaxed">{{ prediction.textual_explanation }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const config = useRuntimeConfig()

const selectedFile = ref(null)
const imagePreview = ref(null)
const loading = ref(false)
const prediction = ref(null)

const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    selectedFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => {
      imagePreview.value = e.target.result
    }
    reader.readAsDataURL(file)
    prediction.value = null
  }
}

const clearImage = () => {
  selectedFile.value = null
  imagePreview.value = null
  prediction.value = null
}

const predictImage = async () => {
  if (!selectedFile.value) return
  
  loading.value = true
  const formData = new FormData()
  formData.append('image', selectedFile.value)
  
  try {
    const response = await $fetch(`${config.public.apiBase}/api/predict`, {
      method: 'POST',
      body: formData
    })
    console.log('API Response:', response)
    console.log('Visualization data:', response.visualization)
    prediction.value = response
  } catch (error) {
    console.error('Prediction failed:', error)
    alert('Analysis failed. Please try again.')
  } finally {
    loading.value = false
  }
}

const formatClassName = (className) => {
  return className.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}
</script>