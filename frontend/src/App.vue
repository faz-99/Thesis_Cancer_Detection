<template>
  <div id="app">
    <header class="header">
      <h1>🔬 Breast Cancer Detection System</h1>
      <p>AI-Powered Histopathological Analysis</p>
    </header>

    <main class="main-content">
      <div class="upload-section">
        <div class="upload-area" @drop="handleDrop" @dragover.prevent @dragenter.prevent>
          <input type="file" ref="fileInput" @change="handleFileSelect" accept="image/*" style="display: none">
          <div v-if="!selectedImage" class="upload-placeholder" @click="$refs.fileInput.click()">
            <i class="upload-icon">📁</i>
            <p>Click or drag image here</p>
            <small>Supports: PNG, JPG, JPEG</small>
          </div>
          <div v-else class="image-preview">
            <img :src="imagePreview" alt="Selected image" />
            <button @click="clearImage" class="clear-btn">✕</button>
          </div>
        </div>
        
        <button @click="analyzeImage" :disabled="!selectedImage || loading" class="analyze-btn">
          {{ loading ? 'Analyzing...' : 'Analyze Image' }}
        </button>
      </div>

      <div v-if="results" class="results-section">
        <div class="prediction-card">
          <h3>🎯 Prediction Results</h3>
          <div class="prediction-result">
            <span class="diagnosis">{{ results.prediction }}</span>
            <span class="confidence">{{ (results.confidence * 100).toFixed(1) }}% confidence</span>
          </div>
          <div class="risk-level" :class="getRiskClass(results.prediction)">
            {{ getRiskLevel(results.prediction) }}
          </div>
        </div>

        <div class="explanation-card">
          <h3>🧠 AI Explanation</h3>
          <p>{{ results.textual_explanation.explanation }}</p>
          
          <div class="medical-facts">
            <h4>📚 Relevant Medical Knowledge:</h4>
            <ul>
              <li v-for="fact in results.textual_explanation.relevant_facts" :key="fact">
                {{ fact }}
              </li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
export default {
  name: 'App',
  data() {
    return {
      selectedImage: null,
      imagePreview: null,
      loading: false,
      results: null
    }
  },
  methods: {
    handleFileSelect(event) {
      const file = event.target.files[0]
      if (file) {
        this.selectedImage = file
        this.imagePreview = URL.createObjectURL(file)
        this.results = null
      }
    },
    
    handleDrop(event) {
      event.preventDefault()
      const file = event.dataTransfer.files[0]
      if (file && file.type.startsWith('image/')) {
        this.selectedImage = file
        this.imagePreview = URL.createObjectURL(file)
        this.results = null
      }
    },
    
    clearImage() {
      this.selectedImage = null
      this.imagePreview = null
      this.results = null
    },
    
    async analyzeImage() {
      if (!this.selectedImage) return
      
      this.loading = true
      const formData = new FormData()
      formData.append('image', this.selectedImage)
      
      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          body: formData
        })
        
        this.results = await response.json()
      } catch (error) {
        console.error('Analysis failed:', error)
        alert('Analysis failed. Please try again.')
      } finally {
        this.loading = false
      }
    },
    
    getRiskLevel(prediction) {
      const malignant = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
      return malignant.includes(prediction) ? 'HIGH RISK - Malignant' : 'LOW RISK - Benign'
    },
    
    getRiskClass(prediction) {
      const malignant = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
      return malignant.includes(prediction) ? 'high-risk' : 'low-risk'
    }
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

.header {
  text-align: center;
  padding: 2rem;
  color: white;
}

.main-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  display: grid;
  gap: 2rem;
}

.upload-section {
  background: white;
  border-radius: 15px;
  padding: 2rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.upload-area {
  border: 3px dashed #ddd;
  border-radius: 10px;
  padding: 3rem;
  text-align: center;
  margin-bottom: 1rem;
}

.upload-placeholder {
  cursor: pointer;
}

.image-preview img {
  max-width: 300px;
  max-height: 300px;
  border-radius: 10px;
}

.analyze-btn {
  width: 100%;
  padding: 1rem 2rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 10px;
  font-size: 1.1rem;
  cursor: pointer;
}

.results-section {
  display: grid;
  gap: 2rem;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
}

.prediction-card, .explanation-card {
  background: white;
  border-radius: 15px;
  padding: 2rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.diagnosis {
  font-size: 1.5rem;
  font-weight: bold;
  text-transform: capitalize;
}

.high-risk {
  background: #ffebee;
  color: #c62828;
  padding: 1rem;
  border-radius: 10px;
  text-align: center;
  font-weight: bold;
}

.low-risk {
  background: #e8f5e8;
  color: #2e7d32;
  padding: 1rem;
  border-radius: 10px;
  text-align: center;
  font-weight: bold;
}
</style>