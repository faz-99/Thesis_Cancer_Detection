<template>
  <div class="explainability-analysis">
    <div class="upload-section">
      <div class="upload-card">
        <h3>🔬 Explainable AI Analysis</h3>
        <div class="upload-area" @click="triggerFileInput" @drop="handleDrop" @dragover.prevent @dragenter.prevent>
          <div v-if="!selectedFile" class="upload-placeholder">
            <div class="upload-icon">📁</div>
            <p>Click to upload or drag & drop</p>
            <small>PNG, JPG, JPEG files supported</small>
          </div>
          <div v-else class="file-selected">
            <div class="upload-icon">🖼️</div>
            <p>{{ selectedFile.name }}</p>
          </div>
        </div>
        <input ref="fileInput" type="file" @change="handleFileSelect" accept="image/*" style="display: none">
        <button @click="analyzeImage" :disabled="!selectedFile || loading" class="analyze-btn">
          {{ loading ? '🔄 Analyzing...' : '🔍 Analyze Image' }}
        </button>
      </div>
    </div>

    <div v-if="results" class="results-section">
      <!-- Prediction Results -->
      <div class="prediction-card">
        <h3>🎯 Prediction Results</h3>
        <div class="prediction-content">
          <div class="class-result">
            <span class="label">Predicted Class:</span>
            <span class="value">{{ results.prediction.class }}</span>
          </div>
          <div class="confidence-result">
            <span class="label">Confidence:</span>
            <span class="value">{{ (results.prediction.confidence * 100).toFixed(2) }}%</span>
          </div>
        </div>
      </div>

      <!-- Explanation -->
      <div class="explanation-card">
        <h3>🧠 AI Explanation</h3>
        <div class="explanation-text">
          {{ results.explanation.text }}
        </div>
        <div v-if="results.explanation.phrases.length > 0" class="key-findings">
          <h4>Key Findings:</h4>
          <ul>
            <li v-for="phrase in results.explanation.phrases" :key="phrase">
              {{ phrase }}
            </li>
          </ul>
        </div>
      </div>

      <!-- Metrics -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-value">{{ results.explanation.nuclei_count }}</div>
          <div class="metric-label">Nuclei Detected</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ (results.explanation.overlap_ratio * 100).toFixed(1) }}%</div>
          <div class="metric-label">Overlap Ratio</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ (results.explanation.confidence_drop * 100).toFixed(1) }}%</div>
          <div class="metric-label">Confidence Drop</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ results.explanation.faithfulness ? '✅' : '❌' }}</div>
          <div class="metric-label">Faithful</div>
        </div>
      </div>

      <!-- Visualization Images -->
      <div class="images-section">
        <h3>📊 Visual Analysis</h3>
        <div class="images-grid">
          <div v-if="results.images.overlay" class="image-card">
            <img :src="'data:image/png;base64,' + results.images.overlay" alt="Grad-CAM Overlay">
            <h4>Grad-CAM Overlay</h4>
          </div>
          <div v-if="results.images.heatmap" class="image-card">
            <img :src="'data:image/png;base64,' + results.images.heatmap" alt="Activation Heatmap">
            <h4>Activation Heatmap</h4>
          </div>
          <div v-if="results.images.activation_mask" class="image-card">
            <img :src="'data:image/png;base64,' + results.images.activation_mask" alt="Activation Mask">
            <h4>Activation Mask</h4>
          </div>
          <div v-if="results.images.nuclei_mask" class="image-card">
            <img :src="'data:image/png;base64,' + results.images.nuclei_mask" alt="Nuclei Segmentation">
            <h4>Nuclei Segmentation</h4>
          </div>
        </div>
      </div>
    </div>

    <div v-if="error" class="error-message">
      ⚠️ {{ error }}
    </div>
  </div>
</template>

<script>
export default {
  name: 'ExplainabilityAnalysis',
  data() {
    return {
      selectedFile: null,
      loading: false,
      results: null,
      error: null
    }
  },
  methods: {
    triggerFileInput() {
      this.$refs.fileInput.click()
    },
    
    handleFileSelect(event) {
      const file = event.target.files[0]
      if (file) {
        this.selectedFile = file
        this.results = null
        this.error = null
      }
    },
    
    handleDrop(event) {
      event.preventDefault()
      const files = event.dataTransfer.files
      if (files.length > 0) {
        this.selectedFile = files[0]
        this.results = null
        this.error = null
      }
    },
    
    async analyzeImage() {
      if (!this.selectedFile) return
      
      this.loading = true
      this.error = null
      
      const formData = new FormData()
      formData.append('file', this.selectedFile)
      
      try {
        const response = await $fetch('http://localhost:8000/analyze', {
          method: 'POST',
          body: formData
        })
        
        this.results = response
      } catch (error) {
        console.error('Analysis failed:', error)
        this.error = error.data?.detail || 'Analysis failed. Please try again.'
      } finally {
        this.loading = false
      }
    }
  }
}
</script>

<style scoped>
.explainability-analysis {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.upload-card {
  background: white;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
  margin-bottom: 30px;
}

.upload-card h3 {
  margin-bottom: 20px;
  color: #2c3e50;
  font-size: 1.5em;
}

.upload-area {
  border: 2px dashed #bdc3c7;
  border-radius: 8px;
  padding: 40px 20px;
  margin-bottom: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-area:hover {
  border-color: #3498db;
  background-color: #f8f9fa;
}

.upload-icon {
  font-size: 3em;
  margin-bottom: 15px;
}

.analyze-btn {
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  border: none;
  padding: 12px 30px;
  border-radius: 25px;
  cursor: pointer;
  font-size: 1em;
  transition: all 0.3s ease;
}

.analyze-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
}

.analyze-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.results-section {
  display: grid;
  gap: 20px;
}

.prediction-card {
  background: linear-gradient(135deg, #27ae60, #2ecc71);
  color: white;
  padding: 25px;
  border-radius: 12px;
}

.prediction-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-top: 15px;
}

.class-result, .confidence-result {
  text-align: center;
}

.label {
  display: block;
  font-size: 0.9em;
  opacity: 0.9;
  margin-bottom: 5px;
}

.value {
  display: block;
  font-size: 1.5em;
  font-weight: bold;
}

.explanation-card {
  background: white;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.explanation-text {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 15px;
  line-height: 1.6;
}

.key-findings ul {
  list-style: none;
  padding: 0;
}

.key-findings li {
  background: #e8f5e8;
  margin: 8px 0;
  padding: 10px;
  border-radius: 6px;
  border-left: 4px solid #27ae60;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.metric-card {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
}

.metric-value {
  font-size: 2em;
  font-weight: bold;
  color: #3498db;
  margin-bottom: 5px;
}

.metric-label {
  color: #7f8c8d;
  font-size: 0.9em;
}

.images-section {
  background: white;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.image-card {
  border: 1px solid #e9ecef;
  border-radius: 8px;
  overflow: hidden;
  transition: transform 0.3s ease;
}

.image-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.image-card img {
  width: 100%;
  height: 200px;
  object-fit: cover;
}

.image-card h4 {
  padding: 15px;
  margin: 0;
  background: #f8f9fa;
  text-align: center;
  font-size: 0.9em;
  color: #495057;
}

.error-message {
  background: #f8d7da;
  color: #721c24;
  padding: 15px;
  border-radius: 8px;
  margin-top: 20px;
  text-align: center;
}

h3 {
  margin-bottom: 15px;
  color: #2c3e50;
}

h4 {
  margin-bottom: 10px;
  color: #34495e;
}
</style>