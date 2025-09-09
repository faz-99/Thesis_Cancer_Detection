<template>
  <div class="explainability-page">
    <div class="page-header">
      <h1>🔬 Breast Cancer Explainability</h1>
      <p>AI-Powered Histopathology Analysis with Human-Readable Explanations</p>
    </div>
    
    <ExplainabilityAnalysis />
  </div>
</template>

<script>
export default {
  name: 'ExplainabilityPage',
  head() {
    return {
      title: 'Breast Cancer Explainability - AI Analysis',
      meta: [
        {
          hid: 'description',
          name: 'description',
          content: 'Explainable AI analysis for breast cancer histopathology images with Grad-CAM visualization and human-readable explanations.'
        }
      ]
    }
  }
}
</script>

<style scoped>
.explainability-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px 0;
}

.page-header {
  text-align: center;
  color: white;
  margin-bottom: 30px;
  padding: 40px 20px;
}

.page-header h1 {
  font-size: 3em;
  margin-bottom: 15px;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.page-header p {
  font-size: 1.2em;
  opacity: 0.9;
  max-width: 600px;
  margin: 0 auto;
}
</style>