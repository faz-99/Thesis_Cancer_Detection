import torch
import torch.nn.functional as F
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

class RAGExplainer:
    def __init__(self, model_name='microsoft/DialoGPT-medium'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Medical knowledge base
        self.knowledge_base = self._create_knowledge_base()
        self.index = self._build_faiss_index()
        
    def _create_knowledge_base(self):
        """Create medical knowledge base for retrieval"""
        knowledge = {
            'adenosis': [
                'Adenosis is a benign breast condition characterized by enlarged lobules',
                'Shows increased glandular tissue with preserved architecture',
                'Often presents with dense breast tissue on imaging',
                'No malignant potential but may mimic carcinoma'
            ],
            'fibroadenoma': [
                'Fibroadenoma is the most common benign breast tumor in young women',
                'Characterized by well-defined borders and uniform cellular pattern',
                'Contains both epithelial and stromal components',
                'Usually mobile and painless on physical examination'
            ],
            'ductal_carcinoma': [
                'Invasive ductal carcinoma is the most common type of breast cancer',
                'Arises from milk ducts and invades surrounding tissue',
                'Shows irregular borders and heterogeneous cellular morphology',
                'Requires immediate treatment and staging evaluation'
            ],
            'lobular_carcinoma': [
                'Invasive lobular carcinoma arises from breast lobules',
                'Characterized by single-file growth pattern',
                'Often difficult to detect on imaging due to growth pattern',
                'May be multifocal or bilateral'
            ]
        }
        
        # Flatten knowledge base
        all_facts = []
        fact_labels = []
        for label, facts in knowledge.items():
            all_facts.extend(facts)
            fact_labels.extend([label] * len(facts))
        
        return {'facts': all_facts, 'labels': fact_labels}
    
    def _build_faiss_index(self):
        """Build FAISS index for fast retrieval"""
        # Simple TF-IDF based embeddings (in practice, use sentence transformers)
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=512)
        embeddings = vectorizer.fit_transform(self.knowledge_base['facts']).toarray()
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        self.vectorizer = vectorizer
        return index
    
    def retrieve_relevant_facts(self, query, k=3):
        """Retrieve relevant medical facts"""
        query_embedding = self.vectorizer.transform([query]).toarray().astype('float32')
        scores, indices = self.index.search(query_embedding, k)
        
        relevant_facts = []
        for idx in indices[0]:
            relevant_facts.append({
                'fact': self.knowledge_base['facts'][idx],
                'label': self.knowledge_base['labels'][idx]
            })
        
        return relevant_facts
    
    def generate_explanation(self, prediction, confidence, image_features=None):
        """Generate textual explanation for prediction"""
        class_names = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma',
                      'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
        
        predicted_class = class_names[prediction]
        
        # Retrieve relevant facts
        query = f"breast cancer {predicted_class} characteristics diagnosis"
        relevant_facts = self.retrieve_relevant_facts(query)
        
        # Create context for generation
        context = f"Diagnosis: {predicted_class} (confidence: {confidence:.2f})\n"
        context += "Medical knowledge:\n"
        for fact in relevant_facts:
            context += f"- {fact['fact']}\n"
        
        # Generate explanation
        prompt = f"{context}\nExplanation: This image shows signs of {predicted_class} because"
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = explanation.replace(prompt, "").strip()
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'explanation': explanation,
            'relevant_facts': [fact['fact'] for fact in relevant_facts]
        }

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

def explain_prediction(model, image, rag_explainer, device='cpu'):
    """Generate comprehensive explanation for a prediction"""
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    # Generate GradCAM
    if hasattr(model, 'backbone'):
        target_layer = model.backbone.features[-1]
    else:
        target_layer = model.features[-1]
    
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(image.unsqueeze(0).to(device), prediction.item())
    
    # Generate textual explanation
    explanation = rag_explainer.generate_explanation(
        prediction.item(), 
        confidence.item()
    )
    
    return {
        'visual_explanation': cam,
        'textual_explanation': explanation,
        'prediction': prediction.item(),
        'confidence': confidence.item()
    }