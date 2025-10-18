from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os, time, json
import csv 
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification
from data_connector import preprocess_text 
from slack_sdk import WebClient 
import torch 
import torch.nn.functional as F

app = Flask(__name__)

#Configurations
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' 
CLEANED_DATA_FILE = 'knowledge_base_with_embeddings.csv'
EMBEDDINGS_FILE = 'kb_embeddings.npy'
SLACK_CHANNEL_ID = "C0XXXXXXX" 

#Log file path
LOG_FILE_PATH = os.path.join(os.getcwd(), 'usage_log.jsonl')

#Fine-Tuned Model Paths
FINE_TUNED_ISSUE_DIR = './fine_tuned_model_issue'
FINE_TUNED_TEAM_DIR = './fine_tuned_model_team'



KB_ARTICLES = None
KB_EMBEDDINGS = None
RECOMMENDATION_MODEL = None
RECOMMENDATION_TOKENIZER = None

# Classification models
SEVERITY_CLASSIFIER = None
ISSUE_MODEL = None      
ISSUE_TOKENIZER = None
TEAM_MODEL = None       
TEAM_TOKENIZER = None

SEVERITY_LABELS = ['Critical', 'High', 'Medium', 'Low']


def load_resources():
    """Loads all KB data, embeddings, and the classification models."""
    global KB_ARTICLES, KB_EMBEDDINGS, RECOMMENDATION_MODEL, RECOMMENDATION_TOKENIZER
    global SEVERITY_CLASSIFIER, ISSUE_MODEL, ISSUE_TOKENIZER, TEAM_MODEL, TEAM_TOKENIZER
    
    try:
        if not os.path.exists(CLEANED_DATA_FILE) or not os.path.exists(EMBEDDINGS_FILE):
             print("ERROR: Missing data files. Did you run categorizer.py?")
             return False

        # Reading with '|' separator as categorizer.py saves it that way
        KB_ARTICLES = pd.read_csv(
            CLEANED_DATA_FILE, 
            sep='|', engine='python', on_bad_lines='warn', quoting=csv.QUOTE_NONE
        )
        # Ensure KB embeddings are loaded as float64
        loaded_embeddings = np.load(EMBEDDINGS_FILE).astype(np.float64) 
        if loaded_embeddings.ndim != 2:
            print(f"FATAL ERROR: KB_EMBEDDINGS loaded with {loaded_embeddings.ndim} dimensions. Expected 2D array.")
            return False
        
        KB_EMBEDDINGS = loaded_embeddings

        # 1. Load Recommendation Model (Sentence Transformer)
        RECOMMENDATION_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        RECOMMENDATION_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

        # 2. Load Severity Classifier (Zero-Shot - unchanged)
        zero_shot_model = "facebook/bart-large-mnli"
        SEVERITY_CLASSIFIER = pipeline("zero-shot-classification", model=zero_shot_model)

        # 3. Load Fine-Tuned Issue Classifier 
        if os.path.exists(FINE_TUNED_ISSUE_DIR):
            ISSUE_TOKENIZER = AutoTokenizer.from_pretrained(FINE_TUNED_ISSUE_DIR)
            ISSUE_MODEL = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_ISSUE_DIR)
            print("Successfully loaded fine-tuned ISSUE model.")
        else:
            print(f"WARNING: Issue model not found at {FINE_TUNED_ISSUE_DIR}. Zero-shot will be skipped.")

        # 4. Load Fine-Tuned Team Classifier 
        if os.path.exists(FINE_TUNED_TEAM_DIR):
            TEAM_TOKENIZER = AutoTokenizer.from_pretrained(FINE_TUNED_TEAM_DIR)
            TEAM_MODEL = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_TEAM_DIR)
            print("Successfully loaded fine-tuned TEAM model.")
        else:
            print(f"WARNING: Team model not found at {FINE_TUNED_TEAM_DIR}. Zero-shot will be skipped.")
            

        print(f"Flask: Loaded {len(KB_ARTICLES)} KB articles, embeddings, and classification models.")
        return True
    except Exception as e:
        print(f"ERROR: Could not load resources. Error: {e}")
        return False

def get_query_embedding(text):
    """Generates an embedding for the incoming ticket text."""
    inputs = RECOMMENDATION_TOKENIZER(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = RECOMMENDATION_MODEL(**inputs)
        
    #Ensure the query vector is float64 and definitively 1D
    return outputs.last_hidden_state.mean(dim=1).numpy().flatten().astype(np.float64) 


slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN")) 

def log_recommendation_usage(ticket_text, ticket_id, suggestions, severity_prediction=None, issue_prediction=None, team_prediction=None):
    """Logs the interaction for Content Gap and Performance Analysis."""
    log_entry = {
        "timestamp": time.time(),
        "ticket_id": ticket_id,
        "ticket_content": ticket_text,
        "suggestions": [s['article_id'] for s in suggestions],
        "severity": severity_prediction,
        "issue": issue_prediction,      
        "team": team_prediction,        
        "gap_flag": 1 if not suggestions or suggestions[0]['similarity_score'] < 0.6 else 0
    }
    
    try:
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"FATAL LOGGING ERROR: Could not write to log file. Error: {e}")
    

def classify_severity(text):
    """Zero-Shot Classification for Severity (Unchanged)."""
    if SEVERITY_CLASSIFIER is None:
        return {"label": "Error", "score": 0.0}

    #Uses the classifier to predict the label and score
    result = SEVERITY_CLASSIFIER(text, candidate_labels=SEVERITY_LABELS)
    
    #Return the top prediction
    return {
        "label": result['labels'][0],
        "score": float(round(result['scores'][0], 4))
    }


def classify_fine_tuned(text, model, tokenizer):
    """Inference logic for fine-tuned sequence classification models."""
    if model is None or tokenizer is None:
        return {"label": "Model Not Loaded", "score": 0.0}

    # 1. Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # 2. Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 3. Apply Softmax to get probabilities
    probs = F.softmax(outputs.logits, dim=1)
    
    # 4. Get the index and score of the highest probability
    score, index = torch.max(probs, dim=1)
    
    # 5. Map the index back to the label string
    predicted_label = model.config.id2label[index.item()]
    
    # 6. Return the result
    return {
        "label": predicted_label,
        "score": float(round(score.item(), 4))
    }

def send_slack_recommendation(channel_id, ticket_id, suggestions, severity_label, issue_label, team_label):
    """Sends the top suggestion and classification to the specified Slack channel."""
    
    emoji = {'Critical': '🚨', 'High': '🔥', 'Medium': '🟡', 'Low': '🟢'}.get(severity_label, '❓')
    
    message = (
        f"{emoji} **TICKET TRIAGE ALERT!** {emoji}\n"
        f"**Ticket ID:** {ticket_id}\n"
        f"**Predicted Severity:** **{severity_label}**\n"
        f"**Predicted Issue Type:** {issue_label}\n"
        f"**Predicted Assigned Team:** {team_label}\n"
        "---"
    )
    
    if suggestions:
        top_suggestion = suggestions[0]
        message += (
            f"\n **AI Suggestion** (Score: {top_suggestion['similarity_score']})\n"
            f"**Top Article:** {top_suggestion['title']}\n"
            f"Preview: {top_suggestion['summary']}"
        )
    else:
        message += "\n **Content Gap Detected:** No strong article suggestion found."

    try:
        slack_client.chat_postMessage(channel=channel_id, text=message)
    except Exception as e:
        print(f"WARNING: Slack message failed. Error: {e}")


@app.route('/recommend', methods=['POST'])
def recommend_article():
    """Analyzes text and returns top articles and classification results."""
    data = request.json
    ticket_text = data.get('ticket_text', '')
    ticket_id = data.get('ticket_id', 'N/A')

    # 1. Classification Predictions (Severity, Issue Type, Team)
    severity_info = classify_severity(ticket_text) 
    issue_info = classify_fine_tuned(ticket_text, ISSUE_MODEL, ISSUE_TOKENIZER) 
    team_info = classify_fine_tuned(ticket_text, TEAM_MODEL, TEAM_TOKENIZER)    

    # Prepare classification response
    severity_prediction = {"label": severity_info['label'], "score": severity_info['score']}
    issue_prediction = {"label": issue_info['label'], "score": issue_info['score']}
    team_prediction = {"label": team_info['label'], "score": team_info['score']}
    
    suggestions = []

    if ticket_text and KB_EMBEDDINGS is not None:
        # 2. Recommendation Logic
        cleaned_query = preprocess_text(ticket_text)
        query_embedding = get_query_embedding(cleaned_query)
        
        # Calculate similarity as stable dot product
        similarities = np.dot(KB_EMBEDDINGS, query_embedding)
        
        # Get the indices of the top 3 highest similarities
        top_indices = np.argsort(similarities)[::-1][:3] 
        
        for i in top_indices:
            similarity = similarities[i] # Similarity is the direct score
            # Content gap threshold is set at 0.5 (can be adjusted later based on performance)
            if similarity > 0.5: 
                article = KB_ARTICLES.iloc[i]
                suggestions.append({
                    "article_id": str(article.get('Ticket ID', i)),
                    "title": article.get('Subject', 'No Subject'),
                    "similarity_score": float(round(similarity, 4)), 
                    "summary": article.get('Full_Ticket_Text', 'N/A')[:100] + '...'
                })

    # 3. Log Interaction 
    log_recommendation_usage(
        ticket_text, 
        ticket_id, 
        suggestions, 
        severity_prediction=severity_prediction,
        issue_prediction=issue_prediction,
        team_prediction=team_prediction
    )
    
    # 4. Integrate Slack Notification 
    send_slack_recommendation(
        SLACK_CHANNEL_ID, 
        ticket_id, 
        suggestions, 
        severity_prediction['label'],
        issue_prediction['label'],
        team_prediction['label']
    )

    # 5. Return combined response
    return jsonify({
        "suggestions": suggestions,
        "severity_prediction": severity_prediction,
        "issue_prediction": issue_prediction,
        "team_prediction": team_prediction
    })


if __name__ == '__main__':
    if load_resources():
        app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)