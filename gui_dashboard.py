import streamlit as st
import requests
import pandas as pd
import random
import os
import io
import json
import matplotlib.pyplot as plt
import seaborn as sns

#Configurations
API_URL = "http://127.0.0.1:5000/recommend"
LOG_FILE_PATH = os.path.join(os.getcwd(), 'usage_log.jsonl') 
REPORT_IMAGE_PATH = 'content_gap_report.png'


#Helper Functions

def get_predictions(ticket_text):
    """Sends the ticket text to the Flask API and returns ALL prediction data."""
    if not ticket_text:
        return None, None, None, None 
    
    ticket_id = f"GUI-{random.randint(1000, 9999)}"
    
    payload = {
        "ticket_id": ticket_id,
        "ticket_text": ticket_text
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() 
        
        data = response.json()
        
        return (
            data.get('suggestions', []), 
            data.get('severity_prediction', None),
            data.get('issue_prediction', None), 
            data.get('team_prediction', None)    
        )
        
    except requests.exceptions.ConnectionError:
        st.error("API Connection failed. Please ensure your backend (app.py) is running.")
        return None, None, None, None
    except requests.exceptions.RequestException as e:
        st.error(f"API Request failed: {e}")
        return None, None, None, None


def display_prediction_score(prediction_info):
    """Formats the prediction label and score for consistent display."""
    if not prediction_info:
        return {"label": prediction_info.get('label', "Model Not Loaded"), 
                "score_text": "0.0%"}
    
    label = prediction_info.get('label', "N/A")
    score = prediction_info.get('score', 0.0)
    
    return {"label": label, "score_text": f"{score * 100:.1f}%"}


def extract_text_from_upload(uploaded_file):
    """
    Extracts text content from uploaded files.
    - CSV: Reads each row as a separate ticket.
    - TXT: Assumes tickets are separated by three newline characters (\n\n\n) and processes in bulk.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        try:
            raw_content = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            ticket_blocks = raw_content.split('\n\n\n')
            
            tickets_list = []
            for i, block in enumerate(ticket_blocks):
                cleaned_text = block.replace('\n', ' ').strip()
                
                if cleaned_text:
                    tickets_list.append({
                        'text': cleaned_text, 
                        'source': f"TXT Bulk Ticket {i + 1}"
                    })

            if not tickets_list:
                st.warning("TXT file contains no tickets, or they are not separated by three empty lines.")
                return []
                
            st.success(f"Extracted {len(tickets_list)} tickets from TXT file using the 'three empty lines' separator.")
            return tickets_list
        
        except Exception as e:
            st.error(f"Error processing bulk TXT file: {e}")
            return []
        
    elif file_extension == 'csv':
        try:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
            
            text_columns = ['Description', 'Subject', 'Full_Ticket_Text', 'text', 'ticket_text']
            ticket_column = next((col for col in text_columns if col in df.columns), None)
            
            if ticket_column:
                tickets_list = []
                for index, row in df.iterrows():
                    raw_text = row[ticket_column]
                    if pd.notna(raw_text) and raw_text.strip():
                        cleaned_text = str(raw_text).replace('\n', ' ').strip()
                        
                        tickets_list.append({
                            'text': cleaned_text, 
                            'source': f"CSV Row {index + 1}"
                        })

                st.success(f"Extracted {len(tickets_list)} clean tickets from CSV using column: '{ticket_column}'.")
                return tickets_list
            else:
                st.error("Could not find a suitable text column (e.g., 'Description', 'Subject') in the CSV.")
                return []

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return []
            
    else:
        st.error(f"Unsupported file format: .{file_extension}. Only .txt and .csv are supported.")
        return []


def generate_gap_report():
    """Reads the log file and generates the Content Gap Analysis chart."""
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w') as f:
            pass
    
    try:
        data = []
        with open(LOG_FILE_PATH, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue 

        if not data:
            return False, "No data logged yet."

        df_log = pd.DataFrame(data)
        
        gap_count = df_log['gap_flag'].sum()
        total_count = len(df_log)
        gap_ratio = gap_count / total_count if total_count > 0 else 0
        
        plot_data = pd.DataFrame({
            'Category': ['Content Gap', 'Successful Match'],
            'Count': [gap_count, total_count - gap_count]
        })
        
        plt.figure(figsize=(7, 4))
        sns.barplot(x='Category', y='Count', data=plot_data, palette=['#FF6347', '#3CB371'])
        
        plt.title('Content Gap Analysis (Log History)', fontsize=14)
        plt.ylabel('Number of Tickets', fontsize=12)
        plt.xlabel(f'Total Tickets Analyzed: {total_count} (Gap Ratio: {gap_ratio*100:.1f}%)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(REPORT_IMAGE_PATH)
        plt.close()
        
        return True, REPORT_IMAGE_PATH

    except Exception as e:
        return False, f"Error generating report: {e}"


#Main Application Logic

def main():
    """Builds the Streamlit GUI."""
    st.set_page_config(layout="wide")
    
    st.title("AI-Powered Support Ticket Dashboard")
    st.markdown("Instantly predict ticket **Severity** and **Assigned Team**, and retrieve relevant Knowledge Base articles.")
    st.divider()

    #Input Area
    st.subheader("Simulate Incoming Ticket")
    
    # 1. Input Area (File Upload Section)
    uploaded_file = st.file_uploader(
        "Upload a Ticket File (.txt for bulk, .csv for bulk)",
        type=['txt', 'csv'],
        help="Upload a plain text file (separate tickets with three empty lines) or a CSV with multiple rows of ticket data."
    )

    # 2. Input Area (Manual Text Area Section)
    ticket_text = st.text_area(
        "OR Paste Single Ticket Description Here", 
        height=150,
        placeholder="Type a single customer's issue here..."
    )

    #User Toggle for Reporting
    show_report = st.checkbox(
        "Show Real-Time Content Gap Analysis after processing?", 
        value=False,
        help="Check this box to generate and display the historical gap report after analysis."
    )

    tickets_to_process = []
    
    if uploaded_file is not None:
        tickets_to_process = extract_text_from_upload(uploaded_file)
    elif ticket_text:
        tickets_to_process = [{'text': ticket_text, 'source': 'Manual Input'}]

    if st.button("Get AI Suggestions & Priority", type="primary"):
        if not tickets_to_process:
            st.warning("Please enter or upload a ticket description to begin.")
            return

        all_results = []
        
        #BULK PROCESSING LOOP 
        with st.spinner(f'Processing {len(tickets_to_process)} tickets...'):
            for i, ticket in enumerate(tickets_to_process):
                # Retrieve all predictions
                suggestions, severity_info, issue_info, team_info = get_predictions(ticket['text'])
                
                # Check for API failure
                if suggestions is None: 
                    return
                
                top_suggestion = suggestions[0] if suggestions else None
                
                # Format for the results table
                sev_display = display_prediction_score(severity_info)
                issue_display = display_prediction_score(issue_info) 
                team_display = display_prediction_score(team_info)

                all_results.append({
                    'Source': ticket['source'],
                    'Ticket Snippet': ticket['text'][:80] + '...',
                    'Severity': f"{sev_display['label']} ({sev_display['score_text']})",
                    'Assigned Team': f"{team_display['label']} ({team_display['score_text']})",  
                    'Top Suggestion': top_suggestion['title'] if top_suggestion else "Content Gap Detected",
                    'Similarity Score': f"{top_suggestion['similarity_score'] * 100:.1f}%" if top_suggestion else "0.0%",
                })
        
        #Display Results
        st.subheader(f"Analysis Complete:- {len(all_results)} Tickets Processed")
        
        #Display the full results table
        st.markdown("### Detailed Analysis Table")
        df_results = pd.DataFrame(all_results)
        st.dataframe(df_results, use_container_width=True, hide_index=True)


        #Content Gap Reporting Hub
        if show_report:
            st.divider()
            st.subheader("Real-Time Content Gap Analysis")
            
            success, report_output = generate_gap_report()
            
            if success:
                st.image(report_output, caption=f"Report updated at {pd.Timestamp.now().strftime('%H:%M:%S')}")
                st.markdown(
                    "**What This Means:** Tickets flagged as 'Content Gap' (low/no matching articles) indicate areas where your Knowledge Base needs new documentation. This chart updates with every analysis performed!"
                )
            else:
                st.warning(f"Could not generate report: {report_output}")


if __name__ == "__main__":
    main()