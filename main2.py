import csv
import requests
import json
from ctransformers import AutoModelForCausalLM

# Function to load API key from config.json
def load_api_key():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        return config.get('access_token', None)

# Define base URL and headers for API request
BASE_URL = "https://data.g2.com/api/v1/survey-responses"
HEADERS = {
    "Authorization": f"Token token={load_api_key()}",
    "Content-Type": "application/vnd.api+json"
}

# Load the SLM model
SLM = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-7B-Chat-GGML', model_file='llama-2-7b-chat.ggmlv3.q4_K_S.bin')

# Function to generate feature sets using the language model
# Function to generate feature sets using the language model
def generate_features(comment_feedback):
    generated_text = ''
    for word in SLM(comment_feedback, stream=True):
        generated_text += word
    return generated_text

# Function to process and summarize a survey response
def process_and_summarize_response(response):
    attributes = response.get('attributes', {})
    if not attributes:
        return None

    response_id = response.get('id')
    product_name = attributes.get('product_name')
    if not product_name:
        return None

    title = attributes.get('title')

    comment_answers = attributes.get('comment_answers', {})
    love_value = comment_answers.get('love', {}).get('value', "")
    hate_value = comment_answers.get('hate', {}).get('value', "")

    # Use the language model to generate feature sets
    generated_features = generate_features(love_value + " " + hate_value)

    secondary_answers = attributes.get('secondary_answers', {})
    secondary_details = [{"Meets Requirements": answer.get('value')} for answer in secondary_answers.values()]

    # Create summarized response dictionary
    summarized_response = {
        "ID": response_id,
        "Title": title,
        "Love Value": love_value,
        "Hate Value": hate_value,
        "Generated Features": generated_features,
        "Secondary Answers": secondary_details
    }

    return summarized_response

# Function to fetch survey responses from the API
def fetch_survey_responses(base_url, headers, start_page_number=1):
    page_number = start_page_number

    while True:
        url = f"{base_url}?page%5Bnumber%5D={page_number}&page%5Bsize%5D=100"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            for survey_response in data['data']:
                summarized_response = process_and_summarize_response(survey_response)
                if summarized_response is not None:
                    print(json.dumps(summarized_response, ensure_ascii=False, indent=4))
            if 'next' in data['links']:
                page_number += 1
            else:
                print("No more 'next' link found. Pagination complete.")
                break
        else:
            print(f"Error occurred for page {page_number}: Status Code {response.status_code}")
            break


# Main function
def main():
  
    fetch_survey_responses(BASE_URL, HEADERS)

if __name__ == "__main__":
    main()
