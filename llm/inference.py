from huggingface_hub import InferenceClient
import nltk
import re 
import requests

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')


client = InferenceClient(api_key="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")


def extract_product_info(text):
    # Initialize result dictionary
    result = {"brand": None, "model": None, "description": None, "price": None}

    # Extract price separately using regex (to avoid confusion with brand name)
    price_match = re.search(r'\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
    if price_match:
        result["price"] = price_match.group().replace("$", "").replace(",", "").strip()
        # Remove the price part from the text to prevent it from being included in the brand/model extraction
        text = text.replace(price_match.group(), "").strip()

    # Tokenize the remaining text and tag parts of speech
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    # Extract brand and model (Proper Nouns + Alphanumeric patterns)
    brand_parts = []
    model_parts = []
    description_parts = []
    
    # First part: Extract brand and model info
    for word, tag in pos_tags:
        if tag == 'NNP' or re.match(r'[A-Za-z0-9-]+', word):
            if len(brand_parts) == 0:  # Assume the first proper noun is the brand
                brand_parts.append(word)
            else:  # Model number tends to follow the brand
                model_parts.append(word)
        else:
            description_parts.append(word)
    
    # Assign brand and model to result dictionary
    if brand_parts:
        result["brand"] = " ".join(brand_parts)
    if model_parts:
        result["model"] = " ".join(model_parts)
    
    # Combine the remaining parts as description
    result["description"] = " ".join(description_parts)

    return result



def extract_info(text):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {"Authorization": "Bearer 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"}
    payload = {"inputs": f"From the given text, extract brand name, model number, description about it, and its average price in today's market. Give me back a python dictionary with keys as brand_name, model_number, desc, price. The text is {text}.",}
    response = requests.post(API_URL, headers=headers, json=payload)
    print('GOOGLEE LLM OUTPUTTTTTTT\n\n',response )
    output = response.json()
    print(output)



def get_name(url, object):
	messages = [
		{
			"role": "user",
			"content": [
				{
					"type": "text",
					"text": f"Is this a {object}?. Can you guess what it is and give me the closest brand it resembles to? or a model number? And give me its average price in today's market in USD. In output, give me its normal name, model name, model number and price. separated by commas. No description is needed." 
				},
				{
					"type": "image_url",
					"image_url": {
						"url": url
					}
				}
			]
		}
	]

	completion = client.chat.completions.create(
		model="meta-llama/Llama-3.2-11B-Vision-Instruct", 
		messages=messages, 
		max_tokens=500
	)


	print(f'\n\nNow output of LLM:\n')
	llm_result = completion.choices[0].message['content']
	print(llm_result)
	print(f'\n\nThat is the output')
     
	result = extract_product_info(llm_result)
	print(f'\n\nResult brand and price:{result}')
    
	# result2 = extract_info(llm_result)
	# print(f'\n\nFrom Google llm:{result2}')

	return result

# url = "https://i.ibb.co/mNYvqDL/crop_39.jpg"
# object="fridge"

# get_name(url, object)