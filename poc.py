import json
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_json_from_response(response_content):
    """
    Extracts JSON objects from the response with <start> and <end> format
    
    Args:
        response_content: The content with <start> and <end> delimiters
    
    Returns:
        A dictionary of parsed data
    """
    if not response_content:
        logger.error("Empty response content provided")
        return {}
    
    try:
        # Match the format in the actual LLM output
        pattern = r'<start>(.*?)<end>'
        matches = re.findall(pattern, response_content, re.DOTALL)
        
        if not matches:
            logger.warning("No <start><end> delimited content found in response")
            return {}
        
        raw_data = {}
        
        for match_idx, match in enumerate(matches):
            try:
                # Trim whitespace that might cause JSON parsing issues
                match = match.strip()
                json_obj = json.loads(match)
                
                # Determine what type of data this is
                if "header" in json_obj:
                    raw_data["header"] = json_obj
                elif "user_stories_order" in json_obj:
                    raw_data["user_stories_order"] = json_obj
                elif any(key.startswith("user_story_") for key in json_obj):
                    # Determine if this is basic info or content
                    try:
                        key_sample = next((key for key in json_obj if key.startswith("user_story_")), None)
                        if not key_sample:
                            logger.warning(f"Found user_story object but no keys start with 'user_story_': {json_obj.keys()}")
                            continue
                            
                        parts = key_sample.split("_")
                        
                        if len(parts) >= 3 and parts[2].isdigit():
                            # Extract story number
                            story_num = parts[2]
                            
                            # Check if this is basic info or content
                            if "content" not in key_sample:
                                # This is basic info (has label, description, etc.)
                                raw_data[f"user_story_{story_num}_basic"] = json_obj
                            else:
                                # This is content info
                                content_parts = key_sample.split("_content_")
                                if len(content_parts) >= 2:
                                    content_type = content_parts[1].split("_")[0]
                                    
                                    # Map content types to numbers
                                    content_map = {
                                        "Acceptance": 1,
                                        "Edge": 2,
                                        "Data": 3
                                    }
                                    
                                    content_num = None
                                    for key_word, num in content_map.items():
                                        if key_word in content_type:
                                            content_num = num
                                            break
                                    
                                    if content_num is not None:
                                        raw_data[f"user_story_{story_num}_content_{content_num}"] = json_obj
                                    else:
                                        logger.warning(f"Unknown content type in key: {key_sample}")
                                else:
                                    logger.warning(f"Malformed content key: {key_sample}")
                        else:
                            logger.warning(f"Invalid user story key format: {key_sample}")
                    except Exception as e:
                        logger.error(f"Error processing user story object: {str(e)}")
                else:
                    logger.warning(f"Unknown JSON object type in match {match_idx}: {list(json_obj.keys())[:5]}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON object (match {match_idx}): {str(e)}")
                # Try to sanitize common JSON issues and retry
                try:
                    # Replace unescaped quotes, fix trailing commas
                    sanitized_match = re.sub(r'(?<!")(".*?)(?<!")', r'\\"', match)
                    sanitized_match = re.sub(r',\s*}', '}', sanitized_match)
                    sanitized_match = re.sub(r',\s*]', ']', sanitized_match)
                    json_obj = json.loads(sanitized_match)
                    logger.info(f"Successfully parsed JSON after sanitization")
                    # Process the sanitized JSON...
                    # (Similar code as above, but kept separate to avoid code duplication)
                except Exception:
                    logger.error(f"Failed to parse JSON even after sanitization attempt")
    except Exception as e:
        logger.error(f"Critical error in extract_json_from_response: {str(e)}")
        return {}
    
    return raw_data

def map_to_structured_format(raw_data):
    """
    Maps the raw data to a structured format, handling any number of user stories dynamically.
    
    Args:
        raw_data: Dictionary containing the parsed user story data
    
    Returns:
        A structured dictionary matching the FinalStructure format
    """
    if not raw_data:
        logger.warning("Empty raw_data provided to map_to_structured_format")
        return {"header": "", "user_stories": {}, "user_stories_order": []}
    
    try:
        # Initialize result structure
        result = {
            "header": "",
            "user_stories": {},
            "user_stories_order": []
        }
        
        # Set the header
        try:
            if "header" in raw_data and isinstance(raw_data["header"], dict):
                result["header"] = raw_data["header"].get("header", "")
            else:
                logger.warning("Header information missing or malformed")
        except Exception as e:
            logger.error(f"Error processing header: {str(e)}")
        
        # Find all user story numbers in the data
        user_story_numbers = set()
        try:
            for key in raw_data:
                if key.startswith("user_story_") and "_basic" in key:
                    parts = key.split("_")
                    if len(parts) >= 3 and parts[2].isdigit():
                        story_num = parts[2]
                        user_story_numbers.add(story_num)
        except Exception as e:
            logger.error(f"Error identifying user story numbers: {str(e)}")
        
        # Process each user story dynamically
        for story_num in user_story_numbers:
            try:
                story_basic_key = f"user_story_{story_num}_basic"
                if story_basic_key not in raw_data:
                    logger.warning(f"Missing basic info for user story {story_num}")
                    continue
                    
                story_basic_data = raw_data[story_basic_key]
                
                # Initialize user story
                story_id = story_num
                result["user_stories"][story_id] = {
                    "label": story_basic_data.get(f"user_story_{story_num}_label", ""),
                    "description": story_basic_data.get(f"user_story_{story_num}_description", ""),
                    "confidence_score": story_basic_data.get(f"user_story_{story_num}_confidence_score", ""),
                    "is_approved": story_basic_data.get(f"user_story_{story_num}_is_approved", False),
                    "content": []
                }
                
                # Add content items - dynamically detect all content types
                content_types = [
                    (1, "Acceptance_Criteria"),
                    (2, "Edge_Cases_Scenarios"),
                    (3, "Data_Input_Considerations")
                ]
                
                for content_id, content_type in content_types:
                    try:
                        content_key = f"user_story_{story_num}_content_{content_id}"
                        if content_key not in raw_data:
                            # This is not an error - it's valid to have missing content types
                            continue
                            
                        content_data = raw_data[content_key]
                        if not isinstance(content_data, dict):
                            logger.warning(f"Content data for {content_key} is not a dictionary")
                            continue
                        
                        text_key = f"user_story_{story_num}_content_{content_type}_text"
                        conf_key = f"user_story_{story_num}_content_{content_type}_confidence_score"
                        appr_key = f"user_story_{story_num}_content_{content_type}_is_approved"
                        
                        text = content_data.get(text_key, "")
                        confidence_score = content_data.get(conf_key, "")
                        is_approved = content_data.get(appr_key, False)
                        
                        result["user_stories"][story_id]["content"].append({
                            "id": str(content_id),
                            "text": text,
                            "confidence_score": confidence_score,
                            "is_approved": is_approved
                        })
                    except Exception as e:
                        logger.error(f"Error processing content {content_id} for story {story_num}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing user story {story_num}: {str(e)}")
        
        # Set the user stories order
        try:
            if "user_stories_order" in raw_data:
                order_obj = raw_data["user_stories_order"]
                if isinstance(order_obj, dict) and "user_stories_order" in order_obj:
                    # Transform the order to simple IDs
                    order_list = []
                    for item in order_obj["user_stories_order"]:
                        if isinstance(item, str):
                            match = re.search(r"user_story_(\d+)", item)
                            if match:
                                order_list.append(match.group(1))
                    # Filter out any empty strings
                    result["user_stories_order"] = [x for x in order_list if x]
                else:
                    logger.warning("user_stories_order object does not contain expected format")
                    result["user_stories_order"] = sorted(list(result["user_stories"].keys()))
            else:
                logger.info("No user_stories_order found, using default order")
                # Fallback: if no order is provided, use the story IDs in sequence
                result["user_stories_order"] = sorted(list(result["user_stories"].keys()))
        except Exception as e:
            logger.error(f"Error processing user stories order: {str(e)}")
            # Ultimate fallback
            result["user_stories_order"] = sorted(list(result["user_stories"].keys()))
        
        # Validate the final result
        if not result["user_stories"]:
            logger.warning("No user stories were successfully parsed")
        
        return result
    except Exception as e:
        logger.error(f"Critical error in map_to_structured_format: {str(e)}")
        return {"header": "", "user_stories": {}, "user_stories_order": []}

def parse_structured_user_story(response_content):
    """
    Processes the LLM response into structured format.
    Replacement for extract_response_from_delimiters_alternative.
    
    Args:
        response_content: The raw LLM response
        
    Returns:
        JSON string of the structured user story data or None if parsing failed
    """
    if not response_content:
        logger.error("Empty response_content provided to parse_structured_user_story")
        return json.dumps({"error": "Empty response content", "header": "", "user_stories": {}, "user_stories_order": []})
    
    try:
        # First extract the raw data from response
        raw_data = extract_json_from_response(response_content)
        
        if not raw_data:
            logger.warning("No valid JSON data extracted from response")
            return json.dumps({"error": "No valid JSON data found", "header": "", "user_stories": {}, "user_stories_order": []})
        
        # Then map it to the structured format
        result = map_to_structured_format(raw_data)
        
        # Convert to JSON string
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Critical error in parse_structured_user_story: {str(e)}")
        # Return a valid JSON with error information rather than None
        return json.dumps({
            "error": f"Failed to parse: {str(e)}",
            "header": "",
            "user_stories": {},
            "user_stories_order": []
        })
    
# Test with the actual LLM output format
def main():
    # Sample response from the actual LLM output
    sample_response = '''data: <start>{"user_story_2_label":"Implement Topic Extraction Agent","user_story_2_description":"As a data analyst or content curator, I want to submit documents or text snippets to a new Topic Extraction Agent so that I can automatically identify the main topics and keywords, enabling quick content summarization and categorization within the AppMod-Agents platform.","user_story_2_confidence_score":"96","user_story_2_is_approved":false}<end>


data: <start>{"user_story_2_content_Acceptance_Criteria_text":"### Acceptance Criteria:\\n#### 1. Agent Creation and Integration\\n- **1.1** A new agent named \\"Topic Extraction Agent\\" shall be created within the `Appmod-Agents` codebase, adhering to the <a href=\\"appmod_citation$4\\">Modular Design</a> principles of existing agents (e.g., Flask-based API service).\\n- **1.2** The agent shall be configured and registered within the `agent_manager` similar to other agents, ensuring it can be discovered and orchestrated.\\n#### 2. API Endpoint for Topic Extraction\\n- **2.1** The Topic Extraction Agent shall expose a dedicated <a href=\\"appmod_citation$5\\">REST API endpoint</a>, e.g., `/utility/topic-extraction/prediction`, accessible via a `POST` request.\\n- **2.2** The endpoint shall accept a JSON payload with a `text` field containing the input string for analysis, as depicted in the whiteboard sketch [output_2.png](image_citation$2).\\n- **2.3** The endpoint shall return a JSON response containing the extracted topics and keywords.\\n#### 3. Topic Extraction Logic\\n- **3.1** The agent shall process the <a href=\\"rca_citation$4\\">Input Text</a> to identify the most relevant topics and keywords.\\n- **3.2** The output shall include a list of identified `topics` (e.g., \\"Artificial Intelligence\\", \\"Machine Learning\\") and a list of `keywords` (e.g., \\"LLM\\", \\"NLP\\", \\"deep learning\\").\\n- **3.3** The agent shall provide an overall confidence `percentage` (an integer between 0 and 100) for the quality of the extraction.\\n- **3.4** The output JSON structure shall conform to `{'topics': <array_of_strings>, 'keywords': <array_of_strings>, 'confidence': <int: 0-100>}`, as depicted in the whiteboard sketch.","user_story_2_content_Acceptance_Criteria_confidence_score":"97","user_story_2_content_Acceptance_Criteria_is_approved":false}<end>


data: <start>{"user_story_2_content_Edge_Cases_Scenarios_text":"### Edge Cases & Scenarios:\\n#### 1. Invalid Input\\n- **1.1** If the input JSON payload is missing the `text` field or the `text` field is not a string, the API should return a `400 Bad Request` error with a descriptive message.\\n- **1.2** If the input text is excessively long (e.g., exceeding a predefined character limit), the agent should handle it gracefully, potentially truncating or returning an error, and log the event.\\n#### 2. No Clear Topics or Keywords\\n- **2.1** For text that is too short, generic, or lacks specific content, the agent should return empty lists for `topics` and `keywords` and a low confidence score.\\n- **2.2** The agent should be robust to noise or irrelevant text segments without letting them dominate the extraction.\\n#### 3. API Communication Failures\\n- **3.1** If the underlying LLM or extraction model fails to respond or encounters an internal error, the API should return a `500 Internal Server Error` and log the detailed error.\\n- **3.2** Network issues during API calls to the topic extraction agent should be handled with appropriate timeouts and retry mechanisms by the calling system.","user_story_2_content_Edge_Cases_Scenarios_confidence_score":"94","user_story_2_content_Edge_Cases_Scenarios_is_approved":false}<end>


data: <start>{"user_story_2_content_Data_Input_Considerations_text":"#### Data and Input Considerations:\\n#### 1. Input Text Format\\n- **1.1** The input `text` field in the JSON payload shall be a UTF-8 encoded string.\\n- **1.2** The agent should be robust to various text inputs, including special characters, emojis, and different languages (though initial focus is English).\\n#### 2. Output Data Structure\\n- **2.1** The API response shall always be a JSON object with three keys: `topics` (array of strings), `keywords` (array of strings), and `confidence` (integer).\\n- **2.2** The `confidence` value must be an integer between 0 and 100, inclusive.\\n#### 3. Agent Configuration\\n- **3.1** The agent's configuration (e.g., LLM model details, API keys, topic model parameters) shall be managed via a <a href=\\"appmod_citation$6\\">Configuration File</a>, following the pattern of `design_pattern_agent/src/configs.py`.\\n- **3.2** The agent should expose `GET /utility/topic-extraction/get_config` and `GET /utility/topic-extraction/get_llm_config` endpoints to retrieve its current configuration, similar to existing agents.","user_story_2_content_Data_Input_Considerations_confidence_score":"96","user_story_2_content_Data_Input_Considerations_is_approved":false}<end>


data: <start>{"user_story_3_label":"Implement Entity Recognition Agent","user_story_3_description":"As a data scientist or knowledge graph developer, I want to submit text to a new Entity Recognition Agent so that I can automatically identify and categorize named entities (e.g., persons, organizations, locations, dates), facilitating information extraction and structuring for various data-driven applications within the AppMod-Agents platform.","user_story_3_confidence_score":"95","user_story_3_is_approved":false}<end>


data: <start>{"user_story_3_content_Acceptance_Criteria_text":"### Acceptance Criteria:\\n#### 1. Agent Creation and Integration\\n- **1.1** A new agent named \\"Entity Recognition Agent\\" shall be created within the `Appmod-Agents` codebase, adhering to the <a href=\\"appmod_citation$7\\">Modular Design</a> principles of existing agents (e.g., Flask-based API service).\\n- **1.2** The agent shall be configured and registered within the `agent_manager` similar to other agents, ensuring it can be discovered and orchestrated.\\n#### 2. API Endpoint for Entity Recognition\\n- **2.1** The Entity Recognition Agent shall expose a dedicated <a href=\\"appmod_citation$8\\">REST API endpoint</a>, e.g., `/utility/entity-recognition/prediction`, accessible via a `POST` request.\\n- **2.2** The endpoint shall accept a JSON payload with a `text` field containing the input string for analysis, as depicted in the whiteboard sketch [output_3.png](image_citation$3).\\n- **2.3** The endpoint shall return a JSON response containing the recognized entities and their classifications.\\n#### 3. Entity Recognition Logic\\n- **3.1** The agent shall process the <a href=\\"rca_citation$5\\">Input Text</a> to identify named entities.\\n- **3.2** The entity classification shall categorize identified entities into predefined types (e.g., \\"PERSON\\", \\"ORGANIZATION\\", \\"LOCATION\\", \\"DATE\\").\\n- **3.3** For each identified entity, the agent shall provide its `text` span, `type`, and a confidence `percentage` (an integer between 0 and 100).\\n- **3.4** The output JSON structure shall conform to `{'entities': [{'text': <string>, 'type': <string>, 'confidence': <int 0-100>}], 'overall_confidence': <int 0-100>}`, as depicted in the whiteboard sketch.","user_story_3_content_Acceptance_Criteria_confidence_score":"96","user_story_3_content_Acceptance_Criteria_is_approved":false}<end>


data: <start>{"user_story_3_content_Edge_Cases_Scenarios_text":"### Edge Cases & Scenarios:\\n#### 1. Invalid Input\\n- **1.1** If the input JSON payload is missing the `text` field or the `text` field is not a string, the API should return a `400 Bad Request` error with a descriptive message.\\n- **1.2** If the input text is excessively long (e.g., exceeding a predefined character limit), the agent should handle it gracefully, potentially truncating or returning an error, and log the event.\\n#### 2. Ambiguous or Unrecognized Entities\\n- **2.1** For text containing entities that are ambiguous or do not clearly fit into predefined categories, the agent should either classify them with low confidence or mark them as \\"UNKNOWN\\" if applicable.\\n- **2.2** For very short or non-descriptive text without clear entities, the agent should return an empty `entities` list and a low `overall_confidence`.\\n#### 3. API Communication Failures\\n- **3.1** If the underlying LLM or entity recognition model fails to respond or encounters an internal error, the API should return a `500 Internal Server Error` and log the detailed error.\\n- **3.2** Network issues during API calls to the entity recognition agent should be handled with appropriate timeouts and retry mechanisms by the calling system.","user_story_3_content_Edge_Cases_Scenarios_confidence_score":"93","user_story_3_content_Edge_Cases_Scenarios_is_approved":false}<end>


data: <start>{"user_story_3_content_Data_Input_Considerations_text":"#### Data and Input Considerations:\\n#### 1. Input Text Format\\n- **1.1** The input `text` field in the JSON payload shall be a UTF-8 encoded string.\\n- **1.2** The agent should be robust to various text inputs, including special characters, emojis, and different languages (though initial focus is English).\\n#### 2. Output Data Structure\\n- **2.1** The API response shall always be a JSON object with two keys: `entities` (array of objects) and `overall_confidence` (integer).\\n- **2.2** Each entity object must contain `text` (string), `type` (string), and `confidence` (integer).\\n- **2.3** All `confidence` values must be integers between 0 and 100, inclusive.\\n#### 3. Agent Configuration\\n- **3.1** The agent's configuration (e.g., LLM model details, API keys, entity types) shall be managed via a <a href=\\"appmod_citation$9\\">Configuration File</a>, following the pattern of `design_pattern_agent/src/configs.py`.\\n- **3.2** The agent should expose `GET /utility/entity-recognition/get_config` and `GET /utility/entity-recognition/get_llm_config` endpoints to retrieve its current configuration, similar to existing agents.","user_story_3_content_Data_Input_Considerations_confidence_score":"95","user_story_3_content_Data_Input_Considerations_is_approved":false}<end>'''
    
    # Test the parser with the sample response
    print("Testing with actual LLM output format...")
    result = parse_structured_user_story(sample_response)
    
    if result:
        print("\nSuccessfully parsed! Result:\n")
        try:
            parsed_result = json.loads(result)
            print(json.dumps(parsed_result, indent=2))
            
        except json.JSONDecodeError:
            print("Error: Result is not valid JSON")
            print(result)
    else:
        print("Failed to parse sample response.")

if __name__ == "__main__":
    main()