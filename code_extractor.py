"""
Code Extractor Utility with Caching

This module provides utilities for extracting code references from AskMod responses
and fetching the actual code content from the file details API, with caching for efficiency.
"""

import re
import json
import logging
import asyncio
import urllib.parse
import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeExtractor:
    """
    Utility class for extracting and fetching code from citation links in AskMod responses.
    Includes caching to avoid redundant API calls for the same file path.
    """
    
    def __init__(self, 
                 base_url: str = "https://dev-appmod.techo.camp/analyzer/get_file_details",
                 default_params: Optional[Dict[str, str]] = None,
                 max_code_length: int = 500):  # Limit code blocks to 500 lines by default
        """
        Initialize the CodeExtractor.
        
        Args:
            base_url: The base URL for the file details API
            default_params: Default parameters to use when building API URLs
            max_code_length: Maximum number of lines to include in a code block
        """
        self.base_url = base_url
        self.max_code_length = max_code_length
        
        # Default parameters to use if they're missing in the URL
        self.default_params = default_params or {
            "assistant_name": "appmod2a5bd9f2dev",
            "organization_name": "84lumber", 
            "user_id": "68e648e8658ff0e1799590c4",
            "task_id": "88bb18aa-2a7d-42bb-9a66-bf6282ae44a3",
            "repo_url": "https://github.com/Techolution/creative-workspace-backend",
            "chunks_search": "false",
            "technical_details_only": "true"
        }

        # Cache for file contents to avoid redundant API calls
        self.file_cache = {}
        
        # Create directory for storing responses
        os.makedirs("responses", exist_ok=True)
        
        # Track already processed URLs in each session
        self.processed_urls = set()
    
    def _normalize_response(self, response: Any) -> str:
        """
        Normalize the response to a string, handling different response formats.
        
        Args:
            response: The response from AskMod (could be string, dict, etc.)
            
        Returns:
            A normalized string representation of the response
        """
        if response is None:
            return ""
            
        # If it's already a string, return it
        if isinstance(response, str):
            return response
            
        # If it's a dictionary, try to extract the answer
        if isinstance(response, dict):
            # Check for the common response format
            if "result" in response and isinstance(response["result"], dict):
                if "Answer" in response["result"]:
                    return str(response["result"]["Answer"])
                    
            # If we can't find a specific answer field, convert the whole response to JSON
            try:
                return json.dumps(response, indent=2)
            except Exception as e:
                logger.error(f"Error converting response to JSON: {str(e)}")
                return str(response)
        
        # For any other type, convert to string
        return str(response)
        
    def _get_normalized_file_path(self, url: str) -> str:
        """
        Extract and normalize a file path from a URL for use as a cache key.
        
        Args:
            url: The URL or file path
            
        Returns:
            Normalized file path for cache lookup
        """
        # Extract the file path from URL parameters if present
        if "file_path=" in url:
            try:
                # Parse the URL to get the query parameters
                query_params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                if "file_path" in query_params and query_params["file_path"]:
                    # URL decode the file path
                    file_path = urllib.parse.unquote(query_params["file_path"][0])
                    file_path = file_path.split("?")[0]  # Remove any query parameters
                    return file_path
            except Exception as e:
                logger.error(f"Error extracting file path from URL: {str(e)}")
        
        # Check for creative-workspace-backend/ pattern
        match = re.search(r'creative-workspace-backend/([^?&]+)', url)
        if match:
            return f"creative-workspace-backend/{match.group(1)}"
        
        # If it's already a simple file path, normalize it
        if ".py" in url or "/" in url:
            # Remove any validation or technical details flags
            url = url.split("?")[0]
            return url
            
        # Default to the original URL if we can't extract a file path
        return url
    
    def _extract_line_range(self, url: str) -> Optional[Tuple[int, int]]:
        """
        Extract start and end line numbers from a URL if present.
        
        Args:
            url: The URL potentially containing line range parameters
            
        Returns:
            Tuple of (start_line, end_line) or None if no line range is specified
        """
        try:
            # Parse the URL to get the query parameters
            parsed_url = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            # Check if startLine and endLine parameters are present
            if "startLine" in query_params and "endLine" in query_params:
                start_line = int(query_params["startLine"][0])
                end_line = int(query_params["endLine"][0])
                return (start_line, end_line)
                
            return None
        except Exception as e:
            logger.error(f"Error extracting line range from URL: {str(e)}")
            return None
        
    async def extract_urls_from_response(self, response: Union[str, Dict, Any]) -> List[Tuple[str, str]]:
        """
        Extract citation URLs from the response.
        
        Args:
            response: The AskMod response (could be string, dict, etc.)
            
        Returns:
            List of tuples containing (display_text, url)
        """
        # Normalize the response to a string
        response_text = self._normalize_response(response)
        
        if not response_text:
            return []
            
        # Pattern to match markdown links: [text](url)
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, response_text)
        
        # Reset the processed URLs set for this new response
        self.processed_urls = set()
        
        # Extract URLs from matches
        url_tuples = []
        for display_text, url in matches:
            # Filter for code file references
            if any([
                "creative-workspace-backend" in url,
                "file_path=" in url, 
                url.endswith('.py'), 
                url.endswith('.js'), 
                url.endswith('.ts'),
                url.endswith('&validation=True'), 
                url.endswith('&validation=False')
            ]):
                # Normalize the URL for duplicate detection
                norm_path = self._get_normalized_file_path(url)
                
                logger.info(f"Found file reference: `{display_text}` -> {norm_path}")
                url_tuples.append((display_text, url))
        
        return url_tuples
    
    def _encode_file_path(self, file_path: str) -> str:
        """
        Properly encode a file path for use in a URL, maintaining forward slashes.
        
        Args:
            file_path: The file path to encode
            
        Returns:
            URL-encoded file path
        """
        # Split the path by slashes
        parts = file_path.split('/')
        # Encode each part individually
        encoded_parts = [urllib.parse.quote(part, safe='') for part in parts]
        # Join with URL-encoded slashes
        return '/'.join(encoded_parts)
    
    def _build_api_url(self, file_path: str) -> str:
        """
        Build the complete API URL with all required parameters.
        
        Args:
            file_path: The path to the file
            
        Returns:
            A properly formatted API URL
        """
        # URL encode the file path properly
        encoded_file_path = urllib.parse.quote(file_path, safe='')
        
        # URL encode the repo URL
        encoded_repo_url = urllib.parse.quote(self.default_params["repo_url"], safe='')
        
        # Build the URL with all required parameters
        return (f"{self.base_url}?file_path={encoded_file_path}"
                f"&assistant_name={self.default_params['assistant_name']}"
                f"&organization_name={self.default_params['organization_name']}"
                f"&user_id={self.default_params['user_id']}"
                f"&task_id={self.default_params['task_id']}"
                f"&repo_url={encoded_repo_url}"
                f"&chunks_search={self.default_params['chunks_search']}"
                f"&technical_details_only={self.default_params['technical_details_only']}")
    
    def _truncate_code(self, code: str) -> str:
        """
        Truncate code to a manageable size if it's too large.
        
        Args:
            code: The code content to truncate
            
        Returns:
            Truncated code with a note
        """
        if not code:
            return ""
            
        lines = code.split('\n')
        if len(lines) <= self.max_code_length:
            return code
            
        # Keep the first part and add an indicator that it was truncated
        truncated_lines = lines[:self.max_code_length]
        truncated_lines.append("\n# ... Code truncated (too large to display in full) ...")
        return '\n'.join(truncated_lines)
    
    async def fetch_code_content(self, url_tuple: Tuple[str, str], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Fetch code content from the file details API, with caching for efficiency.
        
        Args:
            url_tuple: Tuple of (display_text, url) where url points to a code reference
            headers: Optional headers for the API request
            
        Returns:
            Dictionary containing the fetched code content and metadata
        """
        display_text, url = url_tuple
        
        try:
            # Extract the normalized file path for cache lookup
            norm_path = self._get_normalized_file_path(url)
            
            # Extract line range if present in the URL
            line_range = self._extract_line_range(url)
            
            # Create a cache key that includes the line range if present
            cache_key = f"{norm_path}#{line_range}" if line_range else norm_path
            
            # Check if this file path and line range combination is already in the cache
            if cache_key in self.file_cache:
                logger.info(f"Cache hit for: {cache_key}")
                cached_result = self.file_cache[cache_key].copy()
                # Update with the current display text and URL
                cached_result["display_text"] = display_text
                cached_result["url"] = url
                return cached_result
            
            # Check if we have the full file cached but not the specific line range
            if norm_path in self.file_cache and line_range:
                logger.info(f"Cache hit for full file, applying line range: {line_range}")
                cached_result = self.file_cache[norm_path].copy()
                
                # Extract the requested line range from the cached content
                if cached_result.get("content"):
                    # Split the content into lines
                    lines = cached_result["content"].split('\n')
                    start_line, end_line = line_range
                    
                    # Adjust for zero-based indexing if needed
                    if start_line > 0:  # If line numbers are 1-based
                        start_idx = start_line - 1  # Convert to 0-based
                    else:
                        start_idx = start_line
                        
                    # Handle end_line 
                    if end_line < 0:  # Negative end line means all lines from start_line
                        extracted_content = '\n'.join(lines[start_idx:])
                    else:
                        # End line is inclusive in most code editors, so we add 1 for the slice
                        end_idx = end_line if end_line == 0 else end_line  # For 1-based indexing
                        extracted_content = '\n'.join(lines[start_idx:end_idx])
                    
                    # Create a new result with the extracted content
                    result = cached_result.copy()
                    result["display_text"] = display_text
                    result["url"] = url
                    result["content"] = extracted_content
                    result["line_range"] = line_range
                    
                    # Cache the extracted content with the line range
                    self.file_cache[cache_key] = result.copy()
                    
                    return result
            
            # Build the API URL
            api_url = self._build_api_url(norm_path)
            logger.info(f"Fetching code from: {api_url}")
            
            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error fetching code: {response.status} - {error_text}")
                        return {
                            "display_text": display_text,
                            "url": url,
                            "file_path": norm_path,
                            "error": f"Failed to fetch code: Status {response.status}",
                            "content": None
                        }
                    
                    # Parse the response
                    response_text = await response.text()
                    data = json.loads(response_text)
                    
                    # Extract content from the response
                    content = None
                    if isinstance(data, dict):
                        if "content" in data:
                            content = data["content"]
                        elif "0" in data and "content" in data["0"]:
                            content = data["0"]["content"]
                    elif isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], dict) and "content" in data[0]:
                            content = data[0]["content"]
                    
                    # Create the response object for the full file
                    full_result = {
                        "display_text": display_text,
                        "url": url,
                        "file_path": norm_path,
                        "content": content,
                        "summary": data.get("summary", ""),
                    }
                    
                    # Cache the full file content
                    self.file_cache[norm_path] = full_result.copy()
                    
                    # If a line range is requested, extract that portion
                    if line_range and content:
                        # Split the content into lines
                        lines = content.split('\n')
                        start_line, end_line = line_range
                        
                        # Adjust for zero-based indexing
                        start_idx = max(0, start_line - 1)  # Convert to 0-based, ensure non-negative
                        
                        # Handle end_line (adjust for 1-based indexing if needed)
                        if end_line < 0:  # Negative end line means all lines from start_line
                            extracted_content = '\n'.join(lines[start_idx:])
                        else:
                            # End is inclusive in most editors, so we need to add 1 for the slice
                            end_idx = min(len(lines), end_line)  # Ensure within bounds
                            extracted_content = '\n'.join(lines[start_idx:end_idx])
                        
                        # Add line range info to the result
                        range_result = full_result.copy()
                        range_result["content"] = extracted_content
                        range_result["line_range"] = line_range
                        
                        # Cache the extracted content with the line range
                        self.file_cache[cache_key] = range_result.copy()
                        
                        return range_result
                    
                    # Otherwise return the full file content
                    return full_result
                    
        except Exception as e:
            logger.error(f"Exception fetching code: {str(e)}")
            return {
                "display_text": display_text,
                "url": url,
                "file_path": norm_path if 'norm_path' in locals() else None,
                "error": f"Exception: {str(e)}",
                "content": None
            }
    
    def _generate_safe_filename(self, text: Any) -> str:
        """
        Generate a safe filename from text using a hash function.
        
        Args:
            text: The text to hash (could be string, dict, etc.)
            
        Returns:
            A safe filename
        """
        # Handle None or non-string types
        if text is None:
            return "none"
        
        # If it's not a string, convert it to a string
        if not isinstance(text, str):
            try:
                text = json.dumps(text)
            except Exception:
                text = str(text)
                
        # Generate a hash of the text
        try:
            hash_obj = hashlib.md5(text.encode('utf-8'))
            return hash_obj.hexdigest()[:8]
        except Exception as e:
            logger.error(f"Error generating hash for filename: {str(e)}")
            # Fallback to a timestamp-based filename
            import time
            return f"file_{int(time.time())}"
    
    def _determine_language(self, file_path: str, display_text: str) -> str:
        """
        Determine the programming language based on file extension or content clues.
        
        Args:
            file_path: Path to the file
            display_text: Display text of the link
            
        Returns:
            The programming language
        """
        # Default to Python since most of the codebase seems to be Python
        language = "python"
        
        # Check file extension in the file path
        if file_path:
            if file_path.endswith('.py'):
                language = "python"
            elif file_path.endswith(('.js', '.jsx')):
                language = "javascript"
            elif file_path.endswith(('.ts', '.tsx')):
                language = "typescript"
            elif file_path.endswith('.java'):
                language = "java"
            elif file_path.endswith('.html'):
                language = "html"
            elif file_path.endswith('.css'):
                language = "css"
        
        # Check for clues in the display text
        if display_text:
            if display_text.endswith('.py') or 'python' in display_text.lower():
                language = "python"
            elif display_text.endswith(('.js', '.jsx')) or 'javascript' in display_text.lower():
                language = "javascript"
            elif display_text.endswith(('.ts', '.tsx')) or 'typescript' in display_text.lower():
                language = "typescript"
        
        return language
    
    async def enhance_response_with_code(self, response: Any, 
                                        headers: Optional[Dict[str, str]] = None,
                                        max_concurrent_requests: int = 5,
                                        save_to_file: bool = True) -> str:
        """
        Enhance the response by fetching and embedding code for each citation.
        Uses caching to avoid redundant API calls for the same file path.
        
        Args:
            response: The original AskMod response (could be string, dict, etc.)
            headers: Optional headers for API requests
            max_concurrent_requests: Maximum number of concurrent API requests
            save_to_file: Whether to save the enhanced response to a file
            
        Returns:
            Enhanced response text with embedded code
        """
        # Normalize the response to a string
        response_text = self._normalize_response(response)
        
        # If response text is empty, return it as is
        if not response_text:
            logger.warning("Response text is empty. Cannot enhance.")
            return ""
            
        # Extract citation URLs
        url_tuples = await self.extract_urls_from_response(response_text)
        
        if not url_tuples:
            logger.info("No code references found in response")
            return response_text
            
        logger.info(f"Found {len(url_tuples)} code references to enhance")
        
        # Group URL tuples by normalized file path to minimize redundant requests
        file_path_groups = {}
        for display_text, url in url_tuples:
            norm_path = self._get_normalized_file_path(url)
            if norm_path not in file_path_groups:
                file_path_groups[norm_path] = []
            file_path_groups[norm_path].append((display_text, url))
        
        # Fetch code content for each unique file path in batches
        all_results = []
        unique_file_paths = [(group[0][0], group[0][1]) for group in file_path_groups.values()]
        
        for i in range(0, len(unique_file_paths), max_concurrent_requests):
            batch = unique_file_paths[i:i+max_concurrent_requests]
            batch_results = await asyncio.gather(*[
                self.fetch_code_content(url_tuple, headers) 
                for url_tuple in batch
            ])
            all_results.extend(batch_results)
        
        # Create a mapping of display text to code content
        code_lookup = {}
        for result in all_results:
            # Find all display texts for this file path
            norm_path = result.get("file_path", "")
            for display_text, url in file_path_groups.get(norm_path, []):
                # Create a new entry for each display text
                code_lookup[display_text] = {
                    "display_text": display_text,
                    "url": url,
                    "content": result.get("content"),
                    "file_path": norm_path
                }
        
        # Replace citation links with links + code blocks in the response
        enhanced_response = response_text
        for display_text, url in url_tuples:
            if display_text in code_lookup and code_lookup[display_text]["content"]:
                # Determine the language for syntax highlighting
                language = self._determine_language(
                    code_lookup[display_text].get("file_path", ""),
                    display_text
                )
                
                # Create a code block with the content
                code_block = f"\n\n```{language}\n{code_lookup[display_text]['content']}\n```\n"
                
                # Replace just the link with the link + code block
                link_pattern = re.escape(f"[{display_text}]({url})")
                replacement = f"[{display_text}]({url}){code_block}"
                enhanced_response = re.sub(link_pattern, replacement, enhanced_response)
        
        # Save the enhanced response to a file if requested
        if save_to_file:
            try:
                # Create responses directory if it doesn't exist
                os.makedirs("responses", exist_ok=True)
                
                # Generate a unique filename based on response content
                safe_filename = self._generate_safe_filename(response_text)
                response_file = os.path.join("responses", f"enhanced_response_{safe_filename}.txt")
                
                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(enhanced_response)
                logger.info(f"Saved enhanced response to {response_file}")
            except Exception as e:
                logger.error(f"Error saving enhanced response: {str(e)}")
        
        return enhanced_response