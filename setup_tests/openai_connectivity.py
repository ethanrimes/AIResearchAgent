import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Get the root directory
root_dir = Path(__file__).parent.parent
# Load environment variables from .env file at root
load_dotenv(dotenv_path=Path(root_dir, ".env"))

def test_openai_connectivity():
    try:
        import openai
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not found in environment variables")
            return False
            
        # Initialize client
        client = openai.OpenAI(api_key=api_key)
        
        # Test API connectivity with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, connectivity test successful!'"}
            ]
        )
        
        if response and response.choices:
            print("✅ OpenAI API connectivity test successful!")
            print(f"Response: {response.choices[0].message.content}")
            return True
        else:
            print("❌ OpenAI API connectivity test failed: Unexpected response format")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API connectivity test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openai_connectivity()
    sys.exit(0 if success else 1)