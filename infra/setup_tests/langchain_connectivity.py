import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Get the root directory
root_dir = Path(__file__).parent.parent
# Load environment variables from .env file at root
load_dotenv(dotenv_path=Path(root_dir, ".env"))

def test_langchain_connectivity():
    try:
        from langchain_openai import ChatOpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not found in environment variables")
            return False
            
        # Initialize ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo"
        )
        
        # Create a simple LangChain chain
        template = "Say '{text}' in a creative way."
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run the chain
        result = chain.run(text="LangChain connectivity test successful")
        
        if result:
            print("✅ LangChain connectivity test successful!")
            print(f"Response: {result}")
            return True
        else:
            print("❌ LangChain connectivity test failed: No response")
            return False
            
    except Exception as e:
        print(f"❌ LangChain connectivity test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_langchain_connectivity()
    sys.exit(0 if success else 1)