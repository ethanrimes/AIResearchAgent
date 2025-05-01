import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Get the root directory
root_dir = Path(__file__).parent.parent
# Load environment variables from .env file at root
load_dotenv(dotenv_path=Path(root_dir, ".env"))

def test_langsmith_connectivity():
    try:
        from langchain_openai import ChatOpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        from langsmith import Client
        
        # Get API keys from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
        langsmith_project = os.getenv("LANGCHAIN_PROJECT", "langsmith-connectivity-test")
        langsmith_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        
        if not openai_api_key:
            print("ERROR: OPENAI_API_KEY not found in environment variables")
            return False
            
        if not langsmith_api_key:
            print("ERROR: LANGCHAIN_API_KEY not found in environment variables")
            return False
            
        # Set environment variables for tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint
        
        # Initialize LangSmith client
        client = Client(
            api_key=langsmith_api_key,
            api_url=langsmith_endpoint
        )
        
        # Test if client can connect
        projects = client.list_projects()
        project_names = [p.name for p in projects]
        print(f"Available projects: {project_names}")
        
        # Test with a simple LangChain chain
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo"
        )
        
        template = "Say 'LangSmith connectivity test successful' in a creative way."
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run the chain (this should automatically be traced)
        result = chain.run(text="Test")
        
        if result:
            print("✅ LangSmith connectivity test successful!")
            print(f"Response: {result}")
            print(f"Check the '{langsmith_project}' project in LangSmith for the trace")
            return True
        else:
            print("❌ LangSmith connectivity test failed: No response")
            return False
            
    except Exception as e:
        print(f"❌ LangSmith connectivity test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_langsmith_connectivity()
    sys.exit(0 if success else 1)