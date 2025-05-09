import os
import re
from dotenv import load_dotenv
from jinja2 import Template
import pinecone
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Pinecone as LC_Pinecone

from src.parser import Parser

class Model:
    def __init__(
        self,
        docs,
        embedding_model: str = "mxbai-embed-large",
        test_model: ChatOllama = ChatOllama(model="llama3.1:8b"),
        use_rag: bool = True
    ):
        load_dotenv()

        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        )
        index_name = os.getenv("PINECONE_INDEX_NAME", "java-tests")

        self.embedding_model = OllamaEmbeddings(model=embedding_model)

        self.vectorstore = LC_Pinecone.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            index_name=index_name,
            namespace="java_tests"
        )

        self.test_model = test_model
        self.use_rag = use_rag

        # TODO: Improve prompt  template
        self.PROMPT_TEMPLATE = """
                                Given this example JUnit 5 test method:
                                ```
                                {reference_test_code}
                                ```
                                which tests the Java method:
                                ```
                                {reference_method_code}
                                ```
                                
                                Generate a **single** JUnit 5 `@Test`-annotated method that tests this Java method:
                                ```
                                {method_code}
                                ```
                                inside its class:
                                ```
                                {class_code}
                                ```
                                
                                Requirements:
                                - Use JUnit 5 (`org.junit.jupiter.api.*`).
                                - Use `assertEquals`, `assertThrows`, etc.
                                - Do **not** include imports or class declarations—only the method.
                                - Output exactly one valid Java method.
        """

        # TODO: Improve prompt  template
        self.TEST_CLASS_TEMPLATE = """  import org.junit.jupiter.api.Test;
                                        import static org.junit.jupiter.api.Assertions.*;
                                        
                                        public class {{ class_name }}Test {
                                        
                                            {% for test in tests %}
                                            {{ test }}
                                        
                                            {% endfor %}
                                        }
                                    """

    def generate_test_methods(self, java_path: str, k: int = 2) -> str:
        parser = Parser(java_path)
        methods, names = parser.get_methods()

        with open(java_path, "r") as f:
            class_code = f.read()
        class_name = os.path.splitext(os.path.basename(java_path))[0]

        tests = []
        for method_code, method_name in zip(methods, names):
            if not re.search(r"\bpublic\b", method_code):
                continue

            if self.use_rag:
                example_docs = self.vectorstore.similarity_search(method_code, k=k)
                for doc in example_docs:
                    ref_test   = doc.metadata.get("tests", "")
                    ref_method = doc.metadata.get("class", "")
                    prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
                    chain  = prompt | self.test_model | StrOutputParser()
                    response = chain.invoke({
                        "reference_test_code": ref_test,
                        "reference_method_code": ref_method,
                        "method_code": method_code,
                        "class_code": class_code
                    })
                    m = re.search(r"```(?:java)?\s*(public\s+.*?})", response, re.DOTALL)
                    tests.append(m.group(1).strip() if m else response.strip())
            else:
                prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
                chain  = prompt | self.test_model | StrOutputParser()
                response = chain.invoke({
                    "reference_test_code": "",
                    "reference_method_code": "",
                    "method_code": method_code,
                    "class_code": class_code
                })
                m = re.search(r"```(?:java)?\s*(public\s+.*?})", response, re.DOTALL)
                tests.append(m.group(1).strip() if m else response.strip())

        template = Template(self.TEST_CLASS_TEMPLATE)
        test_class_code = template.render(class_name=class_name, tests=tests)
        return test_class_code

    @staticmethod
    def save_test_class(output_dir: str, test_class_code: str, class_name: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{class_name}Test.java")
        with open(path, "w") as f:
            f.write(test_class_code)
        print(f"Generated {path}")
