from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.llms import HuggingFacePipeline
from torch import cuda, bfloat16
import transformers
import gradio as gr

class VerizonAI:
    """ Class VerizonAI to create the chatbot object """
    def __init__(self, db_path):
        """
        Initialize the class object.
        """
        self.DB_FAISS_PATH = '/home/chirayu.tripathi/hackathon/vectorstore/db_faiss'

        self.custom_prompt_template = """[INST] You are a Verizon company's chatbot, Only use the following pieces of context to answer the user's question. If the answer is not present in context, just say that you don't know and display the following link "https://www.verizon.com/support/residential/contact-us/contactuslanding.htm", don't try to make up an answer.[/INST]

        Context: {context}
        Question: {question}
        answer: 
        """
        self.generate_text = self.get_pipeline()
        self.qa_result = self.qa_bot()
        
    def get_pipeline(self):
        # model_id = 'meta-llama/Llama-2-7b-chat-hf'
        model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        hf_auth = '' #not required for mistral, but required for llama-2
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )

        model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
        )

        print(f"Model loaded on {device}")

        generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        # stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
        )
        return generate_text
    
    def set_custom_prompt(self):
        """
        Prompt template for QA retrieval for each vectorstore
        """
        prompt = PromptTemplate(template=self.custom_prompt_template,
                                input_variables=['context', 'question'])
        return prompt



    def retrieval_qa_chain(self, llm, prompt, db):
        """ Create the QA retrieval chain from langchain. """
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 3}),#, search_type="mmr"),
                                           return_source_documents=True,
                                           # memory = memory
                                           chain_type_kwargs={'prompt': prompt}
                                           )

        return qa_chain
    
    def load_llm(self):
        """ Load the model using the huggingface pipeline. """
        llm = HuggingFacePipeline(pipeline=self.generate_text)
        return llm

    
    def qa_bot(self):
        """Setup the QA bot"""
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2",
                                           model_kwargs={'device': 'cuda'})
        db = FAISS.load_local(self.DB_FAISS_PATH, embeddings)
        llm = self.load_llm()
        qa_prompt = self.set_custom_prompt()
        qa = self.retrieval_qa_chain(llm, qa_prompt, db)

        return qa

    
    def final_result(self,message,history):
        """Function to generate the result"""
        response = self.qa_result({'query': message})
        return response['result']
    
if __name__ == "__main__":
    bot = VerizonAI(db_path = '/home/chirayu.tripathi/hackathon/vectorstore/db_faiss')
    gr.ChatInterface(bot.final_result).launch(share=True)
