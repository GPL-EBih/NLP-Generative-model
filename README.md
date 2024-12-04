# Vietnamese Poetry Generation with RAG (Retrieval-Augmented Generation)

This project aims to enhance the generation of Vietnamese poetry using Retrieval-Augmented Generation (RAG) techniques. The focus is on generating poetry that adheres to specific structural rules and generating contextually accurate content based on a large dataset of Vietnamese poems. The system integrates machine learning models and RAG to retrieve relevant data and improve the overall poetry generation process.

## Project Overview

This project explores the use of Retrieval-Augmented Generation (RAG) to generate Vietnamese poetry. The goal is to generate poems that follow specific forms and structures by leveraging the power of language models in conjunction with relevant poem retrieval. The project aims to:
- Generate Vietnamese poetry with accurate structures.
- Use RAG techniques to query a database of Vietnamese poems to improve the context of generated poems.
- Implement model training and data processing techniques to improve poetry generation quality.

### Key Features:
- **Poetry Generation**: The system generates various forms of Vietnamese poetry based on specific poetic structures.
- **RAG Integration**: Retrieval-augmented generation is used to query relevant poems and enhance the generation process.
- **Model Training**: Currently, the model is being trained on a dataset of 200,000+ Vietnamese poems to improve the generation of specific poetic structures.
- **Dataset**: The project uses a publicly available dataset of Vietnamese poems from Fsoft Lab.

## Technologies Used

- **Programming Languages**: Python, R
- **Libraries & Frameworks**:
  - **Machine Learning**: PyTorch, scikit-learn
  - **Natural Language Processing**: Huggingface, Transformer, Underthesea
  - **AI Models**: OpenAI GPT, NVIDIA, Meta models
  - **Data Processing**: Pandas, NumPy
- **Database**:
  - **Faiss** for indexing and querying the Vietnamese poem database.
- **Deployment**:
  - Flask for creating a web interface (currently under development).
  - Docker for containerization.

## Current Status

The project is still in the development phase with several key components implemented:

1. **Data Collection and Preprocessing**: A dataset of 200,000+ Vietnamese poems has been gathered and preprocessed to be used for training.
2. **Model Training**: The training of LSTM and Transformer models has been started to generate poems based on specific structures.
3. **RAG Implementation**: An initial version of the RAG system has been implemented using Faiss to index the poem dataset. However, the query retrieval process is still being optimized.
4. **API Integrations**: API calls from OpenAI, NVIDIA, and Meta have been integrated into the system to augment the generation process, though further integration is planned.
5. **Web Interface**: A basic Flask application has been developed, but it is still being improved for full deployment.

## Future Work

- Fine-tune models for better accuracy and context in poetry generation.
- Optimize the RAG query system to provide better results.
- Continue improving the web interface and make it fully functional.
- Expand the dataset with more diverse Vietnamese poems to increase the model's understanding of different poetic forms.
