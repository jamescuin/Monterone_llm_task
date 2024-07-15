# moneterone_llm_task

This project contains the code presented on 2024-07-15, for the LLM task, as outlined in the `LLM Exercise (Jul24).docx`. 

# Setup
The files in this repository require various dependencies and libraries to be installed in order to run successfully. It is recommended to set up a virtual environment and install the dependencies listed in the requirements.txt file. To set up a virtual environment and install the dependencies, follow these steps:

```
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

To run any agents that leverage the OpenAI API, you will need to specify your `OPENAI_API_KEY` in the `.env` file. At the time of writing, this requires a paid OpenAI account, with credits added.

# Stremalit Apps
To run the streamlit apps, simply execute `streamlit run streamlit_app_example.py` (OpenAI API) or `streamlit run streamlit_app_example_v2.py` (HuggingFace API) in your terminal, from the root of this project. 

# Contributing
Contributions to this repository are welcome! If you have any ideas, bug fixes, or improvements, feel free to open an issue or submit a pull request. Please explain the motivation for a given change, ensuring it is well documented and does not create duplication.

# License
This project falls under the MIT license.