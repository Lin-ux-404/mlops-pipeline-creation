# Azure Machine Learning Pipeline Creation

This repository provides examples of creating reusable components for Azure Machine Learning pipelines using two approaches: **Python function components** and **YAML-based components**. Both methods allow you to modularize your ML workflows, offering flexibility in definition, maintenance, and usage. 

> **Note:** This repository demonstrates basic pipeline deployment using simple steps such as data cleaning, model training, and model evaluation. It is intended as a reference and does not provide an end-to-end solution.

---

## Contents

The repository showcases two implementations of the same use case:

- **Pipeline with Python function components**
- **Pipeline with YAML-based components**

The goal is to demonstrate how to deploy pipelines to Azure Machine Learning Studio using either approach.

---

## How to Run

### Setup

1. **Environment File**
   Create a `.env` file in the root directory by copying `.env.sample` and populating it with the required values.

2. **Azure Resources**
   Ensure you have an Azure resource group containing the following services:

   - Azure Machine Learning Workspace
   - Storage Account (with key-based authentication enabled)

   If needed, you can create the resource group by running the `setup.ps1` script.

3. **Install Dependencies**
   It is recommended to use a virtual environment. Install the required packages with:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebooks**
   After setup, you can start executing the provided notebooks.

---

### Pipelines with Python Function Components

- **Components Definition**
  Python function components are located in the `components` folder. These are reusable, modular functions written in Python.

- **Environment Configuration**
  The `env.yaml` file contains the environment setup for the pipeline.

- **Pipeline Deployment**
  Use the `py_func_pipeline.ipynb` notebook to learn how to deploy pipelines using Python function components with the Azure ML SDK.

  **Reference:**
  [Python Function Components Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipeline-python?view=azureml-api-2)

---

### Pipelines with YAML-based Components

- **Components Definition**
  YAML-based components are defined in the `components` folder, with implementation code in the `src` folder.

- **Pipeline Deployment**
  Use the `yaml_pipeline_deployment.ipynb` notebook to learn how to deploy pipelines using YAML-based components with the Azure ML SDK.

  **Reference:**
  [YAML Components Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipelines-cli?view=azureml-api-2)

---

## Authors

- [Shenglin Xu](shenglinxu@microsoft.com)

---

## Disclaimer

This repository is functional as of the current version. However, compatibility with future updates to Azure or related services is not guaranteed.
