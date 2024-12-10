#!/usr/bin/env pwsh

# Path to the .env file
$envFilePath = ".env"

# Load the .env file and set environment variables
Get-Content $envFilePath | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($name, $value, [System.EnvironmentVariableTarget]::Process)
    }
}

# Access and display a variable
Write-Output "MY_VARIABLE is: $env:RESOURCE_GROUP"

# Create the resource group and set as default
Write-Output "Create a resource group and set as default:"
az group create --name $env:RESOURCE_GROUP --location $env:REGION
az configure --defaults group=$env:RESOURCE_GROUP

# Create an Azure Machine Learning workspace
Write-Output "Create an Azure Machine Learning workspace:"
az ml workspace create --name $env:WORKSPACE_NAME
az configure --defaults workspace=$env:WORKSPACE_NAME

# Create compute instance
Write-Output "Creating a compute instance with name: $env:COMPUTE_INSTANCE"
az ml compute create --name $env:COMPUTE_INSTANCE --size STANDARD_DS11_V2 --type ComputeInstance

# Create compute cluster
Write-Output "Creating a compute cluster with name: $env:COMPUTE_CLUSTER"
az ml compute create --name $env:COMPUTE_CLUSTER --size STANDARD_DS11_V2 --max-instances 2 --type AmlCompute

# Create data assets (components)
# Write-Output "Creating components"
# az ml component create --file .yaml_based_pipeline/components/fix-missing-data.yml
# az ml component create --file .yaml_based_pipeline/components/normalize-data.yml
# az ml component create --file .yaml_based_pipeline/components/train-decision-tree.yml
# az ml component create --file .yaml_based_pipeline/components/train-logistic-regression.yml
