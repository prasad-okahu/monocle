# Azure environment constants
AZURE_ML_ENDPOINT_ENV_NAME = "AZUREML_ENTRY_SCRIPT"
AZURE_FUNCTION_WORKER_ENV_NAME = "FUNCTIONS_WORKER_RUNTIME"
AZURE_APP_SERVICE_ENV_NAME = "WEBSITE_SITE_NAME"
AWS_LAMBDA_ENV_NAME = "AWS_LAMBDA_RUNTIME_API"
GITHUB_CODESPAE_ENV_NAME = "CODESPACES"

# Azure naming reference can be found here
# https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/ready/azure-best-practices/resource-abbreviations
AZURE_FUNCTION_NAME = "azure.func"
AZURE_APP_SERVICE_NAME = "azure.asp"
AZURE_ML_SERVICE_NAME = "azure.mlw"
AWS_LAMBDA_SERVICE_NAME = "aws.lambda"
GITHUB_CODESPASE_SERVICE_NAME = "github.codespace"

azure_service_map = {
    AZURE_ML_ENDPOINT_ENV_NAME: AZURE_ML_SERVICE_NAME,
    AZURE_APP_SERVICE_ENV_NAME: AZURE_APP_SERVICE_NAME,
    AZURE_FUNCTION_WORKER_ENV_NAME: AZURE_FUNCTION_NAME,
    GITHUB_CODESPAE_ENV_NAME: GITHUB_CODESPASE_SERVICE_NAME
}

aws_service_map = {
    AWS_LAMBDA_ENV_NAME: AWS_LAMBDA_SERVICE_NAME
}

git_service_map = {
    GITHUB_CODESPAE_ENV_NAME: GITHUB_CODESPASE_SERVICE_NAME
}

service_maps = [azure_service_map, aws_service_map, git_service_map]