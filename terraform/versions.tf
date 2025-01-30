terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.84.0"
    }

    flux = {
      source  = "fluxcd/flux"
      version = "~> 1.2.3"
    }

    github = {
      source  = "integrations/github"
      version = "~> 6.0.0"
    }

    null = {
      source  = "hashicorp/null"
      version = ">= 3.2.2"
    }
  }

  backend "s3" {
    bucket         = "tf-dev-infra-state"
    key            = "global/terraform.tfstate"
    region         = "eu-west-2"
    dynamodb_table = "tf_dev_infra_lockid"
  }

  required_version = ">= 1.10.0"
}
