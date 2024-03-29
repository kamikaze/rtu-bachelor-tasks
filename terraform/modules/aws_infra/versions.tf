terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.38.0"
    }

    github = {
      source  = "integrations/github"
      version = "~> 6.0.0"
    }
  }
}
