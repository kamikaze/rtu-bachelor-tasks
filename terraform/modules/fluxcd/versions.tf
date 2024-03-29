terraform {
  required_providers {
    flux = {
      source  = "fluxcd/flux"
      version = "~> 1.2.3"
    }

    github = {
      source  = "integrations/github"
      version = "~> 6.0.0"
    }
  }
}
