provider "aws" {
  region = var.aws_region
}

provider "kubernetes" {
  host                   = module.aws_infra.eks_endpoint
  cluster_ca_certificate = base64decode(module.aws_infra.eks_certificate_authority)
  token                  = module.aws_infra.eks_token
}

provider "github" {
  owner = var.github_org
  token = var.github_token
}

provider "flux" {
  kubernetes = {
    host                   = module.aws_infra.eks_endpoint
    cluster_ca_certificate = base64decode(module.aws_infra.eks_certificate_authority)
    exec = {
      api_version = "client.authentication.k8s.io/v1beta1"
      args        = ["eks", "get-token", "--cluster-name", module.aws_infra.eks_cluster_name]
      command     = "aws"
    }
  }

  git = {
    url    = "ssh://git@github.com/${var.github_org}/${var.github_repository}.git"
    branch = "main"
    ssh = {
      username    = "git"
      private_key = tls_private_key.flux.private_key_pem
    }
  }
}

provider "null" {}
