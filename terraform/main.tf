
module "aws_infra" {
  source = "./modules/aws_infra"

  organization              = var.organization
  project                   = var.project
  admin_ssh_public_key      = var.admin_ssh_public_key
  admin_ssh_public_key_name = var.admin_ssh_public_key_name
  aws_region                = var.aws_region
  s3_enabled                = var.s3_enabled
  ecr_enabled               = var.ecr_enabled
  rds_enabled               = var.rds_enabled
  db_name                   = var.project
  db_username               = var.db_username
  db_password               = var.db_password
  dataset_db_username       = var.dataset_db_username
  dataset_db_password       = var.dataset_db_password
  eks_enabled               = var.eks_enabled
  eks_version               = var.eks_version
  eks_cluster_name          = var.eks_cluster_name
  eks_node_instance_type    = var.eks_node_instance_type
  key_pair_name             = module.ssh_key_pair.key_pair_name
  dns_api_token             = var.dns_api_token
  dns_api_secret            = var.dns_api_secret
  github_token              = var.github_token
}

module "kubernetes" {
  count = var.eks_enabled ? 1 : 0
  depends_on = [
    module.aws_infra
  ]

  source                = "./modules/kubernetes"
  eks_cluster_name      = module.aws_infra.eks_cluster_name
  aws_access_key_id     = module.aws_infra.external_secrets_access_key
  aws_access_key_secret = module.aws_infra.external_secrets_secret_key
}

module "fluxcd" {
  count = var.fluxcd_enabled ? 1 : 0
  depends_on = [
    module.aws_infra,
    module.kubernetes
  ]

  source             = "./modules/fluxcd"
  github_org         = var.github_org
  github_token       = var.github_token
  github_repository  = var.github_repository
  git_infra_repo_url = var.git_infra_repo_url
  public_key_openssh = tls_private_key.flux.public_key_openssh
}
