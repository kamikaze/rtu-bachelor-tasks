output "ecr_frontend_url" {
  value = var.ecr_enabled ? aws_ecr_repository.frontend[0].repository_url : null
}

output "ecr_backend_url" {
  value = var.ecr_enabled ? aws_ecr_repository.backend[0].repository_url : null
}

output "node_ip" {
  value = var.ec2_enabled ? aws_spot_instance_request.node[0].public_ip : null
}

output "db_master_ip" {
  value = var.rds_enabled ? aws_db_instance.db_master[0].address : null
}

output "db_master_develop_ip" {
  value = var.rds_enabled ? aws_db_instance.db_master_develop[0].address : null
}

output "eks_cluster_name" {
  value = var.eks_enabled ? data.aws_eks_cluster.eks[0].name : null
}

output "eks_endpoint" {
  value = var.eks_enabled ? data.aws_eks_cluster.eks[0].endpoint : null
}

output "eks_certificate_authority" {
  value     = var.eks_enabled ? data.aws_eks_cluster.eks[0].certificate_authority[0].data : null
  sensitive = true
}

output "eks_token" {
  value     = var.eks_enabled ? data.aws_eks_cluster_auth.eks.token : null
  sensitive = true
}

output "external_secrets_access_key" {
  value = aws_iam_access_key.external_secrets.id
}

output "external_secrets_secret_key" {
  value = aws_iam_access_key.external_secrets.secret
}
