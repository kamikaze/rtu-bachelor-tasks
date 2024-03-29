resource "aws_ssm_parameter" "dev_aws_access_key_id" {
  name  = "dev-aws-access-key-id"
  type  = "SecureString"
  value = aws_iam_access_key.backend_service.id
}

resource "aws_ssm_parameter" "dev_aws_secret_access_key" {
  name  = "dev-aws-secret-access-key"
  type  = "SecureString"
  value = aws_iam_access_key.backend_service.secret
}


resource "aws_ssm_parameter" "dns_api_token" {
  name  = "dns-api-token"
  type  = "SecureString"
  value = var.dns_api_token
}

resource "aws_ssm_parameter" "dns_api_secret" {
  name  = "dns-api-secret"
  type  = "SecureString"
  value = var.dns_api_secret
}

locals {
  dockerconfig = {
    "auths" = {
      "ghcr.io" = {
        "username" = "_json_key",
        "password" = var.github_token,
        "auth"     = base64encode("_json_key:${var.github_token}")
      }
    }
  }
}

resource "aws_ssm_parameter" "container_registry_creds" {
  name  = "container-registry-creds"
  type  = "SecureString"
  value = jsonencode(local.dockerconfig)
}

resource "aws_ssm_parameter" "prod_db_dsn" {
  count = var.rds_enabled ? 1 : 0
  name  = "db-dsn"
  type  = "SecureString"
  value = "postgresql+asyncpg://${var.db_username}:${var.db_password}@${aws_db_instance.db_master[0].address}:5432/${var.db_name}"
}

resource "aws_ssm_parameter" "dev_db_dsn" {
  count = var.rds_enabled ? 1 : 0
  name  = "db-dsn-develop"
  type  = "SecureString"
  value = "postgresql+asyncpg://${var.db_username}:${var.db_password}@${aws_db_instance.db_master_develop[0].address}:5432/${var.db_name}"
}

locals {
  datasets_db_creds = {
    "password" = var.dataset_db_password
  }
}

resource "aws_ssm_parameter" "datasets_db_creds" {
  name  = "datasets-db-creds"
  type  = "SecureString"
  value = jsonencode(local.datasets_db_creds)
}


locals {
  ebs_csi_driver_creds = {
    "key_id"     = aws_iam_access_key.ebs_csi_driver.id,
    "access_key" = aws_iam_access_key.ebs_csi_driver.secret
  }
}

resource "aws_ssm_parameter" "ebs_csi_driver_creds" {
  name  = "ebs-csi-driver-creds"
  type  = "SecureString"
  value = jsonencode(local.ebs_csi_driver_creds)
}


locals {
  label_studio_aws_creds = {
    "access_key" = aws_iam_access_key.label_studio.id,
    "secret_key" = aws_iam_access_key.label_studio.secret
  }
}

resource "aws_ssm_parameter" "label_studio_aws_creds" {
  name  = "label-studio-aws-creds"
  type  = "SecureString"
  value = jsonencode(local.label_studio_aws_creds)
}
